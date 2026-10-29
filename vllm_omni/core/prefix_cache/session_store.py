# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Session-scoped prefix store for the omni prefix cache.

The slot-mirrored pool (``PrefixBlockPool``) is keyed by KV block id: any
request's traffic can hand a block to a new tenant, so a live duplex
session's prefix rows can be evicted by unrelated load. This store is the
session-level counterpart: per-session, token-ordered segments of
immediate-key rows (hidden + non-deferred mm), independent of block-pool
eviction. A session's rows survive until the session itself mutates its
history (coarse invalidation) or ends — never because another request
reused a block.

Ownership rules, all enforced here so callers cannot misalign rows:

- Append-only, contiguous streams. ``append`` adds rows at the current
  end of the (session, key) stream; there is no positional write. The
  manager only appends when the turn's hit length equals the stored
  length (exact continuation), so a stream always describes one token
  sequence from position 0.
- Coarse invalidation. ``invalidate`` drops a session's segments
  wholesale. After it, the stored length resets to 0 and the next
  append starts a fresh stream; a turn whose hit length is nonzero
  after an invalidation is refused by the manager's continuation rule
  until a full-prefix turn rebuilds the stream.
- Per-session copies, never shared rows. Segments belong to exactly one
  session, so dropping a session can never corrupt another session's
  hits — the hazard that rules out namespacing the slot pool itself.
- Bounded host memory. ``max_bytes`` caps total stored bytes; an append
  that would exceed it is refused (the caller falls back to the slot
  pool), never partially applied.

Thread safety: every method takes the store's own lock. The manager
additionally serializes its own tables with ``_state_lock``; lock order
is always manager lock, then store lock, and the store never calls back
into the manager.
"""

from __future__ import annotations

import threading

import torch

SessionId = str
ReqId = str
TensorName = str


class SessionPrefixStore:
    """Per-session token-ordered segments of cached prefix rows."""

    def __init__(self, max_bytes: int = 2 * 1024 * 1024 * 1024) -> None:
        self._max_bytes = int(max_bytes)
        self._lock = threading.Lock()
        # (session, key) -> list of row segments in token order.
        self._segments: dict[tuple[SessionId, TensorName], list[torch.Tensor]] = {}
        # (session, key) -> tokens stored (sum of segment first dims).
        self._lengths: dict[tuple[SessionId, TensorName], int] = {}
        # req -> session, for session-end cleanup of bindings.
        self._bindings: dict[ReqId, SessionId] = {}
        # req -> whether the bound request is a long-lived resumable
        # duplex stage request (vs a one-shot ephemeral request).
        self._resumable: dict[ReqId, bool] = {}
        self._stored_bytes = 0
        self.refused_appends = 0

    # ---------------------------------------------------------- bindings

    def bind(self, req_id: ReqId, session_id: SessionId, *, resumable: bool = False) -> None:
        """Record that ``req_id`` belongs to ``session_id``.

        ``resumable`` marks a long-lived duplex stage request: its
        termination is the session-end signal (see :meth:`terminate`).
        """
        with self._lock:
            self._bindings[req_id] = session_id
            self._resumable[req_id] = resumable

    def unbind(self, req_id: ReqId) -> None:
        """Drop a request's binding. Segments are untouched: a finished
        turn's rows still belong to its session's stream."""
        with self._lock:
            self._bindings.pop(req_id, None)
            self._resumable.pop(req_id, None)

    def terminate(self, req_id: ReqId) -> SessionId | None:
        """End a request's binding; return the session dropped, if any.

        A resumable duplex stage request is resumed across turns, so its
        FINISHED/ABORTED is the worker-visible session-end signal: the
        session's rows are private to it and can never serve again, so
        the whole session is dropped. An ephemeral request's termination
        only unbinds; the session continues and its rows stay servable.
        """
        with self._lock:
            session_id = self._bindings.pop(req_id, None)
            resumable = self._resumable.pop(req_id, False)
        if session_id is None or not resumable:
            return None
        self.drop(session_id)
        return session_id

    def session_of(self, req_id: ReqId) -> SessionId | None:
        """Session a request belongs to, or None when unbound."""
        with self._lock:
            return self._bindings.get(req_id)

    # ------------------------------------------------------------ stream

    def stored_tokens(self, session_id: SessionId, key: TensorName) -> int:
        """Tokens stored for (session, key); 0 when nothing is stored."""
        with self._lock:
            return self._lengths.get((session_id, key), 0)

    def stored_bytes(self) -> int:
        """Total bytes held across all sessions."""
        with self._lock:
            return self._stored_bytes

    def append(self, session_id: SessionId, key: TensorName, rows: torch.Tensor) -> bool:
        """Append ``rows`` at the end of the (session, key) stream.

        Returns False (without storing anything) when the byte budget
        would be exceeded. Raises ValueError when ``rows`` is not a
        per-token tensor or its trailing shape disagrees with the stream.
        The rows are cloned: callers may hand over views into reusable
        step buffers.
        """
        return self._append(session_id, key, rows, expected_length=None)

    def append_if_length(
        self, session_id: SessionId, key: TensorName, expected_length: int, rows: torch.Tensor
    ) -> bool:
        """Append ``rows`` only when the stored length still equals
        ``expected_length``, atomically under the store lock.

        This is the only safe form for the manager's continuation rule:
        a plain length check followed by ``append`` lets a concurrent
        ``invalidate`` slip between the two and land the rows at the
        head of a fresh stream, where a later turn could serve them as
        its prefix. A failed check returns False and stores nothing.
        """
        return self._append(session_id, key, rows, expected_length=expected_length)

    def _append(
        self,
        session_id: SessionId,
        key: TensorName,
        rows: torch.Tensor,
        *,
        expected_length: int | None,
    ) -> bool:
        if not isinstance(rows, torch.Tensor) or rows.ndim < 2 or rows.shape[0] == 0:
            raise ValueError(f"session store append needs a non-empty per-token tensor, got {type(rows)}")
        entry = (session_id, key)
        with self._lock:
            if expected_length is not None and self._lengths.get(entry, 0) != expected_length:
                return False
            segments = self._segments.get(entry)
            if segments is not None and tuple(segments[0].shape[1:]) != tuple(rows.shape[1:]):
                raise ValueError(
                    f"session store shape mismatch for session={session_id} key={key}: "
                    f"stream {tuple(segments[0].shape[1:])} vs rows {tuple(rows.shape[1:])}"
                )
            nbytes = rows.numel() * rows.element_size()
            if self._stored_bytes + nbytes > self._max_bytes:
                self.refused_appends += 1
                return False
            stored = rows.detach().clone()
            if segments is None:
                self._segments[entry] = [stored]
                self._lengths[entry] = 0
            else:
                segments.append(stored)
            self._lengths[entry] += int(rows.shape[0])
            self._stored_bytes += nbytes
            return True

    def covers(self, session_id: SessionId, key: TensorName, upto: int) -> bool:
        """True when the stream holds a contiguous [0, upto) prefix.

        Length arithmetic only — no tensor work, so the manager can call
        it on the prefetch path without moving data.
        """
        if upto <= 0:
            return False
        with self._lock:
            return self._lengths.get((session_id, key), 0) >= upto

    def fetch(self, session_id: SessionId, key: TensorName, upto: int) -> torch.Tensor | None:
        """Concatenated [0, upto) rows, or None when not fully covered.

        The result is a fresh tensor (``torch.cat`` copies), safe for
        callers to write into.
        """
        parts = self.snapshot(session_id, key, upto)
        if parts is None:
            return None
        return torch.cat(parts, dim=0) if len(parts) > 1 else parts[0].clone()

    def snapshot(self, session_id: SessionId, key: TensorName, upto: int) -> list[torch.Tensor] | None:
        """Segment views covering [0, upto), or None when not fully covered.

        Unlike ``fetch``, this copies nothing: callers concatenate
        unlocked. The returned views keep their segments alive, so a
        concurrent ``invalidate`` cannot corrupt a snapshot already
        taken — it only stops later snapshots from covering the range.
        """
        if upto <= 0:
            return None
        with self._lock:
            segments = self._segments.get((session_id, key))
            if segments is None or self._lengths.get((session_id, key), 0) < upto:
                return None
            parts: list[torch.Tensor] = []
            remaining = upto
            for seg in segments:
                if remaining <= 0:
                    break
                take = min(int(seg.shape[0]), remaining)
                parts.append(seg[:take])
                remaining -= take
            if remaining > 0:
                return None
            return parts

    # -------------------------------------------------------- lifecycle

    def invalidate(self, session_id: SessionId) -> int:
        """Drop all of a session's segments (coarse invalidation).

        Bindings are kept: the session continues and later appends
        rebuild a fresh stream. Returns the number of segments dropped.
        """
        with self._lock:
            dropped = 0
            for entry in [e for e in self._segments if e[0] == session_id]:
                segments = self._segments.pop(entry)
                self._lengths.pop(entry, None)
                dropped += len(segments)
                self._stored_bytes -= sum(s.numel() * s.element_size() for s in segments)
            return dropped

    def drop(self, session_id: SessionId) -> int:
        """End a session: invalidate its segments and unbind its requests."""
        dropped = self.invalidate(session_id)
        with self._lock:
            for req_id in [r for r, s in self._bindings.items() if s == session_id]:
                del self._bindings[req_id]
                self._resumable.pop(req_id, None)
        return dropped

    def session_ids(self) -> list[SessionId]:
        """Sessions with at least one stored segment."""
        with self._lock:
            return sorted({entry[0] for entry in self._segments})
