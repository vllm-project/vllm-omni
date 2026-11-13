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
  manager appends a batch at its session position when that position
  equals the stored length (exact continuation); a batch starting
  behind the stored length first content-checks its overlapping
  prefix, then appends only its not-yet-stored tail, and
  a batch starting ahead (a gap) is still refused — so a stream
  always describes one token sequence from position 0. A turn that
  starts with a pool hit instead seeds the stream first: materialize
  copies the pool-hit prefix [0, hit_end) into the store, so the
  suffix append lands on exact continuation (see ``_seed_session_rows``).
- Coarse invalidation. ``invalidate`` drops a session's segments
  wholesale. After it, the stored length resets to 0 and the next
  append starts a fresh stream; a later turn whose hit length is
  nonzero rebuilds the stream from its pool-hit prefix when the pool
  covers it, and only stalls when the pool does not cover the prefix.
  ``append_beyond`` also invalidates: a batch overlapping stored rows
  is content-checked (within tolerance) against them, and a mismatch
  means the session forked, so the whole session is dropped rather
  than serving stale rows later.
- Per-session copies, never shared rows. Segments belong to exactly one
  session, so dropping a session can never corrupt another session's
  hits — the hazard that rules out namespacing the slot pool itself.
- Bounded host memory. ``max_bytes`` caps stored plus reserved bytes; an
  append or reservation that would exceed it is refused (the caller falls
  back to the slot pool), never partially applied. ``reserve_pending``
  covers rows the manager cloned and queued for an out-of-order append,
  so the queue cannot grow past the budget either.

Thread safety: every method takes the store's own lock. The manager
additionally serializes its own tables with ``_state_lock``; lock order
is always manager lock, then store lock, and the store never calls back
into the manager.
"""

from __future__ import annotations

import logging
import threading
import time

import torch

logger = logging.getLogger(__name__)

SessionId = str
ReqId = str
TensorName = str

# Fork-detection tolerance for append_beyond's overlap check.
# UNMEASURED starting default, not a validated threshold: bf16 rows
# recomputed in different batch shapes differ by a few ULPs, so the check
# must tolerate low-bit noise or it would false-positive normal
# continuation. A too-tight bound costs cache hits (fail-closed
# invalidation), never correctness; validate against real model outputs
# in different batch shapes before relying on these values.
_CONTENT_RTOL = 1e-2
_CONTENT_ATOL = 1e-2


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
        # session -> history generation (DuplexFence.epoch) recorded by
        # the manager on STARTED. The store never infers an epoch: only
        # the manager writes it, and only drop() forgets it — an
        # invalidate keeps the generation (the session continues under
        # the same generation with a fresh stream).
        self._epochs: dict[SessionId, int] = {}
        # session -> monotonic seconds of its last activity (bind, append,
        # serve, invalidate). The reaper drops sessions idle longer than
        # the configured TTL; drop() forgets the record.
        self._last_touch: dict[SessionId, float] = {}
        # Sessions already warned once about a budget-refused append;
        # cleared on drop/invalidate so a refilled session can warn again.
        self._refusal_warned: set[SessionId] = set()
        self._stored_bytes = 0
        # Bytes reserved for rows the manager cloned and queued for a later
        # out-of-order append (see reserve_pending). Counted against the
        # budget so a long turn of queued rows cannot grow unbounded.
        self._pending_bytes = 0
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
            self._touch_locked(session_id)

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

    def set_epoch(self, session_id: SessionId, epoch: int) -> None:
        """Record the history generation a session's rows belong to.

        Low-level writer: production advances the generation through
        :meth:`compare_and_set_epoch` (STARTED, see
        ``OmniPrefixCacheManager._check_session_epoch``), which pairs the
        write with invalidation. The store never infers an epoch itself.
        """
        with self._lock:
            self._epochs[session_id] = int(epoch)

    def epoch_of(self, session_id: SessionId) -> int | None:
        """History generation recorded for a session, or None when unknown."""
        with self._lock:
            return self._epochs.get(session_id)

    def compare_and_set_epoch(self, session_id: SessionId, epoch: int) -> int | None:
        """Atomically advance a session's generation, dropping stale rows.

        The manager calls this on STARTED. Invalidation and the epoch write
        must be one lock hold: doing them as separate ``invalidate`` /
        ``set_epoch`` calls leaves a window where an in-flight append sees
        the old epoch, stores rows, and only then does the epoch advance.
        Those rows would sit under the new generation and be servable.

        A first-seen session only records the epoch. Returns the previous
        epoch when the generation changed (segments invalidated), else
        None.
        """
        with self._lock:
            known = self._epochs.get(session_id)
            if known is None:
                self._epochs[session_id] = int(epoch)
                return None
            if known == epoch:
                return None
            self._invalidate_locked(session_id)
            self._epochs[session_id] = int(epoch)
            return known

    def _epoch_matches_locked(self, session_id: SessionId, expected_epoch: int | None) -> bool:
        """True when an append for ``expected_epoch`` still belongs here.

        Caller holds ``_lock``. ``expected_epoch`` None (a pre-epoch payload)
        and an unknown stored epoch both mean "no gate", matching the
        manager's ``_session_epoch_mismatch``. A known, different epoch means
        the step predates the session's current generation: its rows must
        not enter the new stream.
        """
        if expected_epoch is None:
            return True
        current = self._epochs.get(session_id)
        return current is None or current == int(expected_epoch)

    def _touch_locked(self, session_id: SessionId) -> None:
        """Mark a session live; the caller must hold ``_lock``."""
        self._last_touch[session_id] = time.monotonic()

    # ------------------------------------------------------------ stream

    def stored_tokens(self, session_id: SessionId, key: TensorName) -> int:
        """Tokens stored for (session, key); 0 when nothing is stored."""
        with self._lock:
            return self._lengths.get((session_id, key), 0)

    def stored_bytes(self) -> int:
        """Total bytes held across all sessions."""
        with self._lock:
            return self._stored_bytes

    def pending_bytes(self) -> int:
        """Bytes reserved for queued appends that have not landed yet."""
        with self._lock:
            return self._pending_bytes

    def reserve_pending(self, session_id: SessionId, nbytes: int) -> bool:
        """Reserve budget for rows that will be appended later.

        The manager clones out-of-order rows and queues them until the gap
        ahead fills. Without a reservation those clones would be invisible
        to ``max_bytes``; counting them here keeps the bound honest. Returns
        False (nothing reserved) when the budget would be exceeded, and the
        caller drops the rows exactly as a refused append.
        """
        with self._lock:
            if not self._within_budget_locked(nbytes):
                self._refuse_append_locked(session_id)
                return False
            self._pending_bytes += int(nbytes)
            return True

    def release_pending(self, nbytes: int) -> None:
        """Release a reservation made by :meth:`reserve_pending`."""
        with self._lock:
            self._pending_bytes = max(0, self._pending_bytes - int(nbytes))

    def _within_budget_locked(self, nbytes: int) -> bool:
        """True when ``nbytes`` more fit; the caller must hold ``_lock``.

        Stored and reserved-but-unappended bytes share one budget, so a
        queue of out-of-order rows can never exceed ``max_bytes``.
        """
        return self._stored_bytes + self._pending_bytes + nbytes <= self._max_bytes

    def _refuse_append_locked(self, session_id: SessionId) -> None:
        """Count and log a budget refusal; the caller must hold ``_lock``.

        The first refusal per session warns (budget, stored, pending); later
        refusals drop to debug until the flag clears on drop/invalidate.
        """
        self.refused_appends += 1
        if session_id not in self._refusal_warned:
            self._refusal_warned.add(session_id)
            logger.warning(
                "session %s append refused: session store budget %s bytes would be "
                "exceeded (stored %s bytes, pending %s bytes); falling back to the slot pool",
                session_id,
                self._max_bytes,
                self._stored_bytes,
                self._pending_bytes,
            )
        else:
            logger.debug(
                "session %s append refused again (stored %s, pending %s of %s bytes)",
                session_id,
                self._stored_bytes,
                self._pending_bytes,
                self._max_bytes,
            )

    def append(self, session_id: SessionId, key: TensorName, rows: torch.Tensor) -> bool:
        """Append ``rows`` at the end of the (session, key) stream.

        Returns False (without storing anything) when the byte budget
        would be exceeded. Raises ValueError when ``rows`` is not a
        per-token tensor or its trailing shape disagrees with the stream.
        The rows are cloned: callers may hand over views into reusable
        step buffers.
        """
        return self._append(session_id, key, rows, expected_length=None, expected_epoch=None)

    def append_if_length(
        self,
        session_id: SessionId,
        key: TensorName,
        expected_length: int,
        rows: torch.Tensor,
        *,
        expected_epoch: int | None = None,
    ) -> bool:
        """Append ``rows`` only when the stored length still equals
        ``expected_length``, atomically under the store lock.

        A safe form for the manager's continuation rule (see also
        ``append_beyond``): a plain length check followed by ``append``
        lets a concurrent ``invalidate`` slip between the two and land
        the rows at the head of a fresh stream, where a later turn
        could serve them as its prefix. ``expected_epoch`` closes the
        matching TOCTOU against an epoch advance: the generation check
        and the append happen under the same lock hold. A failed check
        returns False and stores nothing.
        """
        return self._append(
            session_id, key, rows, expected_length=expected_length, expected_epoch=expected_epoch
        )

    def append_beyond(
        self,
        session_id: SessionId,
        key: TensorName,
        batch_start: int,
        rows: torch.Tensor,
        *,
        expected_epoch: int | None = None,
    ) -> bool:
        """Append the not-yet-stored tail of a batch starting at ``batch_start``.

        ``rows`` covers [batch_start, batch_start + len(rows)). Atomically
        under the store lock:
        - batch fully covered (batch end <= stored): the batch is
          content-checked against the stored rows at the same positions;
          a match is a no-op returning True, a mismatch means the session
          forked, so the whole session is invalidated and False is
          returned (nothing is stored);
        - batch overlaps the stored prefix (stored in [batch_start,
          batch end)): the overlapping prefix is content-checked the
          same way; on a match only rows[stored - batch_start:] appends,
          so a block-truncated pool hit (offset < stored) no longer
          rejects the whole batch and freezes the stream; on a mismatch
          the session is invalidated and nothing is stored;
        - batch starts ahead of the stored length (a gap): returns False
          and stores nothing — the missing middle cannot be filled from
          this batch, so the stream stalls exactly as before.
        The manager ignores the return value: the real brake is the
        invalidation itself (lengths reset to 0, so later same-round
        batches are gap-refused and the next round reseeds). False is
        only the truthful "nothing stored" signal.
        ``expected_epoch`` refuses the batch (without touching the
        session) when the session's generation moved on since the step was
        published, closing the window between the manager's epoch check
        and this append.
        Raises ValueError for a non-tensor/empty batch, a negative
        batch_start, or a trailing-shape mismatch with the stream.
        """
        if not isinstance(rows, torch.Tensor) or rows.ndim < 2 or rows.shape[0] == 0:
            raise ValueError(f"session store append needs a non-empty per-token tensor, got {type(rows)}")
        if batch_start < 0:
            raise ValueError(f"session store append_beyond needs batch_start >= 0, got {batch_start}")
        entry = (session_id, key)
        with self._lock:
            if not self._epoch_matches_locked(session_id, expected_epoch):
                return False
            self._touch_locked(session_id)
            stored = self._lengths.get(entry, 0)
            batch_end = batch_start + int(rows.shape[0])
            if batch_end <= stored:
                if not self._overlap_matches_locked(entry, batch_start, batch_end, rows):
                    self._invalidate_locked(session_id)
                    return False
                return True
            if stored < batch_start:
                return False
            if batch_start < stored and not self._overlap_matches_locked(
                entry, batch_start, stored, rows[: stored - batch_start]
            ):
                self._invalidate_locked(session_id)
                return False
            return self._append_locked(session_id, key, rows[stored - batch_start :], expected_length=stored)

    def _slice_locked(self, entry: tuple[SessionId, TensorName], start: int, end: int) -> torch.Tensor | None:
        """Concatenated stored rows for [start, end); caller holds ``_lock``.

        Cross-segment: the window may span several segments. Returns None
        when the window is not fully covered (callers treat that as a
        content mismatch: fail closed, never serve or keep stale rows).
        """
        segments = self._segments.get(entry)
        if segments is None:
            return None
        parts: list[torch.Tensor] = []
        pos = 0
        for seg in segments:
            seg_end = pos + int(seg.shape[0])
            if seg_end <= start:
                pos = seg_end
                continue
            if pos >= end:
                break
            parts.append(seg[max(start - pos, 0) : min(end - pos, int(seg.shape[0]))])
            pos = seg_end
        if sum(int(p.shape[0]) for p in parts) != end - start:
            return None
        return torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]

    def _overlap_matches_locked(
        self, entry: tuple[SessionId, TensorName], start: int, end: int, rows: torch.Tensor
    ) -> bool:
        """True when ``rows`` matches stored [start, end) within tolerance.

        Caller holds ``_lock``. Comparison is fail-closed: any shape
        mismatch, device/dtype error, or non-finite difference counts as
        a mismatch (the session is invalidated) rather than risking a
        stale stream. Tolerance is approximate on purpose: the same
        tokens recomputed in different batch shapes differ by low-bit
        bf16 noise, and exact equality would false-positive normal
        continuation.
        """
        if end <= start:
            return True
        try:
            stored = self._slice_locked(entry, start, end)
            if stored is None or tuple(stored.shape) != tuple(rows.shape):
                return False
            return bool(
                torch.allclose(
                    stored.detach().cpu(),
                    rows.detach().cpu(),
                    rtol=_CONTENT_RTOL,
                    atol=_CONTENT_ATOL,
                )
            )
        except (RuntimeError, ValueError, TypeError):
            return False

    def _append(
        self,
        session_id: SessionId,
        key: TensorName,
        rows: torch.Tensor,
        *,
        expected_length: int | None,
        expected_epoch: int | None,
    ) -> bool:
        if not isinstance(rows, torch.Tensor) or rows.ndim < 2 or rows.shape[0] == 0:
            raise ValueError(f"session store append needs a non-empty per-token tensor, got {type(rows)}")
        with self._lock:
            if not self._epoch_matches_locked(session_id, expected_epoch):
                return False
            self._touch_locked(session_id)
            return self._append_locked(session_id, key, rows, expected_length=expected_length)

    def _append_locked(
        self,
        session_id: SessionId,
        key: TensorName,
        rows: torch.Tensor,
        *,
        expected_length: int | None,
    ) -> bool:
        """Append body; the caller must hold ``_lock`` (``_append`` / ``append_beyond``)."""
        entry = (session_id, key)
        if expected_length is not None and self._lengths.get(entry, 0) != expected_length:
            return False
        segments = self._segments.get(entry)
        if segments is not None and tuple(segments[0].shape[1:]) != tuple(rows.shape[1:]):
            raise ValueError(
                f"session store shape mismatch for session={session_id} key={key}: "
                f"stream {tuple(segments[0].shape[1:])} vs rows {tuple(rows.shape[1:])}"
            )
        nbytes = rows.numel() * rows.element_size()
        if not self._within_budget_locked(nbytes):
            self._refuse_append_locked(session_id)
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
            self._touch_locked(session_id)
            return parts

    def snapshot_range(self, session_id: SessionId, key: TensorName, start: int, end: int) -> list[torch.Tensor] | None:
        """Segment views covering [start, end), or None when not fully covered.

        Like :meth:`snapshot` but for an arbitrary window, not just a
        prefix: a replay hit [read_start, hit_upto) must serve exactly
        its span, not the [0, hit_upto) head (A5/B1). The store is
        append-only and contiguous from 0, so covering [start, end)
        is the same length check as the prefix case (``length >= end``);
        the window only changes which segment views are returned.
        """
        if end <= start or start < 0:
            return None
        with self._lock:
            segments = self._segments.get((session_id, key))
            if segments is None or self._lengths.get((session_id, key), 0) < end:
                return None
            parts: list[torch.Tensor] = []
            pos = 0
            for seg in segments:
                seg_len = int(seg.shape[0])
                seg_end = pos + seg_len
                if seg_end <= start:
                    pos = seg_end
                    continue
                if pos >= end:
                    break
                parts.append(seg[max(start - pos, 0) : min(end - pos, seg_len)])
                pos = seg_end
            if sum(int(p.shape[0]) for p in parts) != end - start:
                return None
            self._touch_locked(session_id)
            return parts

    # -------------------------------------------------------- lifecycle

    def invalidate(self, session_id: SessionId) -> int:
        """Drop all of a session's segments (coarse invalidation).

        Bindings are kept: the session continues and later appends
        rebuild a fresh stream. Returns the number of segments dropped.
        """
        with self._lock:
            return self._invalidate_locked(session_id)

    def _invalidate_locked(self, session_id: SessionId) -> int:
        """Invalidate body; the caller must hold ``_lock``.

        Separate from ``invalidate`` because ``append_beyond`` already
        holds the lock when a fork is detected, and the lock is
        non-reentrant: calling ``invalidate`` there would deadlock.
        The recorded epoch is generation identity, not stream content,
        so an invalidate keeps it; only ``drop`` forgets it.
        """
        dropped = 0
        # The session continues with a fresh stream, so it is alive
        # again — and a refilled stream may warn about the budget again.
        self._touch_locked(session_id)
        self._refusal_warned.discard(session_id)
        for entry in [e for e in self._segments if e[0] == session_id]:
            segments = self._segments.pop(entry)
            self._lengths.pop(entry, None)
            dropped += len(segments)
            self._stored_bytes -= sum(s.numel() * s.element_size() for s in segments)
        return dropped

    def drop(self, session_id: SessionId) -> int:
        """End a session: invalidate its segments, unbind its requests,
        and forget its recorded epoch and last-touch record.

        Invalidation and bookkeeping run under one lock hold, atomically
        with respect to appends. Doing them in two separate lock holds
        would let an ``append_beyond`` land in the window: it stores
        segments and touches the session, then ``_drop_locked`` removes the
        touch record, leaving segments that no reaper iterates but that
        still count against ``max_bytes`` until restart.
        """
        with self._lock:
            dropped = self._invalidate_locked(session_id)
            self._drop_locked(session_id)
        return dropped

    def _drop_locked(self, session_id: SessionId) -> None:
        """Drop body; the caller must hold ``_lock`` (``drop`` /
        ``drop_idle_sessions``). ``invalidate`` already ran: this only
        forgets the session's bookkeeping and unbinds its requests."""
        self._epochs.pop(session_id, None)
        self._last_touch.pop(session_id, None)
        self._refusal_warned.discard(session_id)
        for req_id in [r for r, s in self._bindings.items() if s == session_id]:
            del self._bindings[req_id]
            self._resumable.pop(req_id, None)

    def drop_idle_sessions(self, now: float, ttl_s: float) -> list[SessionId]:
        """Drop sessions idle longer than ``ttl_s``; return those dropped.

        A session is idle when no touch (bind, append, serve,
        invalidate) happened within ``ttl_s``. The reaper is the only
        worker-visible session-end signal for ephemeral sessions, whose
        termination only unbinds. Scan and drops run atomically under
        the store lock, so a touch racing the reap lands after the drop
        instead of resurrecting a half-dropped session. (The invalidate
        below touches, and ``_drop_locked`` immediately forgets that
        touch again — net effect is no touch record, as intended.)
        """
        with self._lock:
            idle = [s for s, t in self._last_touch.items() if now - t > ttl_s]
            for session_id in idle:
                self._invalidate_locked(session_id)
                self._drop_locked(session_id)
        return idle

    def session_ids(self) -> list[SessionId]:
        """Sessions with at least one stored segment."""
        with self._lock:
            return sorted({entry[0] for entry in self._segments})
