# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Torch-free session-progress control plane for disaggregated diffusion (RFC #4590 Part A).

A disaggregated AR-diffusion model (e.g. DreamZero) splits one logical forward
across three worker processes: ``encode -> denoise -> decode``. The per-session
AR window cursor (``current_start_frame``) must advance consistently across those
process boundaries, but each worker keeps its own process-local model state and
there is no denoise->encode back-channel in the whole-request topology.

This module is the small, model-agnostic **control plane** that keeps session
progress coherent without transporting the (large, model-private) AR-Diffusion KV
or the model state. It is intentionally torch-free so the ordering/idempotency
logic can be unit-tested on CPU without the model runtime.

Two roles use one coordinator class, on their own process:

* **Encode** (issuer): calls :meth:`SessionProgressCoordinator.issue` to stamp the
  outgoing carrier with ``(session_epoch, sequence_no)`` and to advance its own
  view of the window. Encode is NOT the commit authority — issuing a sequence
  does not mean it is committed.
* **Denoise** (authority): calls :meth:`SessionProgressCoordinator.authorize` to
  validate an incoming carrier against the committed session state (rejecting
  stale, duplicate, gapped, or wrong-epoch requests), then :meth:`commit` only
  after the DiT loop + KV commit succeed. A failed request never commits, so the
  same sequence can be retried; a duplicate/stale sequence is rejected clearly.

Design notes / invariants (RFC #4590 Part A):

1. Denoise owns committed progress (``last_committed_sequence`` /
   ``current_start_frame``); encode's ``next_sequence`` is only its issue cursor.
2. A chunk advances committed progress only via :meth:`commit`, called after a
   successful denoise — never on failure (invariants 4, 5).
3. A duplicate/stale sequence is rejected before any KV/model mutation
   (invariants 6, 7): the denoise runner raises before touching the pool.
4. ``epoch`` fences an explicit/session reset: a pre-reset in-flight attempt
   carries the old epoch and is rejected once the session advances (invariant 8).
5. ``sequence_no`` is monotonic per ``(session, epoch)`` and independent of
   ``current_start_frame`` (which resets to 0 at an AR window boundary WITHOUT an
   epoch bump), so window slides do not look like stale requests (invariant 7 vs
   the window-boundary case).
6. Sessions are tracked independently, keyed by ``session_id``, so interleaved
   sessions progress without sharing state (invariant 10).

The coordinator is LRU-bounded (``max_sessions``) so session-id churn in a
long-running server cannot grow the progress map without bound; eviction only
drops the small progress record (no pool blocks live here).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum


class ProgressDecision(str, Enum):
    """Outcome of :meth:`SessionProgressCoordinator.authorize`."""

    #: The request is the expected next sequence for the current epoch — run it.
    PROCEED = "proceed"
    #: The sequence is already committed (duplicate retry of a committed chunk).
    DUPLICATE = "duplicate"
    #: The sequence is older than the next expected one (out-of-order / replayed).
    STALE = "stale"
    #: The sequence skips ahead of the next expected one (a gap — lost request).
    GAP = "gap"
    #: The carrier's epoch predates the committed epoch (a pre-reset attempt).
    EPOCH_STALE = "epoch_stale"

    @property
    def ok(self) -> bool:
        return self is ProgressDecision.PROCEED


@dataclass
class DiffusionSessionProgress:
    """Committed + issue state for one disaggregated diffusion session.

    ``last_committed_sequence`` and ``current_start_frame`` are the authoritative
    committed position (owned by the denoise stage). ``next_sequence`` is the
    encode stage's issue cursor. Both sides use the same record type on their own
    process; the fields each side mutates differ by role.
    """

    session_id: str
    epoch: int = 0
    #: Highest sequence_no committed by denoise; -1 means nothing committed yet.
    last_committed_sequence: int = -1
    #: Next sequence_no encode will issue for this session/epoch.
    next_sequence: int = 0
    #: Committed AR window cursor (start frame of the next chunk).
    current_start_frame: int = 0
    #: attempt/request id of the last committed chunk (diagnostics + retry detect).
    last_attempt_id: str | None = None
    #: idle | in_flight | committed | reset — coarse lifecycle for observability.
    status: str = "idle"


class SessionProgressError(ValueError):
    """Raised when a stage request violates session-progress ordering.

    A subclass of ``ValueError`` (like ``StagePayloadError``) so the disaggregated
    runner can surface it as a clear, request-scoped transport/ordering error.
    """


class SessionProgressCoordinator:
    """LRU-bounded per-session progress records shared by one worker's requests.

    One instance lives per pipeline (per process). The encode pipeline uses it as
    an issue cursor; the denoise pipeline uses it as the commit authority. Because
    the two run in different processes they hold independent coordinators — the
    ``(epoch, sequence_no)`` stamped on the carrier is what reconciles them.
    """

    def __init__(self, max_sessions: int = 64) -> None:
        if max_sessions < 1:
            raise ValueError(f"max_sessions must be >= 1, got {max_sessions}.")
        self._max_sessions = int(max_sessions)
        self._sessions: OrderedDict[str, DiffusionSessionProgress] = OrderedDict()

    # -- lookup --------------------------------------------------------------

    def get(self, session_id: str) -> DiffusionSessionProgress:
        """Return (creating if needed) the progress record for ``session_id``."""
        key = str(session_id)
        record = self._sessions.get(key)
        if record is None:
            record = DiffusionSessionProgress(session_id=key)
            self._sessions[key] = record
            self._evict_if_needed()
        else:
            self._sessions.move_to_end(key)
        return record

    def peek(self, session_id: str) -> DiffusionSessionProgress | None:
        """Return the record for ``session_id`` without creating one."""
        return self._sessions.get(str(session_id))

    def _evict_if_needed(self) -> None:
        while len(self._sessions) > self._max_sessions:
            self._sessions.popitem(last=False)

    def drop(self, session_id: str) -> None:
        """Forget a session entirely (e.g. after a fatal teardown)."""
        self._sessions.pop(str(session_id), None)

    # -- encode side (issuer) ------------------------------------------------

    def issue(self, session_id: str) -> tuple[int, int]:
        """Reserve and return ``(epoch, sequence_no)`` for the next encode chunk.

        Advances the issue cursor so the next call returns the following sequence.
        Encode stamps the returned pair on the outgoing carrier. This does NOT
        commit anything (encode is not the authority); it only tracks what encode
        has handed downstream so its own window view stays ordered.
        """
        record = self.get(session_id)
        seq = record.next_sequence
        record.next_sequence = seq + 1
        record.status = "in_flight"
        return record.epoch, seq

    def begin_epoch_reset(self, session_id: str) -> int:
        """Bump the epoch and restart sequencing for an explicit/session reset.

        Called by the issuing (encode) stage when it decides a session reset, so
        the next issued chunk starts a fresh epoch at sequence 0. Returns the new
        epoch, which encode stamps on the carrier so the denoise authority can
        fence any older in-flight attempts.
        """
        record = self.get(session_id)
        record.epoch += 1
        record.next_sequence = 0
        record.last_committed_sequence = -1
        record.current_start_frame = 0
        record.last_attempt_id = None
        record.status = "reset"
        return record.epoch

    # -- denoise side (authority) --------------------------------------------

    def authorize(
        self,
        session_id: str,
        *,
        epoch: int,
        sequence_no: int,
        attempt_id: str | None = None,
    ) -> ProgressDecision:
        """Validate an incoming carrier against the committed session state.

        Returns a :class:`ProgressDecision`; the denoise runner runs the DiT only
        on :attr:`ProgressDecision.PROCEED` and raises on any other outcome. This
        never mutates committed progress — the caller commits separately on
        success. If the carrier carries a newer epoch than the authority has seen
        (an encode-driven reset), the authority adopts it first (see
        :meth:`adopt_epoch`) so a post-reset sequence 0 is accepted.
        """
        record = self.get(session_id)

        # Encode-driven reset: the carrier's epoch is ahead of ours. Adopt it so
        # the fresh epoch's sequence 0 authorizes cleanly. A carrier epoch OLDER
        # than committed is a stale pre-reset attempt -> fence it.
        if epoch > record.epoch:
            self.adopt_epoch(session_id, epoch)
            record = self.get(session_id)
        elif epoch < record.epoch:
            return ProgressDecision.EPOCH_STALE

        expected = record.last_committed_sequence + 1
        if sequence_no == expected:
            return ProgressDecision.PROCEED
        if sequence_no == record.last_committed_sequence:
            # Re-submit of the sequence that was just committed (a duplicate retry,
            # possibly with a different attempt_id). Must NOT append KV or advance
            # again -> caller rejects clearly (idempotent "do not double-apply").
            return ProgressDecision.DUPLICATE
        if sequence_no < record.last_committed_sequence:
            # An even older, already-superseded sequence (out-of-order/replayed).
            return ProgressDecision.STALE
        return ProgressDecision.GAP

    def adopt_epoch(self, session_id: str, epoch: int) -> None:
        """Advance the committed epoch to ``epoch`` (encode-driven reset).

        Resets committed sequence/window for the new epoch. Idempotent when the
        epoch already matches; refuses to move the epoch backwards.
        """
        record = self.get(session_id)
        if epoch < record.epoch:
            raise SessionProgressError(
                f"session {session_id!r}: cannot adopt older epoch {epoch} (committed epoch is {record.epoch})."
            )
        if epoch == record.epoch:
            return
        record.epoch = epoch
        record.last_committed_sequence = -1
        record.current_start_frame = 0
        record.last_attempt_id = None
        record.status = "reset"

    def commit(
        self,
        session_id: str,
        *,
        epoch: int,
        sequence_no: int,
        current_start_frame: int,
        attempt_id: str | None = None,
    ) -> DiffusionSessionProgress:
        """Advance committed progress after a successful denoise + KV commit.

        Only the expected next sequence for the current epoch may commit; anything
        else is a programming error (the caller must :meth:`authorize` first and
        run only on ``PROCEED``). Records the post-chunk window cursor so the next
        request's authoritative start position is available on this worker.
        """
        record = self.get(session_id)
        if epoch != record.epoch:
            raise SessionProgressError(
                f"session {session_id!r}: commit epoch {epoch} != committed epoch {record.epoch}."
            )
        expected = record.last_committed_sequence + 1
        if sequence_no != expected:
            raise SessionProgressError(
                f"session {session_id!r}: commit sequence {sequence_no} != expected {expected} "
                f"(last_committed={record.last_committed_sequence}); authorize() must gate commit()."
            )
        record.last_committed_sequence = sequence_no
        record.current_start_frame = int(current_start_frame)
        record.last_attempt_id = attempt_id
        record.status = "committed"
        # Keep the issue cursor at least one past the committed sequence so an
        # authority that also issues (monolithic-like reuse) never re-issues a
        # committed sequence.
        if record.next_sequence <= sequence_no:
            record.next_sequence = sequence_no + 1
        return record
