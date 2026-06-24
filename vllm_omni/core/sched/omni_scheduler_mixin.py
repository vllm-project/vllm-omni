from __future__ import annotations

import os
import time
from collections.abc import Iterable
from typing import Any

from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.request import RequestStatus

from vllm_omni.core.sched.output import OmniChunkRecvHandle, OmniSchedulerOutput

logger = init_logger(__name__)

_STATS_INTERVAL_S = 1.0

# Upper bound on how long a request may sit in full-payload-input wait
# (the state ``OmniSchedulingCoordinator`` records via ``_waiting_since``)
# before the scheduler force-fails it.  Defends against stuck consumer-side
# requests when the producer drops a full-payload, send fails, or recv
# never arrives.  Override per-deployment via
# VLLM_OMNI_INPUT_WAIT_TIMEOUT_S; set <=0 to disable the safety net.
#
# Scope: this constant only covers full-payload ``WAITING_FOR_INPUT`` waits.
# Async-chunk requests park in ``WAITING_FOR_CHUNK`` and are intentionally not
# force-failed by this timeout during normal realtime pauses.
_INPUT_WAIT_TIMEOUT_RAW = os.environ.get("VLLM_OMNI_INPUT_WAIT_TIMEOUT_S", "300")
try:
    DEFAULT_INPUT_WAIT_TIMEOUT_S: float = float(_INPUT_WAIT_TIMEOUT_RAW)
except ValueError:
    logger.warning(
        "Invalid VLLM_OMNI_INPUT_WAIT_TIMEOUT_S=%r; falling back to 300 seconds.",
        _INPUT_WAIT_TIMEOUT_RAW,
    )
    DEFAULT_INPUT_WAIT_TIMEOUT_S = 300.0


class OmniSchedulerMixin:
    """Shared scheduler helpers for omni-specific request handling."""

    def _free_input_coordinator_request(self, request_id: str) -> None:
        """Prune full-payload coordinator state for a completed request."""
        input_coordinator = getattr(self, "input_coordinator", None)
        if input_coordinator is not None:
            input_coordinator.free_finished_request(request_id)

    # ------------------------------------------------------------------ #
    #  Shared scheduler/output helpers (lift the AR / generation duplicates)
    # ------------------------------------------------------------------ #

    def _consume_pending_connector_output(self, model_mode: str) -> None:
        """Drain ``self._latest_omni_connector_output`` into the coordinator.

        Called at the top of every ``schedule()`` cycle.  Identical between
        AR and generation schedulers except for the ``model_mode`` argument
        forwarded to ``update_request_metadata``.
        """
        connector_output = getattr(self, "_latest_omni_connector_output", None)
        self._latest_omni_connector_output = None
        input_coordinator = getattr(self, "input_coordinator", None)
        if input_coordinator is None:
            return
        if connector_output:
            chunk_ready_req_ids = connector_output.chunk_ready_req_ids
            chunk_finished_req_ids = connector_output.chunk_finished_req_ids
        else:
            chunk_ready_req_ids = set()
            chunk_finished_req_ids = set()
        if connector_output and connector_output.request_metadata:
            input_coordinator.update_request_metadata(
                self.requests, connector_output.request_metadata, model_mode=model_mode
            )
        if connector_output and connector_output.chunk_segment_finished_req_ids:
            pending_segment_stop = getattr(self, "_omni_pending_upstream_segment_finished", None)
            model_config = getattr(getattr(self, "vllm_config", None), "model_config", None)
            if pending_segment_stop is not None and getattr(model_config, "final_output", False):
                segment_finished_req_ids = set(connector_output.chunk_segment_finished_req_ids)
                pending_segment_stop.update(segment_finished_req_ids)
                chunk_ready_req_ids = set(chunk_ready_req_ids)
                chunk_ready_req_ids.update(segment_finished_req_ids)
        # Both calls self-guard on the coordinator's async_chunk mode
        # (process_pending_chunks returns early when async_chunk is False;
        # process_pending_full_payload_inputs branches internally), so exactly
        # one path is live per deployment.  With an empty async allowlist the
        # coordinator is never in async_chunk mode -> the chunk call is a no-op.
        input_coordinator.process_pending_chunks(
            self.waiting,
            self.running,
            chunk_ready_req_ids,
            chunk_finished_req_ids,
        )
        input_coordinator.process_pending_full_payload_inputs(
            self.waiting,
            self.running,
            connector_output.stage_recv_req_ids if connector_output else set(),
        )

    def _process_pending_input_timeouts(self) -> None:
        """Force-fail requests waiting on the full-payload coordinator too long.

        Called at the top of every ``schedule()`` cycle, right after
        ``_consume_pending_connector_output``.  Without this hook, a request
        whose producer dropped a payload would sit in the
        full-payload-input wait state indefinitely (the runner mixin
        protects ``_pending_load_reqs`` from prune sweeps).

        Reads ``_waiting_since`` timestamps maintained by the input
        coordinator and delegates to the base scheduler's
        ``finish_requests`` to mark expired requests FINISHED_ERROR.
        Disabled when ``DEFAULT_INPUT_WAIT_TIMEOUT_S`` is <= 0.

        Scope: only covers full-payload ``WAITING_FOR_INPUT`` requests.
        Async-chunk ``WAITING_FOR_CHUNK`` requests are not handled here, so
        normal realtime pauses are not bounded by this timeout.
        """
        if DEFAULT_INPUT_WAIT_TIMEOUT_S <= 0:
            return
        input_coordinator = getattr(self, "input_coordinator", None)
        if input_coordinator is None:
            return
        timed_out_ids = input_coordinator.collect_timed_out_request_ids(timeout_s=DEFAULT_INPUT_WAIT_TIMEOUT_S)
        if not timed_out_ids:
            return
        present_ids = {req_id for req_id in timed_out_ids if req_id in self.requests}
        if not present_ids:
            return
        logger.warning(
            "Marking %d request(s) as FINISHED_ERROR after waiting > %.0fs for connector input: %s",
            len(present_ids),
            DEFAULT_INPUT_WAIT_TIMEOUT_S,
            sorted(present_ids),
        )
        self.finish_requests(present_ids, RequestStatus.FINISHED_ERROR)

    def _capture_omni_connector_output(self, model_runner_output: Any) -> None:
        """Stash the model runner's omni_connector_output for next schedule().

        Called at the tail of every ``update_from_output()`` -- identical
        between AR and generation schedulers.  Only stashes the output;
        applying the metadata is the responsibility of
        ``_consume_pending_connector_output()`` at the start of the next
        ``schedule()`` cycle.  Applying it twice (once here, once on
        consume) is unsafe under ``update_request_metadata`` in
        generation mode, which resets ``prompt_token_ids`` /
        ``_output_token_ids`` / ``num_computed_tokens`` and would
        clobber any progress between the two calls.
        """
        omni_output = getattr(model_runner_output, "omni_connector_output", None)
        if omni_output is None:
            return
        self._latest_omni_connector_output = omni_output

    def _wrap_omni_scheduler_output(
        self,
        base: SchedulerOutput,
        *,
        finished_requests_needing_kv_transfer: dict | None = None,
        pending_connector_registrations: list[OmniChunkRecvHandle] | None = None,
    ) -> OmniSchedulerOutput:
        """Wrap a base ``SchedulerOutput`` in ``OmniSchedulerOutput``.

        Pulls each base ``SchedulerOutput`` dataclass field via ``getattr``
        and forwards optional omni-specific fields.  Lifted from 4 separate
        copy-pastes between AR (1) and generation (3) schedulers.
        """
        base_data = {name: getattr(base, name) for name in SchedulerOutput.__dataclass_fields__}
        input_coordinator = getattr(self, "input_coordinator", None)
        if pending_connector_registrations is None:
            pending_connector_registrations = (
                input_coordinator.pending_connector_registrations if input_coordinator else []
            )
        # Drain segment-finished ids recorded by update_from_output for resumable
        # realtime stops (coordinator stages only); the runner emits one
        # is_segment_finished terminal per id next cycle, mirroring finished_req_ids.
        pending_segment = getattr(self, "_omni_pending_segment_finished", None)
        if pending_segment:
            segment_finished_req_ids = set(pending_segment)
            pending_segment.clear()
        else:
            segment_finished_req_ids = set()
        return OmniSchedulerOutput(
            **base_data,
            finished_requests_needing_kv_transfer=finished_requests_needing_kv_transfer or {},
            pending_connector_registrations=pending_connector_registrations,
            segment_finished_req_ids=segment_finished_req_ids,
        )

    def make_stats(self, *args, **kwargs) -> SchedulerStats | None:
        now = time.monotonic()
        if now - getattr(self, "_last_stats_time", 0.0) < _STATS_INTERVAL_S:
            return None
        self._last_stats_time = now
        return super().make_stats(*args, **kwargs)

    def _realign_request_status_to_queues(
        self,
        request_ids: str | Iterable[str] | None,
    ) -> None:
        """Realign ``request.status`` to actual queue membership.

        ``OmniChunkTransferAdapter._process_chunk_queue`` stamps
        ``requests_origin_status[req.id] = WAITING`` (or ``RUNNING``) when
        first parking a request in a chunk-transfer deque. On the next
        tick, when the chunk arrives, ``_process_chunk_queue`` sets
        ``request.status = target_status`` and continues, but
        ``requests_origin_status`` is left at its first-park value -- no
        hook updates it on the ``waiting → running`` admit transition
        that ``super().schedule()`` later performs. The table stays
        stale until the request makes another deque round-trip.

        If an abort lands in the gap between admit and the next deque
        round-trip, ``chunk_transfer_adapter.finish_requests`` reads the
        stale ``WAITING`` from ``requests_origin_status``, stomps it
        onto ``request.status``, and the upstream
        ``Scheduler.finish_requests`` else branch silently fails to
        remove from ``self.running`` -- the request stays alive in
        ``self.running`` and the worker's ``input_batch`` slot leaks.
        After ``max_num_seqs`` such aborts every new request hangs at
        ``chunks=0`` until the client times out.

        Realign here: if a request lives in ``self.running`` but its
        status is not ``RUNNING``, set it to ``RUNNING``; symmetrically
        flip ``RUNNING → WAITING`` when the request is actually in
        ``self.waiting``. This is a localized safety net for
        ``requests_origin_status`` staleness on the admit transition;
        it does not touch the adapter's invariants and is complementary
        to the chunk-transfer-adapter deque purge that already runs
        inside ``process_pending_chunks`` / ``restore_queues``.

        Note on scope: only the ``async_chunk`` path actually triggers
        the ``requests_origin_status`` staleness this helper repairs.
        When ``async_chunk`` is disabled, no chunk-transfer round-trip
        occurs between admit and finish, so the realignment walk is a
        cheap O(n) no-op over an already-aligned set. The call is kept
        unconditional in ``finish_requests`` to (a) keep the abort path
        uniform and (b) defend any future configuration that re-enables
        chunk transfer from rediscovering the same regression.

        See https://github.com/vllm-project/vllm-omni/pull/3774 and the
        residual-hang reproduction discussed in that PR.
        """
        # Mirror the upstream Scheduler.finish_requests resolution of
        # ``request_ids`` so realignment touches exactly the set that
        # ``super().finish_requests`` will then walk.
        if isinstance(request_ids, str):
            ids_to_align: Iterable[str] = (request_ids,)
        elif request_ids is None:
            ids_to_align = list(self.requests.keys())
        else:
            ids_to_align = list(request_ids)

        if not ids_to_align:
            return

        running_ids = {r.request_id for r in self.running}
        waiting_ids = {r.request_id for r in self.waiting}

        for rid in ids_to_align:
            req = self.requests.get(rid)
            if req is None or req.is_finished():
                continue
            if rid in running_ids and req.status != RequestStatus.RUNNING:
                req.status = RequestStatus.RUNNING
            elif rid in waiting_ids and req.status == RequestStatus.RUNNING:
                req.status = RequestStatus.WAITING

    def _purge_finished_from_running(self) -> None:
        """Defensive post-finish sweep of ``self.running``.

        Belt-and-suspenders to ``_realign_request_status_to_queues``:
        even after status realignment lets upstream
        ``Scheduler.finish_requests`` pick the right removal branch,
        a future regression or an unexpected ``status`` mid-transition
        could still leave already-finished entries in ``self.running``.
        Sweeping here guarantees the worker's ``input_batch`` slot is
        not pinned by a freed request.

        Complementary to ``_realign_request_status_to_queues``: realign
        is preventive (fix ``status`` before ``super().finish_requests``
        so the right branch fires); this purge is defensive (sweep the
        residue after ``super().finish_requests`` so any stale entries
        are reclaimed).

        Scope of the predicate. ``is_finished()`` covers entries the
        upstream ``finish_requests`` already drained from ``self.requests``
        but failed to remove from ``self.running``; the
        ``request_id not in self.requests`` arm catches the same surface
        from a different angle and is the post-cleanup mirror of the
        deque purge ``_purge_untracked_chunk_requests`` already runs at
        the chunk-transfer-adapter layer. It does **not** by itself make
        arbitrary direct deletions of ``self.requests`` safe -- callers
        that pop ``self.requests`` outside the standard finish path
        still have to go through ``_free_request`` (or equivalent) for
        block / connector / coordinator cleanup. This sweep only
        reclaims the ``self.running`` slot reference.

        In-place via ``self.running[:] = ...`` for minor consistency
        with idiomatic vLLM scheduler mutation; upstream
        ``Scheduler.finish_requests`` itself rebinds ``self.running``,
        so list identity across the whole call is not preserved -- the
        slice form is just to avoid an extra rebind inside this helper.

        Assumes the upstream V1 invariant that scheduler ticks are
        serialized on a single thread; in-place mutation here is no more
        racy than the rest of the scheduler under that assumption.

        See https://github.com/vllm-project/vllm-omni/pull/3774
        discussion.
        """
        if not self.running:
            return
        self.running[:] = [req for req in self.running if not req.is_finished() and req.request_id in self.requests]
