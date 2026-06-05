from __future__ import annotations

import os
import time
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
# Scope: this constant only covers the full-payload coordinator path
# (``input_coordinator``).  The async-chunk path uses
# ``chunk_transfer_adapter`` and is not affected by this constant.
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
        if connector_output and connector_output.request_metadata:
            input_coordinator.update_request_metadata(
                self.requests, connector_output.request_metadata, model_mode=model_mode
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

        Scope: only covers ``input_coordinator`` (full-payload path).
        Async-chunk requests park in ``chunk_transfer_adapter`` instead
        and are not handled here -- if a similar safety net is needed
        for the chunk path, it belongs in the chunk adapter.
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
        pending_input_registrations: list[OmniChunkRecvHandle] | None = None,
    ) -> OmniSchedulerOutput:
        """Wrap a base ``SchedulerOutput`` in ``OmniSchedulerOutput``.

        Pulls each base ``SchedulerOutput`` dataclass field via ``getattr``
        and forwards optional omni-specific fields.  Lifted from 4 separate
        copy-pastes between AR (1) and generation (3) schedulers.
        """
        base_data = {name: getattr(base, name) for name in SchedulerOutput.__dataclass_fields__}
        input_coordinator = getattr(self, "input_coordinator", None)
        if pending_input_registrations is None:
            pending_input_registrations = input_coordinator.pending_input_registrations if input_coordinator else []
        return OmniSchedulerOutput(
            **base_data,
            finished_requests_needing_kv_transfer=finished_requests_needing_kv_transfer or {},
            pending_input_registrations=pending_input_registrations,
        )

    def make_stats(self, *args, **kwargs) -> SchedulerStats | None:
        now = time.monotonic()
        if now - getattr(self, "_last_stats_time", 0.0) < _STATS_INTERVAL_S:
            return None
        self._last_stats_time = now
        return super().make_stats(*args, **kwargs)
