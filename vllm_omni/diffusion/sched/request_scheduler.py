# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import fields
from typing import TYPE_CHECKING

from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.base_scheduler import BaseScheduler
from vllm_omni.diffusion.sched.interface import (
    DiffusionRequestStatus,
    DiffusionSchedulerOutput,
    RequestBatchSamplingParamsKey,
    _AdmissionWaitDecision,
)

if TYPE_CHECKING:
    from vllm_omni.diffusion.worker.utils import RunnerOutput

# LoRA identity is derived from `sampling.lora_request`, not a same-named field
# on sampling params, so it must be resolved separately from the bulk lookup.
_REQUEST_BATCH_SAMPLING_PARAMS_KEY_FIELD_NAMES = frozenset(
    field.name for field in fields(RequestBatchSamplingParamsKey)
) - {"condition_key", "flow_shift", "lora_int_id", "sample_solver"}


def _normalize_explicit_sample_solver(value: object | None) -> str | None:
    """Normalize an explicitly provided solver without selecting a default."""
    if value is None:
        return None
    return str(value).strip().lower()


def _normalize_explicit_flow_shift(value: object | None) -> float | None:
    """Normalize an explicitly provided flow shift without selecting a default."""
    if value is None:
        return None
    return float(value)


def build_request_batch_sampling_params_key(request: OmniDiffusionRequest) -> RequestBatchSamplingParamsKey:
    """Build the compatibility key shared by scheduling and DP dispatch."""
    sampling = request.sampling_params
    # LoRA identity is optional on sampling params (and on test stubs).
    lora_request = getattr(sampling, "lora_request", None)
    key_kwargs = {name: getattr(sampling, name) for name in _REQUEST_BATCH_SAMPLING_PARAMS_KEY_FIELD_NAMES}
    extra_args = sampling.extra_args or {}
    # Match pipeline resolution for explicit overrides, but preserve None:
    # pipeline/engine defaults are configuration-dependent and must not be
    # inferred while building the request-batch key.
    key_kwargs["sample_solver"] = _normalize_explicit_sample_solver(extra_args.get("sample_solver"))
    key_kwargs["flow_shift"] = _normalize_explicit_flow_shift(extra_args.get("flow_shift"))
    key_kwargs["condition_key"] = getattr(request, "batch_compatibility_key", None)
    key_kwargs["lora_int_id"] = lora_request.lora_int_id if lora_request is not None else None
    return RequestBatchSamplingParamsKey(**key_kwargs)


class RequestScheduler(BaseScheduler):
    """Scheduler for static request waves, including admission coalescing."""

    def get_admission_wait_decision(
        self,
        *,
        now: float,
        dp_concurrent: bool = False,
    ) -> _AdmissionWaitDecision:
        assert self.od_config is not None
        max_wait_ms = self.od_config.request_batch_max_wait_ms
        if max_wait_ms <= 0.0 or self.max_num_running_reqs <= 1:
            return _AdmissionWaitDecision(should_wait=False)
        if self.num_running_requests() > 0:
            return _AdmissionWaitDecision(should_wait=False)

        max_wait_s = max_wait_ms / 1000.0
        stable_window_s = min(0.3, max_wait_s / 2.0) if dp_concurrent else min(0.05, max_wait_s / 5.0)
        return _AdmissionWaitDecision(
            should_wait=True,
            deadline=now + max_wait_s,
            stable_window_s=stable_window_s,
            max_batch=self.max_num_running_reqs,
        )

    def should_end_admission_wait(
        self,
        decision: _AdmissionWaitDecision,
        *,
        now: float,
        stable_since: float,
    ) -> bool:
        waiting = self.num_waiting_requests()
        return (
            waiting >= decision.max_batch
            or (waiting > 0 and now - stable_since >= decision.stable_window_s)
            or (decision.deadline is not None and now >= decision.deadline)
        )

    def _build_sampling_params_key(self, request: OmniDiffusionRequest) -> RequestBatchSamplingParamsKey:
        return build_request_batch_sampling_params_key(request)

    def update_from_output(self, sched_output: DiffusionSchedulerOutput, output: RunnerOutput) -> set[str]:
        scheduled_request_ids = sched_output.scheduled_request_ids
        if not scheduled_request_ids:
            return set()

        terminal_statuses: dict[str, DiffusionRequestStatus] = {}
        terminal_errors: dict[str, str | None] = {}
        for request_id in scheduled_request_ids:
            state = self._request_states.get(request_id)
            if state is None or state.is_finished():
                continue
            req_output = output.get_request_output(request_id)
            result = req_output.result if req_output is not None else None
            if result is None:
                # Async mode: result=None with async_output_id means compute done,
                # final output will arrive later via wait_output_ready.
                if req_output is not None and req_output.async_output_id is not None:
                    terminal_statuses[request_id] = DiffusionRequestStatus.FINISHED_COMPLETED
                    terminal_errors[request_id] = None
                else:
                    terminal_statuses[request_id] = DiffusionRequestStatus.FINISHED_ERROR
                    terminal_errors[request_id] = "No output result"
            elif result.aborted:
                terminal_statuses[request_id] = DiffusionRequestStatus.FINISHED_ABORTED
                terminal_errors[request_id] = None
            elif result.error:
                terminal_statuses[request_id] = DiffusionRequestStatus.FINISHED_ERROR
                terminal_errors[request_id] = result.error
            else:
                terminal_statuses[request_id] = DiffusionRequestStatus.FINISHED_COMPLETED
                terminal_errors[request_id] = None

        return self._finalize_update_from_output(sched_output, terminal_statuses, terminal_errors)
