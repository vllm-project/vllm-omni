# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import json
from dataclasses import fields, replace
from typing import TYPE_CHECKING

from vllm_omni.diffusion.data import uses_rank_local_dp_concurrency
from vllm_omni.diffusion.offloader.config import TEXT_ENCODER_COMPONENT, resolve_offload
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.base_scheduler import BaseScheduler
from vllm_omni.diffusion.sched.interface import (
    DiffusionRequestStatus,
    DiffusionSchedulerOutput,
    RequestBatchSamplingParamsKey,
    SchedulerRequestState,
    _AdmissionWaitDecision,
)

if TYPE_CHECKING:
    from vllm_omni.diffusion.worker.utils import BaseRunnerOutput

# Derived and request-owned fields must be resolved separately from the bulk
# sampling-param lookup.
_REQUEST_BATCH_SAMPLING_PARAMS_KEY_FIELD_NAMES = frozenset(
    field.name for field in fields(RequestBatchSamplingParamsKey)
) - {
    "condition_key",
    "flow_shift",
    "lora_int_id",
    "negative_conditioning",
    "rank_local_dp_extra_args_signature",
    "sample_solver",
    "text_encoder_input_signature",
}


def _normalize_explicit_sample_solver(value: object | None) -> str | None:
    """Normalize an explicitly provided solver without selecting a default."""
    if value is None:
        return None
    return str(value).strip().lower()


def _normalize_explicit_flow_shift(value: float | str | None) -> float | None:
    """Normalize an explicitly provided flow shift without selecting a default."""
    if value is None:
        return None
    return float(value)


def build_rank_local_dp_extra_args_signature(request: OmniDiffusionRequest) -> str:
    """Compare full extra_args for admission and dispatch of collective waves."""
    return json.dumps(getattr(request.sampling_params, "extra_args", None), sort_keys=True, default=repr)


def is_empty_dp_prompt(prompt: object) -> bool:
    """Return whether a DP request has no text, tokens, or prompt embeddings."""
    if prompt is None:
        return True
    if isinstance(prompt, (str, list, tuple)):
        return not prompt
    if isinstance(prompt, dict):
        return (
            not prompt.get("prompt")
            and not prompt.get("prompt_token_ids")
            and not prompt.get("prompt_ids")
            and prompt.get("prompt_embeds") is None
        )
    return False


def text_encoder_input_signature(prompt: object) -> tuple[bool, bool]:
    """Describe precomputed embeddings that change encoder forward counts."""
    if not isinstance(prompt, dict):
        return False, False
    return prompt.get("prompt_embeds") is not None, prompt.get("negative_prompt_embeds") is not None


def uses_text_encoder_allgather(config: object) -> bool:
    resolved = resolve_offload(config)
    return resolved.offloads(TEXT_ENCODER_COMPONENT) and resolved.uses_allgather(TEXT_ENCODER_COMPONENT)


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
    prompt = request.prompt
    if isinstance(prompt, dict):
        # Pipelines can enable true CFG from negative inputs even when the
        # generic CFG flag is false. Empty negative text is still present.
        # Compare field presence without inspecting tensor values or inferring
        # model-specific guidance defaults here.
        key_kwargs["negative_conditioning"] = frozenset(
            name for name, value in prompt.items() if name.startswith("negative_") and value is not None
        )
    key_kwargs["condition_key"] = getattr(request, "batch_compatibility_key", None)
    key_kwargs["lora_int_id"] = lora_request.lora_int_id if lora_request is not None else None
    return RequestBatchSamplingParamsKey(**key_kwargs)


class RequestScheduler(BaseScheduler):
    """Scheduler for static request waves, including admission coalescing."""

    def _make_request_state(self, request_id: str, request: OmniDiffusionRequest) -> SchedulerRequestState:
        # Inspect prompt structure before queueing, where invalid input only
        # fails this submission. Empty prompts retain the serial worker path.
        requires_single_request = uses_rank_local_dp_concurrency(self.od_config) and is_empty_dp_prompt(request.prompt)
        state = super()._make_request_state(request_id, request)
        state.requires_single_request = requires_single_request
        return state

    def _can_schedule_waiting(self, state: SchedulerRequestState) -> bool:
        if not super()._can_schedule_waiting(state):
            return False
        if not self._running:
            return True
        current_state = self._request_states.get(self._running[0])
        if current_state is None or state.requires_single_request or current_state.requires_single_request:
            return False
        manager = self._diffusion_kv_manager
        if manager is None or not manager.has_request(state.request_id):
            return True
        # The request-level runner uses one first-step query boundary per wave.
        # The Manager already aligns all CFG sequences within each request.
        current = manager.get_metadata(self._running[0]).sequences[0].cached_prefix_len
        candidate = manager.get_metadata(state.request_id).sequences[0].cached_prefix_len
        return candidate == current

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
        key = build_request_batch_sampling_params_key(request)
        if uses_rank_local_dp_concurrency(self.od_config):
            # Serialize once before the request enters the queue. Invalid
            # extra_args then fail this submission, rather than schedule()
            # raising and shutting down the engine with other requests active.
            key = replace(
                key,
                rank_local_dp_extra_args_signature=build_rank_local_dp_extra_args_signature(request),
                text_encoder_input_signature=(
                    text_encoder_input_signature(request.prompt)
                    if uses_text_encoder_allgather(self.od_config)
                    else None
                ),
            )
        return key

    def update_from_output(self, sched_output: DiffusionSchedulerOutput, output: BaseRunnerOutput) -> set[str]:
        scheduled_request_ids = sched_output.scheduled_request_ids
        if not scheduled_request_ids and not sched_output.finished_req_ids:
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
