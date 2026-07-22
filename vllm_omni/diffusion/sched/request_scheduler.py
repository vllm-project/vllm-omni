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
)

if TYPE_CHECKING:
    from vllm_omni.diffusion.worker.utils import RunnerOutput

# LoRA identity is derived from `sampling.lora_request`, not a same-named field
# on sampling params, so it must be resolved separately from the bulk lookup.
_REQUEST_BATCH_SAMPLING_PARAMS_KEY_FIELD_NAMES = frozenset(
    field.name for field in fields(RequestBatchSamplingParamsKey)
) - {"lora_int_id"}


class RequestScheduler(BaseScheduler):
    """Diffusion scheduler with vLLM-style waiting/running queues."""

    def _build_sampling_params_key(self, request: OmniDiffusionRequest) -> RequestBatchSamplingParamsKey:
        """Build a request-batch compatibility key from sampling parameters."""
        sampling = request.sampling_params
        # LoRA identity is optional on sampling params (and on test stubs).
        lora_request = getattr(sampling, "lora_request", None)
        key_kwargs = {name: getattr(sampling, name) for name in _REQUEST_BATCH_SAMPLING_PARAMS_KEY_FIELD_NAMES}
        key_kwargs["lora_int_id"] = lora_request.lora_int_id if lora_request is not None else None
        return RequestBatchSamplingParamsKey(**key_kwargs)

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
