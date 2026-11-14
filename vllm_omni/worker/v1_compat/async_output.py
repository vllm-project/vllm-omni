# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Omni fork of vLLM V1-era worker data model, vendored from upstream
4ba787ec9959e566d3b587a796dff07673324da5 (vllm/v1/worker/gpu_model_runner.py
lines 269-370) because aac1b2a4f3 (#63373, Remove Model Runner V1) deleted it
in vLLM 0.34. Omni's MRv1 stage runners depend on its exact async-output
semantics. Non-import deviations from the source: none.
"""

from __future__ import annotations

import torch
from vllm import envs
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    get_ep_all2all_manager,
)
from vllm.v1.outputs import (
    AsyncModelRunnerOutput,
    KVConnectorOutput,
    LogprobsTensors,
    ModelRunnerOutput,
)
from vllm.v1.worker.utils import raise_if_nan_logits

from vllm_omni.worker.v1_compat.rejection_sampler import RejectionSampler


def nans_to_dict(counts: list[int], req_id_to_index: dict[str, int]) -> dict[str, int]:
    """Map per-row NaN counts onto request ids.

    Rows and requests need not correspond one-to-one, so requests without a
    row are reported as 0.
    """
    return {req_id: int(counts[i]) if i < len(counts) else 0 for req_id, i in req_id_to_index.items()}


class AsyncGPUModelRunnerOutput(AsyncModelRunnerOutput):
    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        sampled_token_ids: torch.Tensor,
        logprobs_tensors: LogprobsTensors | None,
        invalid_req_indices: list[int],
        async_output_copy_stream: torch.cuda.Stream,
        vocab_size: int,
        check_ep_fault: bool = False,
        num_nans: torch.Tensor | None = None,
    ):
        self._model_runner_output = model_runner_output
        self._invalid_req_indices = invalid_req_indices

        # Event on the copy stream so we can synchronize the non-blocking copy.
        # Blocking (sleep) event to avoid busy-polling the CUDA driver lock.
        self.async_copy_ready_event = torch.cuda.Event(blocking=True)

        # Keep a reference to the device tensor to avoid it being
        # deallocated until we finish copying it to the host.
        self._sampled_token_ids = sampled_token_ids
        self.vocab_size = vocab_size
        self._logprobs_tensors = logprobs_tensors
        self._num_nans = num_nans
        self._has_fault: torch.Tensor | None = None

        # Initiate the copy on a separate stream, but do not synchronize it.
        default_stream = torch.cuda.current_stream()
        with torch.cuda.stream(async_output_copy_stream):
            async_output_copy_stream.wait_stream(default_stream)
            self.sampled_token_ids_cpu = self._sampled_token_ids.to("cpu", non_blocking=True)
            self._logprobs_tensors_cpu = self._logprobs_tensors.to_cpu_nonblocking() if self._logprobs_tensors else None
            self._num_nans_cpu = self._num_nans.to("cpu", non_blocking=True) if self._num_nans is not None else None
            if check_ep_fault:
                has_fault = get_ep_all2all_manager().query_fault()
                self._has_fault = has_fault.to("cpu", non_blocking=True)
            self.async_copy_ready_event.record()

    def get_output(self) -> ModelRunnerOutput:
        """Copy the device tensors to the host and return a ModelRunnerOutput.

        This function blocks until the copy is finished.
        """
        max_gen_len = self.sampled_token_ids_cpu.shape[-1]
        self.async_copy_ready_event.synchronize()

        # Release the device tensors once the copy has completed.
        del self._logprobs_tensors
        del self._sampled_token_ids
        if max_gen_len == 1:
            valid_sampled_token_ids = self.sampled_token_ids_cpu.tolist()
            for i in self._invalid_req_indices:
                valid_sampled_token_ids[i].clear()
            logprobs_lists = None
            if self._logprobs_tensors_cpu is not None:
                logprobs_lists = self._logprobs_tensors_cpu.tolists()
        else:
            valid_sampled_token_ids, logprobs_lists = RejectionSampler.parse_output(
                self.sampled_token_ids_cpu,
                self.vocab_size,
                self._invalid_req_indices,
                logprobs_tensors=self._logprobs_tensors_cpu,
            )

        output = self._model_runner_output
        output.sampled_token_ids = valid_sampled_token_ids
        output.logprobs = logprobs_lists

        if self._num_nans_cpu is not None:
            output.num_nans_in_logits = nans_to_dict(self._num_nans_cpu.tolist(), output.req_id_to_index)
            if envs.VLLM_RAISE_ON_LOGIT_NANS:
                raise_if_nan_logits(output.num_nans_in_logits)
        del self._num_nans

        if self._has_fault is not None and self._has_fault.item():
            mask = get_ep_all2all_manager().query_active_mask()
            raise RuntimeError(
                "Fault detected in EP all2all communication: "
                "one or more ranks timed out during dispatch/combine. "
                f"Mask: {mask.cpu().tolist()}"
            )

        return output

    def get_kv_connector_output(self) -> KVConnectorOutput | None:
        """Get the KV connector output for this async output."""
        return self._model_runner_output.kv_connector_output
