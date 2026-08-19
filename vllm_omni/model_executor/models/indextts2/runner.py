# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""IndexTTS continuous S2Mel generation worker and model runner."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext

import torch
from vllm.distributed.kv_transfer import has_kv_transfer_group
from vllm.distributed.parallel_state import get_pp_group
from vllm.forward_context import set_forward_context
from vllm.sequence import IntermediateTensors
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.outputs import AsyncModelRunnerOutput
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.worker.gpu_model_runner import AsyncGPUModelRunnerOutput

from vllm_omni.outputs import OmniModelRunnerOutput
from vllm_omni.utils.mm_outputs import partition_payload_list
from vllm_omni.worker.gpu_ar_model_runner import ExecuteModelState, _ensure_tensor_values
from vllm_omni.worker.gpu_generation_model_runner import GPUGenerationModelRunner
from vllm_omni.worker.gpu_generation_worker import GPUGenerationWorker


class IndexTTS2GenerationModelRunner(GPUGenerationModelRunner):
    """Advance recurrent CFM requests without manufacturing token/KV work."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._generation_finished_req_ids: set[str] = set()
        self._stepwise_output_req_ids: list[str] | None = None
        self._stepwise_empty_input_ids: torch.Tensor | None = None
        self._active_stepwise_req_ids: list[str] | None = None

    def _build_model_kwargs_extra(self) -> dict:
        request_ids = self._active_stepwise_req_ids
        if request_ids is None:
            return super()._build_model_kwargs_extra()
        request_infos = [self.model_intermediate_buffer[request_id] for request_id in request_ids]
        return {
            "model_intermediate_buffer": request_infos,
            "runtime_additional_information": request_infos,
        }

    def _collect_finished_request_ids(self) -> None:
        take_finished = getattr(self.model, "take_finished_request_ids", None)
        if callable(take_finished):
            self._generation_finished_req_ids = set(take_finished())

    def _execute_scheduled_model_work(
        self,
        scheduler_output: SchedulerOutput,
        deferred_state_corrections_fn,
    ):
        finished_req_ids = set(scheduler_output.finished_req_ids)
        new_req_ids = {request.req_id for request in scheduler_output.scheduled_new_reqs}
        stepwise_req_ids = list(getattr(scheduler_output, "stepwise_req_ids", ()))
        if finished_req_ids:
            stepwise_req_ids = [
                request_id
                for request_id in stepwise_req_ids
                if request_id not in finished_req_ids or request_id in new_req_ids
            ]

        flush_finished_requests = getattr(self.model, "flush_finished_requests", None)
        if not stepwise_req_ids:
            if callable(flush_finished_requests):
                flush_finished_requests()
            return NotImplemented

        if finished_req_ids and callable(flush_finished_requests):
            # Same-ID resubmissions must not inherit the old CFM state.
            flush_finished_requests()
        if not getattr(self.model, "requires_request_ids", False):
            raise RuntimeError("Stepwise scheduler output requires a stateful model that consumes request_ids")
        return self._execute_stepwise_generation(
            scheduler_output,
            stepwise_req_ids,
            deferred_state_corrections_fn,
        )

    def _execute_stepwise_generation(
        self,
        scheduler_output: SchedulerOutput,
        request_ids: list[str],
        deferred_state_corrections_fn,
    ) -> OmniModelRunnerOutput | None:
        self._sync_local_stage_payloads()
        missing_payloads = [
            request_id for request_id in request_ids if request_id not in self.model_intermediate_buffer
        ]
        if missing_payloads:
            raise RuntimeError(f"Zero-token stepwise requests have no resident payload: {missing_payloads}")

        empty_input_ids = getattr(self, "_stepwise_empty_input_ids", None)
        if empty_input_ids is None or empty_input_ids.device != self.device:
            empty_input_ids = torch.empty(0, dtype=torch.long, device=self.device)
            self._stepwise_empty_input_ids = empty_input_ids

        has_kv_transfer = has_kv_transfer_group()
        forward_context = set_forward_context(None, self.vllm_config) if has_kv_transfer else nullcontext()
        with (
            record_function_or_nullcontext("Forward"),
            forward_context,
            self.maybe_get_kv_connector_output(
                scheduler_output,
                defer_finalize=self.speculative_config is not None,
            ) as kv_connector_output,
        ):
            self._active_stepwise_req_ids = request_ids
            try:
                outputs = self._model_forward(
                    input_ids=empty_input_ids,
                    positions=None,
                    intermediate_tensors=None,
                    inputs_embeds=None,
                    request_ids=request_ids,
                )
            finally:
                self._active_stepwise_req_ids = None
            self._collect_finished_request_ids()

        _, multimodal_outputs = self.extract_multimodal_outputs(outputs)
        self._stepwise_output_req_ids = list(request_ids)
        self.execute_model_state = ExecuteModelState(
            scheduler_output,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            multimodal_outputs,
            None,
        )
        self.kv_connector_output = kv_connector_output
        if deferred_state_corrections_fn:
            deferred_state_corrections_fn()
        if scheduler_output.total_num_scheduled_tokens == 0:
            return self.sample_tokens()
        return None

    @staticmethod
    def _build_stepwise_payloads(
        multimodal_outputs_raw: object,
        num_reqs: int,
    ) -> list[dict[str, object] | None]:
        """Serialize IndexTTS's sparse completed-audio rows."""
        if multimodal_outputs_raw is None:
            return [None] * num_reqs
        if not isinstance(multimodal_outputs_raw, Mapping):
            raise RuntimeError("IndexTTS stepwise generation must return a mapping or None")

        per_req_payloads: list[dict[str, object] | None] = []
        for index in range(num_reqs):
            payload: dict[str, object] = {}
            for key, output in multimodal_outputs_raw.items():
                if isinstance(output, list):
                    if len(output) != num_reqs:
                        raise ValueError(
                            f"IndexTTS output list for key '{key}' has length {len(output)} but expected {num_reqs}"
                        )
                    item = output[index]
                else:
                    item = output
                if isinstance(item, torch.Tensor):
                    payload[key] = item.detach().to("cpu").contiguous()
            tensor_payload = _ensure_tensor_values(payload)
            per_req_payloads.append(tensor_payload or None)
        return per_req_payloads

    @torch.inference_mode()
    def sample_tokens(
        self,
        grammar_output: GrammarOutput | None = None,
    ) -> OmniModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors:
        if self._stepwise_output_req_ids is None:
            return super().sample_tokens(grammar_output)

        kv_connector_output = self.kv_connector_output
        self.kv_connector_output = None
        if self.execute_model_state is None:
            if self.use_async_scheduling and not get_pp_group().is_last_rank:
                self._pp_receive_prev_sampled_token_ids_to_input_batch()
            self._stepwise_output_req_ids = None
            return self.attach_omni_connector_output(
                OmniModelRunnerOutput.with_kv_conn_output_only(kv_connector_output)
            )

        (
            scheduler_output,
            _logits,
            _spec_decode_metadata,
            _spec_decode_common_attn_metadata,
            _hidden_states,
            _hidden_states_cpu,
            _sample_hidden_states,
            _aux_hidden_states,
            ec_connector_output,
            cudagraph_stats,
            multimodal_outputs_raw,
            _slot_mappings,
        ) = self.execute_model_state
        self.execute_model_state = None

        if self.speculative_config is not None:
            self.finalize_kv_connector()

        req_ids = list(self._stepwise_output_req_ids)
        per_req_payloads = self._build_stepwise_payloads(
            multimodal_outputs_raw,
            len(req_ids),
        )
        if self._async_chunk:
            inter_stage_outputs, multimodal_outputs = partition_payload_list(
                [payload or {} for payload in per_req_payloads]
            )
        elif all(payload is None for payload in per_req_payloads):
            inter_stage_outputs = None
            multimodal_outputs = None
        else:
            inter_stage_outputs = per_req_payloads
            multimodal_outputs = per_req_payloads

        routed_experts = None
        if self.routed_experts_initialized:
            routed_experts = self._omni_extract_routed_experts(scheduler_output)
        if inter_stage_outputs and self._should_accumulate_full_payload_output():
            for index, request_id in enumerate(req_ids):
                request_state = self.requests.get(request_id)
                if request_state is not None and inter_stage_outputs[index]:
                    self.accumulate_full_payload_output(
                        request_id,
                        inter_stage_outputs[index],
                        request_state,
                    )

        output = OmniModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={request_id: index for index, request_id in enumerate(req_ids)},
            sampled_token_ids=[],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=None,
            multimodal_outputs=multimodal_outputs,
            inter_stage_outputs=inter_stage_outputs,
            kv_connector_output=kv_connector_output,
            num_nans_in_logits={},
            cudagraph_stats=cudagraph_stats,
            ec_connector_output=ec_connector_output if self.supports_mm_inputs else None,
            generation_finished_req_ids=set(self._generation_finished_req_ids),
        )
        self._generation_finished_req_ids.clear()
        self._stepwise_output_req_ids = None
        output.omni_connector_output = self.get_omni_connector_output()
        output.routed_experts = routed_experts

        if not self.use_async_scheduling:
            return output
        return AsyncGPUModelRunnerOutput(
            model_runner_output=output,
            sampled_token_ids=torch.tensor([], device=self.device),
            invalid_req_indices=[],
            async_output_copy_stream=self.async_output_copy_stream,
            vocab_size=self.input_batch.vocab_size,
            logprobs_tensors=None,
        )


class IndexTTS2GenerationWorker(GPUGenerationWorker):
    model_runner_cls = IndexTTS2GenerationModelRunner
