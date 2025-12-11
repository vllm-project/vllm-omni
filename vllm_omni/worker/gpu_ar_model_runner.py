"""AR GPU Model Runner for vLLM-Omni.

Exposes per-request hidden representations via ModelRunnerOutput.pooler_output
and also outputs sampled tokens.
"""

from __future__ import annotations

import torch
from vllm.forward_context import BatchDescriptor
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import AsyncModelRunnerOutput
from vllm.v1.structured_output.utils import apply_grammar_bitmask
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.worker.gpu_model_runner import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    AsyncGPUModelRunnerOutput,
    IntermediateTensors,
    get_pp_group,
    get_tp_group,
    has_kv_transfer_group,
    set_forward_context,
)
from vllm.v1.worker.utils import is_residual_scattered_for_sp

from vllm_omni.outputs import OmniModelRunnerOutput
from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

logger = init_logger(__name__)


class GPUARModelRunner(OmniGPUModelRunner):
    """Autoregressive GPU model runner that returns hidden states per request.

    This runner follows the same preparation and forward path as GPUModelRunner
    (inputs assembly, multi-modal handling, TP/PP/DP integration, CUDA graphs),
    and additionally performs lightweight sampling so that sampled tokens are
    available in outputs. Hidden representations are taken at the same indices
    that GPUModelRunner would use for sampling/logits (i.e. `logits_indices`).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_ids = self._make_buffer(self.max_num_tokens, dtype=torch.int32)
        # each model stage has their own hidden size
        self.hidden_size = self.model_config.hf_text_config.hidden_size
        self.inputs_embeds = self._make_buffer(self.max_num_tokens, self.hidden_size, dtype=self.dtype, numpy=False)

    def _make_buffer(self, *size, dtype, numpy=True):
        # Prevent ray from pinning the buffer due to large size
        from vllm_omni.distributed.ray_utils.utils import (
            calculate_total_bytes,
            maybe_disable_pin_memory_for_ray,
        )

        total_bytes = calculate_total_bytes(size, dtype)

        # Use the context manager to temporarily disable pinning if needed
        with maybe_disable_pin_memory_for_ray(self, total_bytes):
            return super()._make_buffer(*size, dtype=dtype, numpy=numpy)

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> OmniModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors:
        with record_function_or_nullcontext("Preprocess"):
            with self.synchronize_input_prep():
                # Update persistent batch states.
                self._update_states(scheduler_output)

                # Decode per-request prompt_embeds / additional_information payloads
                # (if present) into CPU tensors
                self._decode_and_store_request_payloads(scheduler_output)

                if not scheduler_output.total_num_scheduled_tokens:
                    if not has_kv_transfer_group():
                        # Return empty ModelRunnerOutput if no work to do.
                        return EMPTY_MODEL_RUNNER_OUTPUT
                    return self.kv_connector_no_forward(scheduler_output, self.vllm_config)
                if self.cache_config.kv_sharing_fast_prefill:
                    assert not self.input_batch.num_prompt_logprobs, (
                        "--kv-sharing-fast-prefill produces incorrect "
                        "logprobs for prompt tokens, tokens, please disable "
                        "it when the requests need prompt logprobs"
                    )

                # Prepare the decoder inputs.
                (
                    attn_metadata,
                    logits_indices,
                    spec_decode_metadata,
                    num_scheduled_tokens_np,
                    spec_decode_common_attn_metadata,
                    max_query_len,
                    ubatch_slices,
                    num_tokens_after_padding,
                ) = self._prepare_inputs(scheduler_output)

            (
                num_scheduled_tokens,
                num_input_tokens,
                num_tokens_across_dp,
                input_ids,
                inputs_embeds,
                positions,
                intermediate_tensors,
                model_kwargs,
                per_req_additional_information,
            ) = self._preprocess(
                scheduler_output,
                num_scheduled_tokens_np,
                intermediate_tensors,
                ubatch_slices,
                num_tokens_after_padding,
            )

            uniform_decode = (max_query_len == self.uniform_decode_query_len) and (
                num_scheduled_tokens == self.input_batch.num_reqs * max_query_len
            )
            batch_descriptor = BatchDescriptor(num_tokens=num_input_tokens, uniform_decode=uniform_decode)
            cudagraph_runtime_mode, batch_descriptor = self.cudagraph_dispatcher.dispatch(batch_descriptor)

        # This is currently to get around the assert in the DPMetadata
        # where it wants `num_tokens_across_dp` to align with `num_tokens`
        if ubatch_slices is not None:
            num_input_tokens = ubatch_slices[0].num_tokens

        # Run the model.
        # Use persistent buffers for CUDA graphs.
        with (
            set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_input_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                batch_descriptor=batch_descriptor,
                ubatch_slices=ubatch_slices,
            ),
            record_function_or_nullcontext("Forward"),
            self.maybe_get_kv_connector_output(scheduler_output) as kv_connector_output,
        ):
            model_kwargs_extra = self._build_model_kwargs_extra(per_req_additional_information, num_scheduled_tokens_np)
            model_output = self.model(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **model_kwargs,
                sampling_metadata=self.input_batch.sampling_metadata,
                logits_index=logits_indices,
                sampler=self.sampler,
                **model_kwargs_extra,
            )

        with record_function_or_nullcontext("Postprocess"):
            if self.use_aux_hidden_state_outputs:
                # True when EAGLE 3 is used.
                hidden_states, aux_hidden_states = model_output
            else:
                # Common case.
                hidden_states = model_output
                aux_hidden_states = None

            # hidden_states, multimodal_outputs = self.extract_multimodal_outputs(
            #     hidden_states
            # )
            multimodal_outputs = model_output.multimodal_outputs
            hidden_states = model_output.text_hidden_states
            # The model side may return per-request additional_information updates (model-agnostic channel).
            # Convention: multimodal_outputs["additional_information_update"] is a list[dict] in batch order;
            # the runner merges it into the corresponding request's additional_information_cpu for subsequent decode.
            self._process_additional_information_updates(multimodal_outputs)
            if not self.broadcast_pp_output:
                # Common case.
                if not get_pp_group().is_last_rank:
                    # Return the intermediate tensors.
                    assert isinstance(hidden_states, IntermediateTensors)
                    hidden_states.kv_connector_output = kv_connector_output
                    return hidden_states

                if self.is_pooling_model:
                    # Return the pooling output.
                    output = self._pool(hidden_states, num_scheduled_tokens, num_scheduled_tokens_np)
                    output.kv_connector_output = kv_connector_output
                    return output

                sample_hidden_states = hidden_states[logits_indices]
                logits = self.model.compute_logits(sample_hidden_states)
            else:
                # Rare case.
                assert not self.is_pooling_model

                if not get_pp_group().is_last_rank:
                    all_gather_tensors = {
                        "residual": not is_residual_scattered_for_sp(self.vllm_config, num_input_tokens)
                    }
                    get_pp_group().send_tensor_dict(
                        hidden_states.tensors,
                        all_gather_group=get_tp_group(),
                        all_gather_tensors=all_gather_tensors,
                    )
                    logits = None
                else:
                    sample_hidden_states = hidden_states[logits_indices]
                    logits = self.model.compute_logits(sample_hidden_states)

                model_output_broadcast_data = {}
                if logits is not None:
                    model_output_broadcast_data["logits"] = logits.contiguous()

                model_output_broadcast_data = get_pp_group().broadcast_tensor_dict(
                    model_output_broadcast_data, src=len(get_pp_group().ranks) - 1
                )
                assert model_output_broadcast_data is not None
                logits = model_output_broadcast_data["logits"]

            # Apply structured output bitmasks if present
            if scheduler_output.grammar_bitmask is not None:
                apply_grammar_bitmask(scheduler_output, self.input_batch, logits, self.device)

        with record_function_or_nullcontext("Sample"):
            sampler_output = self._sample(logits, spec_decode_metadata)

        def propose_draft_token_ids(sampled_token_ids):
            assert spec_decode_common_attn_metadata is not None
            with record_function_or_nullcontext("Draft"):
                self._draft_token_ids = self.propose_draft_token_ids(
                    scheduler_output,
                    sampled_token_ids,
                    self.input_batch.sampling_metadata,
                    hidden_states,
                    sample_hidden_states,
                    aux_hidden_states,
                    spec_decode_metadata,
                    spec_decode_common_attn_metadata,
                )

        use_padded_batch_for_eagle = (
            self.speculative_config
            and self.speculative_config.use_eagle()
            and not self.speculative_config.disable_padded_drafter_batch
        )
        effective_drafter_max_model_len = self.max_model_len
        if effective_drafter_max_model_len is None:
            effective_drafter_max_model_len = self.model_config.max_model_len
        if (
            self.speculative_config
            and self.speculative_config.draft_model_config is not None
            and self.speculative_config.draft_model_config.max_model_len is not None
        ):
            effective_drafter_max_model_len = self.speculative_config.draft_model_config.max_model_len
        input_fits_in_drafter = spec_decode_common_attn_metadata and (
            spec_decode_common_attn_metadata.seq_lens.max() + self.speculative_config.num_speculative_tokens
            <= effective_drafter_max_model_len
        )
        if use_padded_batch_for_eagle and input_fits_in_drafter:
            # EAGLE speculative decoding can use the GPU sampled tokens
            # as inputs, and does not need to wait for bookkeeping to finish.
            propose_draft_token_ids(sampler_output.sampled_token_ids)

        with record_function_or_nullcontext("Bookkeep"):
            (
                num_nans_in_logits,
                logprobs_lists,
                valid_sampled_token_ids,
                prompt_logprobs_dict,
                req_ids_output_copy,
                req_id_to_index_output_copy,
                invalid_req_indices,
            ) = self._bookkeeping_sync(
                scheduler_output,
                sampler_output,
                logits,
                hidden_states,
                num_scheduled_tokens,
            )

        if self.speculative_config and not use_padded_batch_for_eagle and input_fits_in_drafter:
            # ngram and other speculative decoding methods use the sampled
            # tokens on the CPU, so they are run after bookkeeping.
            propose_draft_token_ids(valid_sampled_token_ids)

        with record_function_or_nullcontext("EPLB"):
            self.eplb_step()

        # Convert to per-request tensors on CPU
        hidden_states_cpu = hidden_states.detach().to("cpu").contiguous()
        # pooler_output: list[torch.Tensor | None] = []
        pooler_output: list[torch.Tensor | None] = []
        prev_logits_index = 0
        for rid, logits_index in zip(req_ids_output_copy, logits_indices):
            # Base payload: hidden slice for this request in this iteration
            hidden_slice = hidden_states_cpu[prev_logits_index : logits_index + 1]
            payload: dict[str, object] = {"hidden": hidden_slice}
            # Merge multimodal_outputs if present
            if isinstance(multimodal_outputs, dict) and multimodal_outputs:
                mm_payload: dict[str, object] = {}
                for k, v in multimodal_outputs.items():
                    try:
                        # Case 1: tensor aligned on token dimension
                        if isinstance(v, torch.Tensor) and v.shape[0] == hidden_states_cpu.shape[0]:
                            mm_payload[k] = v.detach().to("cpu")[prev_logits_index : logits_index + 1].contiguous()
                        elif isinstance(v, torch.Tensor) and v.shape[0] != hidden_states_cpu.shape[0]:
                            logger.error(
                                f"Error in merge multimodal outputs: Tensor dimension mismatch, \
                                          {v.shape} != {hidden_states_cpu.shape} for {k}"
                            )
                        # Case 2: nested dict of tensors aligned on token dimension (e.g., selected_hidden_layers)
                        elif isinstance(v, dict):
                            sub_dict: dict[str, torch.Tensor] = {}
                            for sk, sv in v.items():
                                if isinstance(sv, torch.Tensor) and sv.shape[0] == hidden_states_cpu.shape[0]:
                                    sub_dict[str(sk)] = (
                                        sv.detach().to("cpu")[prev_logits_index : logits_index + 1].contiguous()
                                    )
                            if sub_dict:
                                mm_payload[k] = sub_dict
                        elif isinstance(v, list):
                            element = v[0]
                            if isinstance(element, torch.Tensor):
                                element = element.detach().to("cpu").contiguous()
                            multimodal_outputs[k] = v[1:] if len(v) > 1 else v
                            mm_payload[k] = element
                    except Exception as e:
                        # Best-effort; skip malformed entries
                        logger.error(f"Error in merge multimodal outputs: {e}")
                if mm_payload:
                    payload.update(mm_payload)
            pooler_output.append(payload)  # type: ignore[arg-type]
            prev_logits_index = logits_index + 1
        output = OmniModelRunnerOutput(
            req_ids=req_ids_output_copy,
            req_id_to_index=req_id_to_index_output_copy,
            sampled_token_ids=valid_sampled_token_ids,
            logprobs=logprobs_lists,
            prompt_logprobs_dict=prompt_logprobs_dict,
            pooler_output=(pooler_output if self.vllm_config.model_config.engine_output_type != "text" else None),
            kv_connector_output=kv_connector_output,
            num_nans_in_logits=num_nans_in_logits,
        )

        if not self.use_async_scheduling:
            return output

        return AsyncGPUModelRunnerOutput(
            model_runner_output=output,
            sampled_token_ids=sampler_output.sampled_token_ids,
            invalid_req_indices=invalid_req_indices,
            async_output_copy_stream=self.async_output_copy_stream,
        )

    def _merge_additional_information_update(self, req_id: str, upd: dict) -> None:
        req_state = self.requests.get(req_id)
        if req_state is None:
            return
        existing = getattr(req_state, "additional_information_cpu", {})
        if not isinstance(existing, dict):
            existing = {}
        merged = dict(existing)
        for k, v in upd.items():
            if isinstance(v, torch.Tensor):
                merged[k] = v.detach().to("cpu").contiguous()
            elif isinstance(v, list):
                new_list = []
                for item in v:
                    if isinstance(item, torch.Tensor):
                        new_list.append(item.detach().to("cpu").contiguous())
                    else:
                        new_list.append(item)
                merged[k] = new_list
            else:
                merged[k] = v
        setattr(req_state, "additional_information_cpu", merged)

    # ===== Helper functions extracted for clarity and reuse =====
