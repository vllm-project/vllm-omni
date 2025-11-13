"""AR GPU Model Runner for vLLM-omni.

Exposes per-request hidden representations via ModelRunnerOutput.pooler_output
and also outputs sampled tokens.
"""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
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
from vllm.v1.worker.ubatch_utils import UBatchSlices
from vllm.v1.worker.utils import is_residual_scattered_for_sp
from vllm_omni.engine import AdditionalInformationPayload, PromptEmbedsPayload
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

    def _preprocess(
        self,
        scheduler_output: "SchedulerOutput",
        num_scheduled_tokens_np: np.ndarray,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        ubatch_slices: Optional[UBatchSlices] = None,
        num_tokens_after_padding: Optional[torch.Tensor] = None,
    ) -> tuple[
        int,
        int,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        torch.Tensor,
        Optional[IntermediateTensors],
        dict[str, Any],
        Optional[dict[str, dict]],
    ]:

        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        if ubatch_slices:
            assert num_tokens_after_padding is not None
            num_input_tokens = int(num_tokens_after_padding[0].item() * 2)
            self.pad_out_ubatch_slice(ubatch_slices, num_input_tokens)
        elif ubatch_slices is None:
            num_input_tokens = self._get_num_input_tokens(num_scheduled_tokens)
            num_pad, num_tokens_after_padding = self.get_dp_padding(num_input_tokens)
            num_input_tokens += num_pad

        # _prepare_inputs may reorder the batch, so we must gather multi
        # modal outputs after that to ensure the correct order
        per_req_additional_information: Optional[dict[str, dict]] = None
        if (
            self.supports_mm_inputs
            and get_pp_group().is_first_rank
            and not self.model_config.is_encoder_decoder
        ):
            # Run the multimodal encoder if any.
            self._execute_mm_encoder(scheduler_output)
            mm_embeds = self._gather_mm_embeddings(scheduler_output)

            # NOTE(woosuk): To unify token ids and soft tokens (vision
            # embeddings), we always use embeddings (rather than token ids)
            # as input to the multimodal model, even when the input is text.
            inputs_embeds_scheduled = self.model.get_input_embeddings(
                input_ids=self.input_ids.gpu[:num_scheduled_tokens],
                multimodal_embeddings=mm_embeds or None,
            )

            # TODO(woosuk): Avoid the copy. Optimize.
            self.inputs_embeds.gpu[:num_scheduled_tokens].copy_(inputs_embeds_scheduled)

            # Reset per-step additional information collector (deprecated concat path)
            if hasattr(self, "_forward_additional_information"):
                self._forward_additional_information = None
            # New: per-request additional information for this step
            per_req_additional_information = {}

            # Overlay custom prompt_embeds per request for the prompt portion;
            # collect additional_information (tensor/list) for prefill portion only
            for req_index, req_id in enumerate(self.input_batch.req_ids):
                req_state = self.requests[req_id]
                pe_cpu = getattr(req_state, "prompt_embeds_cpu", None)
                addi_cpu = getattr(req_state, "additional_information_cpu", None)
                num_computed_tokens = int(
                    self.input_batch.num_computed_tokens_cpu[req_index]
                )
                prompt_len = len(req_state.prompt_token_ids)
                prompt_remaining = max(0, prompt_len - num_computed_tokens)
                sched_tokens = int(num_scheduled_tokens_np[req_index])
                overlay_len = min(sched_tokens, prompt_remaining)
                if overlay_len <= 0:
                    continue
                if pe_cpu is not None:
                    src = pe_cpu[
                        num_computed_tokens : num_computed_tokens + overlay_len
                    ].to(dtype=self.dtype, device=self.device, non_blocking=True)
                    start_offset = int(self.query_start_loc.cpu[req_index])
                    self.inputs_embeds[start_offset : start_offset + overlay_len].copy_(
                        src
                    )
                # Build per-request additional information (no cross-request concat)
                if addi_cpu is not None and isinstance(addi_cpu, dict):
                    req_info: dict[str, object] = {}
                    for k, v in addi_cpu.items():
                        if isinstance(v, torch.Tensor):
                            # For prefill tokens, pass only the scheduled slice; for decode or no scheduled tokens, pass whole tensor
                            if overlay_len > 0:
                                try:
                                    seg = v[
                                        num_computed_tokens : num_computed_tokens + overlay_len
                                    ].detach().to("cpu").contiguous()
                                except Exception:
                                    seg = v.detach().to("cpu").contiguous()
                                req_info[k] = seg
                            else:
                                req_info[k] = v.detach().to("cpu").contiguous()
                        elif isinstance(v, list):
                            req_info[k] = v
                        else:
                            req_info[k] = v
                    per_req_additional_information[req_id] = req_info

            input_ids = self.input_ids.gpu[:num_input_tokens]
            inputs_embeds = self.inputs_embeds.gpu[:num_input_tokens]
            model_kwargs = {
                **self._init_model_kwargs(num_scheduled_tokens),
                **self._extract_mm_kwargs(scheduler_output),
            }
        elif self.enable_prompt_embeds and get_pp_group().is_first_rank:
            # Get the input embeddings for the tokens that are not input embeds,
            # then put them into the appropriate positions.
            # TODO(qthequartermasterman): Since even when prompt embeds are
            # enabled, (a) not all requests will use prompt embeds, and (b)
            # after the initial prompt is processed, the rest of the generated
            # tokens will be token ids, it is not desirable to have the
            # embedding layer outside of the CUDA graph all the time. The v0
            # engine avoids this by "double compiling" the CUDA graph, once
            # with input_ids and again with inputs_embeds, for all num_tokens.
            # If a batch only has token ids, then including the embedding layer
            # in the CUDA graph will be more performant (like in the else case
            # below).
            token_ids_idx = (
                self.is_token_ids.gpu[:num_scheduled_tokens]
                .nonzero(as_tuple=False)
                .squeeze(1)
            )
            # Some tokens ids may need to become embeds
            if token_ids_idx.numel() > 0:
                token_ids = self.input_ids.gpu[token_ids_idx]
                tokens_to_embeds = self.model.get_input_embeddings(input_ids=token_ids)
                self.inputs_embeds.gpu[token_ids_idx] = tokens_to_embeds

            inputs_embeds = self.inputs_embeds.gpu[:num_input_tokens]
            model_kwargs = self._init_model_kwargs(num_input_tokens)
            input_ids = None
        else:
            # For text-only models, we use token ids as input.
            # While it is possible to use embeddings as input just like the
            # multimodal models, it is not desirable for performance since
            # then the embedding layer is not included in the CUDA graph.
            input_ids = self.input_ids.gpu[:num_input_tokens]
            inputs_embeds = None
            model_kwargs = self._init_model_kwargs(num_input_tokens)
        if self.uses_mrope:
            positions = self.mrope_positions.gpu[:, :num_input_tokens]
        else:
            positions = self.positions.gpu[:num_input_tokens]

        if get_pp_group().is_first_rank:
            intermediate_tensors = None
        else:
            intermediate_tensors = self.sync_and_slice_intermediate_tensors(
                num_input_tokens, intermediate_tensors, True
            )

        if (
            self.model_config.is_encoder_decoder
            and scheduler_output.scheduled_encoder_inputs
        ):
            encoder_inputs = self._extract_encoder_inputs(scheduler_output)
            model_kwargs.update(encoder_inputs)

        return (
            num_scheduled_tokens,
            num_input_tokens,
            num_tokens_after_padding,
            input_ids,
            inputs_embeds,
            positions,
            intermediate_tensors,
            model_kwargs,
            per_req_additional_information,
        )

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[OmniModelRunnerOutput, AsyncModelRunnerOutput, IntermediateTensors]:
        with record_function_or_nullcontext("Preprocess"):
            with self.synchronize_input_prep():
                # Update persistent batch states.
                self._update_states(scheduler_output)

                # Decode per-request prompt_embeds / additional_hidden_states payloads
                # (if present) into CPU tensors
                try:
                    new_reqs = getattr(scheduler_output, "scheduled_new_reqs", [])
                    if new_reqs:
                        for nr in new_reqs:
                            req_id = getattr(nr, "req_id", None) or getattr(
                                nr, "request_id", None
                            )
                            if req_id is None:
                                continue
                            # prompt_embeds
                            payload_pe = getattr(nr, "prompt_embeds", None)
                            if payload_pe is not None:
                                if isinstance(payload_pe, torch.Tensor):
                                    pe_cpu = payload_pe.detach().to("cpu").contiguous()
                                elif isinstance(payload_pe, PromptEmbedsPayload):
                                    dt = np.dtype(
                                        getattr(payload_pe, "dtype", "float32")
                                    )
                                    arr = np.frombuffer(payload_pe.data, dtype=dt)
                                    arr = arr.reshape(payload_pe.shape)
                                    pe_cpu = torch.from_numpy(arr)
                                else:
                                    pe_cpu = None
                                if pe_cpu is not None and req_id in self.requests:
                                    setattr(
                                        self.requests[req_id],
                                        "prompt_embeds_cpu",
                                        pe_cpu,
                                    )
                            # additional_information
                            payload_info = getattr(nr, "additional_information", None)
                            if payload_info is not None:
                                info_dict = {}
                                if isinstance(payload_info, dict):
                                    # Already decoded
                                    info_dict = payload_info
                                elif isinstance(
                                    payload_info, AdditionalInformationPayload
                                ):
                                    for k, entry in payload_info.entries.items():
                                        if entry.tensor_data is not None:
                                            dt = np.dtype(
                                                getattr(
                                                    entry, "tensor_dtype", "float32"
                                                )
                                            )
                                            arr = np.frombuffer(
                                                entry.tensor_data, dtype=dt
                                            )
                                            arr = arr.reshape(entry.tensor_shape)
                                            info_dict[k] = torch.from_numpy(arr)
                                        else:
                                            info_dict[k] = entry.list_data
                                if info_dict and req_id in self.requests:
                                    setattr(
                                        self.requests[req_id],
                                        "additional_information_cpu",
                                        info_dict,
                                    )
                except Exception as e:
                    logger.error(
                        f"Error decoding prompt_embeds / additional_information: {e}"
                    )
                    pass

                if not scheduler_output.total_num_scheduled_tokens:
                    if not has_kv_transfer_group():
                        # Return empty ModelRunnerOutput if no work to do.
                        return EMPTY_MODEL_RUNNER_OUTPUT
                    return self.kv_connector_no_forward(
                        scheduler_output, self.vllm_config
                    )
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
            batch_descriptor = BatchDescriptor(
                num_tokens=num_input_tokens, uniform_decode=uniform_decode
            )
            cudagraph_runtime_mode, batch_descriptor = (
                self.cudagraph_dispatcher.dispatch(batch_descriptor)
            )

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

            model_kwargs_extra = {}
            # Pass per-request additional information map for this step (no concat)
            if per_req_additional_information:
                model_kwargs_extra["additional_information_by_req_id"] = per_req_additional_information
            # Always pass per-request runtime additional_information (persisted in request state)
            try:
                per_req_runtime_info = []
                for req_id in self.input_batch.req_ids:
                    req_state = self.requests.get(req_id)
                    info = (
                        getattr(req_state, "additional_information_cpu", None)
                        if req_state is not None
                        else None
                    )
                    per_req_runtime_info.append(info if isinstance(info, dict) else {})
                model_kwargs_extra["runtime_additional_information"] = (
                    per_req_runtime_info
                )
                model_kwargs_extra["request_ids"] = self.input_batch.req_ids
                # Pass each request's token span within the flattened sequence for this step, enabling the model to map decode/prefill by request
                req_token_spans = []
                for req_index in range(len(self.input_batch.req_ids)):
                    start_offset = int(self.query_start_loc.cpu[req_index])
                    sched_tokens = int(num_scheduled_tokens_np[req_index])
                    req_token_spans.append((start_offset, start_offset + sched_tokens))
                model_kwargs_extra["request_token_spans"] = req_token_spans
            except Exception:
                pass
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

            hidden_states, multimodal_outputs = self.extract_multimodal_outputs(
                hidden_states
            )
            # The model side may return per-request additional_information updates (model-agnostic channel).
            # Convention: multimodal_outputs["additional_information_update"] is a list[dict] in batch order;
            # the runner merges it into the corresponding request's additional_information_cpu for subsequent decode.
            try:
                if (
                    isinstance(multimodal_outputs, dict)
                    and (
                        "additional_information_update" in multimodal_outputs
                        or "additional_information_update_by_req_id" in multimodal_outputs
                    )
                ):
                    # Option A: list[dict] in batch order
                    updates_list = multimodal_outputs.get(
                        "additional_information_update"
                    )
                    if isinstance(updates_list, list):
                        for idx, upd in enumerate(updates_list):
                            if not isinstance(upd, dict) or idx >= len(self.input_batch.req_ids):
                                continue
                            req_id = self.input_batch.req_ids[idx]
                            self._merge_additional_information_update(req_id, upd)
                    # Option B: dict[str, dict] keyed by req_id
                    updates_map = multimodal_outputs.get(
                        "additional_information_update_by_req_id"
                    )
                    if isinstance(updates_map, dict):
                        for req_id, upd in updates_map.items():
                            if not isinstance(upd, dict):
                                continue
                            if req_id not in self.requests:
                                continue
                            self._merge_additional_information_update(req_id, upd)
            except Exception as e:
                logger.error(f"Error merging for requests:{self.input_batch.req_ids} additional information update: {e}, with the multimodal_outputs as {multimodal_outputs}")
            if not self.broadcast_pp_output:

                # Common case.
                if not get_pp_group().is_last_rank:
                    # Return the intermediate tensors.
                    assert isinstance(hidden_states, IntermediateTensors)
                    hidden_states.kv_connector_output = kv_connector_output
                    return hidden_states

                if self.is_pooling_model:
                    # Return the pooling output.
                    output = self._pool(
                        hidden_states, num_scheduled_tokens, num_scheduled_tokens_np
                    )
                    output.kv_connector_output = kv_connector_output
                    return output

                sample_hidden_states = hidden_states[logits_indices]
                logits = self.model.compute_logits(sample_hidden_states)
            else:
                # Rare case.
                assert not self.is_pooling_model

                if not get_pp_group().is_last_rank:
                    all_gather_tensors = {
                        "residual": not is_residual_scattered_for_sp(
                            self.vllm_config, num_input_tokens
                        )
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
                apply_grammar_bitmask(
                    scheduler_output, self.input_batch, logits, self.device
                )

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
            effective_drafter_max_model_len = (
                self.speculative_config.draft_model_config.max_model_len
            )
        input_fits_in_drafter = spec_decode_common_attn_metadata and (
            spec_decode_common_attn_metadata.seq_lens.max()
            + self.speculative_config.num_speculative_tokens
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

        if (
            self.speculative_config
            and not use_padded_batch_for_eagle
            and input_fits_in_drafter
        ):
            # ngram and other speculative decoding methods use the sampled
            # tokens on the CPU, so they are run after bookkeeping.
            propose_draft_token_ids(valid_sampled_token_ids)

        with record_function_or_nullcontext("EPLB"):
            self.eplb_step()

        # Convert to per-request tensors on CPU
        hidden_states_cpu = hidden_states.detach().to("cpu").contiguous()
        pooler_output: list[Optional[torch.Tensor]] = []
        prev_logits_index = 0
        for logits_index in logits_indices:
            pooler_output.append(
                hidden_states_cpu[prev_logits_index : logits_index + 1]
            )
            prev_logits_index = logits_index + 1

        output = OmniModelRunnerOutput(
            req_ids=req_ids_output_copy,
            req_id_to_index=req_id_to_index_output_copy,
            sampled_token_ids=valid_sampled_token_ids,
            logprobs=logprobs_lists,
            prompt_logprobs_dict=prompt_logprobs_dict,
            pooler_output=(
                pooler_output
                if self.vllm_config.model_config.engine_output_type != "text"
                else None
            ),
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