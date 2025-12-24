"""AR GPU Model Runner for vLLM-Omni.

Exposes per-request hidden representations via ModelRunnerOutput.pooler_output
and also outputs sampled tokens.
"""

from __future__ import annotations

from copy import copy
from typing import Any, NamedTuple

import numpy as np
import torch
from vllm.config import CUDAGraphMode
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.outputs import AsyncModelRunnerOutput
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.structured_output.utils import apply_grammar_bitmask
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.worker.gpu_model_runner import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    AsyncGPUModelRunnerOutput,
    IntermediateTensors,
    get_pp_group,
    get_tp_group,
    has_kv_transfer_group,
)
from vllm.v1.worker.utils import is_residual_scattered_for_sp

from vllm_omni.core.sched.omni_ar_scheduler import KVCacheTransferData
from vllm_omni.outputs import OmniModelRunnerOutput
from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

logger = init_logger(__name__)


class ExecuteModelState(NamedTuple):
    scheduler_output: SchedulerOutput
    logits: torch.Tensor | None
    spec_decode_metadata: Any
    spec_decode_common_attn_metadata: Any
    hidden_states: torch.Tensor
    sample_hidden_states: torch.Tensor
    aux_hidden_states: list[torch.Tensor] | None
    ec_connector_output: Any
    multimodal_outputs: Any


class GPUARModelRunner(OmniGPUModelRunner):
    """Autoregressive GPU model runner that returns hidden states per request.

    Follows the v0.12 two-phase execute/sample flow from GPUModelRunner, and
    reuses Omni hooks for additional_information / multimodal outputs. This
    class only overrides sample_tokens to expose hidden states + multimodal
    outputs per request while keeping Async output semantics.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_ids = self._make_buffer(self.max_num_tokens, dtype=torch.int32)
        # each model stage has their own hidden size
        self.hidden_size = self.model_config.hf_text_config.hidden_size
        self.inputs_embeds = self._make_buffer(self.max_num_tokens, self.hidden_size, dtype=self.dtype, numpy=False)
        self.omni_connector = None

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
    ) -> OmniModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors | None:
        with record_function_or_nullcontext("Preprocess"):
            with self.synchronize_input_prep():
                # [Fix] Handle KV transfer BEFORE updating states (which removes finished requests)
                self.kv_extracted_req_ids = self._handle_finished_requests_kv_transfer(scheduler_output)

                self._update_states(scheduler_output)
                self._decode_and_store_request_payloads(scheduler_output)

                if not scheduler_output.total_num_scheduled_tokens:
                    if not has_kv_transfer_group():
                        return EMPTY_MODEL_RUNNER_OUTPUT
                    return self.kv_connector_no_forward(scheduler_output, self.vllm_config)
                if self.cache_config.kv_sharing_fast_prefill:
                    assert not self.input_batch.num_prompt_logprobs, (
                        "--kv-sharing-fast-prefill produces incorrect "
                        "logprobs for prompt tokens, tokens, please disable "
                        "it when the requests need prompt logprobs"
                    )

                num_reqs = self.input_batch.num_reqs
                req_ids = self.input_batch.req_ids
                tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
                num_scheduled_tokens_np = np.array(tokens, dtype=np.int32)
                max_num_scheduled_tokens = int(num_scheduled_tokens_np.max())
                num_tokens_unpadded = scheduler_output.total_num_scheduled_tokens

                logits_indices, spec_decode_metadata = self._prepare_inputs(
                    scheduler_output,
                    num_scheduled_tokens_np,
                )

                (
                    cudagraph_mode,
                    batch_desc,
                    ubatch_slices,
                    num_tokens_across_dp,
                ) = self._determine_batch_execution_and_padding(
                    num_tokens=num_tokens_unpadded,
                    num_reqs=num_reqs,
                    num_scheduled_tokens_np=num_scheduled_tokens_np,
                    max_num_scheduled_tokens=max_num_scheduled_tokens,
                    use_cascade_attn=False,
                )

                num_tokens_padded = batch_desc.num_tokens
                num_reqs_padded = batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs
                use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
                pad_attn = cudagraph_mode == CUDAGraphMode.FULL

                (
                    attn_metadata,
                    spec_decode_common_attn_metadata,
                ) = self._build_attention_metadata(
                    num_tokens=num_tokens_unpadded,
                    num_tokens_padded=num_tokens_padded if pad_attn else None,
                    num_reqs=num_reqs,
                    num_reqs_padded=num_reqs_padded if pad_attn else None,
                    max_query_len=max_num_scheduled_tokens,
                    ubatch_slices=ubatch_slices,
                    logits_indices=logits_indices,
                    use_spec_decode=use_spec_decode,
                    num_scheduled_tokens=scheduler_output.num_scheduled_tokens,
                    cascade_attn_prefix_lens=None,
                )

            (
                input_ids,
                inputs_embeds,
                positions,
                intermediate_tensors,
                model_kwargs,
                ec_connector_output,
            ) = self._preprocess(
                scheduler_output,
                num_tokens_padded,
                intermediate_tensors,
            )

        if self.calculate_kv_scales:
            cudagraph_mode = CUDAGraphMode.NONE
            self.calculate_kv_scales = False

        with (
            set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_tokens_padded,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=cudagraph_mode,
                batch_descriptor=batch_desc,
                ubatch_slices=ubatch_slices,
            ),
            record_function_or_nullcontext("Forward"),
            self.maybe_get_kv_connector_output(scheduler_output) as kv_connector_output,
        ):
            model_output = self._model_forward(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **model_kwargs,
                sampling_metadata=self.input_batch.sampling_metadata,
                logits_index=logits_indices,
                sampler=self.sampler,
            )

        with record_function_or_nullcontext("gpu_model_runner: postprocess"):
            if self.use_aux_hidden_state_outputs:
                # True when EAGLE 3 is used.
                hidden_states, aux_hidden_states = model_output
            else:
                # Common case.
                hidden_states = model_output
                aux_hidden_states = None

            multimodal_outputs = model_output.multimodal_outputs
            hidden_states = model_output.text_hidden_states

            if multimodal_outputs is not None:
                keys_or_type = (
                    list(multimodal_outputs.keys())
                    if isinstance(multimodal_outputs, dict)
                    else type(multimodal_outputs)
                )
                logger.debug(f"[AR] execute_model: multimodal_outputs keys = {keys_or_type}")
            else:
                logger.debug("[AR] execute_model: multimodal_outputs is None")

            if not self.broadcast_pp_output:
                if not get_pp_group().is_last_rank:
                    assert isinstance(hidden_states, IntermediateTensors)
                    hidden_states.kv_connector_output = kv_connector_output
                    return hidden_states

                if self.is_pooling_model:
                    output = self._pool(
                        hidden_states,
                        num_tokens_padded,
                        num_scheduled_tokens_np,
                    )
                    output.kv_connector_output = kv_connector_output
                    return output

                sample_hidden_states = hidden_states[logits_indices]
                logits = self.model.compute_logits(sample_hidden_states)
            else:
                assert not self.is_pooling_model

                if not get_pp_group().is_last_rank:
                    all_gather_tensors = {
                        "residual": not is_residual_scattered_for_sp(self.vllm_config, num_tokens_padded)
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

                model_output_broadcast_data: dict[str, Any] = {}
                if logits is not None:
                    model_output_broadcast_data["logits"] = logits.contiguous()

                broadcasted = get_pp_group().broadcast_tensor_dict(
                    model_output_broadcast_data, src=len(get_pp_group().ranks) - 1
                )
                assert broadcasted is not None
                logits = broadcasted["logits"]

        self.execute_model_state = ExecuteModelState(
            scheduler_output,
            logits,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            ec_connector_output,
            multimodal_outputs,
        )
        self.kv_connector_output = kv_connector_output

        return None

    @torch.inference_mode()
    def sample_tokens(
        self,
        grammar_output: GrammarOutput | None,
    ) -> OmniModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors:
        kv_connector_output = self.kv_connector_output
        self.kv_connector_output = None

        kv_extracted_req_ids = getattr(self, "kv_extracted_req_ids", None)
        self.kv_extracted_req_ids = None

        if self.execute_model_state is None:
            if not kv_connector_output:
                return None  # type: ignore[return-value]
            if kv_connector_output.is_empty():
                return EMPTY_MODEL_RUNNER_OUTPUT
            output = copy(EMPTY_MODEL_RUNNER_OUTPUT)
            output.kv_connector_output = kv_connector_output
            return output

        (
            scheduler_output,
            logits,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            ec_connector_output,
            multimodal_outputs,
        ) = self.execute_model_state
        self.execute_model_state = None

        if grammar_output is not None:
            apply_grammar_bitmask(scheduler_output, grammar_output, self.input_batch, logits)

        with record_function_or_nullcontext("gpu_model_runner: sample"):
            sampler_output = self._sample(logits, spec_decode_metadata)

        self.input_batch.prev_sampled_token_ids = None

        def propose_draft_token_ids(sampled_token_ids):
            assert spec_decode_common_attn_metadata is not None
            with record_function_or_nullcontext("gpu_model_runner: draft"):
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

        spec_config = self.speculative_config
        use_padded_batch_for_eagle = (
            spec_config is not None and spec_config.use_eagle() and not spec_config.disable_padded_drafter_batch
        )
        effective_drafter_max_model_len = self.max_model_len
        if effective_drafter_max_model_len is None:
            effective_drafter_max_model_len = self.model_config.max_model_len
        if (
            spec_config is not None
            and spec_config.draft_model_config is not None
            and spec_config.draft_model_config.max_model_len is not None
        ):
            effective_drafter_max_model_len = spec_config.draft_model_config.max_model_len
        input_fits_in_drafter = spec_decode_common_attn_metadata and (
            spec_decode_common_attn_metadata.max_seq_len + self.num_spec_tokens <= effective_drafter_max_model_len
        )
        if use_padded_batch_for_eagle:
            assert self.speculative_config is not None
            assert isinstance(self.drafter, EagleProposer)
            sampled_token_ids = sampler_output.sampled_token_ids
            if input_fits_in_drafter:
                propose_draft_token_ids(sampled_token_ids)
            elif self.valid_sampled_token_count_event is not None:
                assert spec_decode_common_attn_metadata is not None
                next_token_ids, valid_sampled_tokens_count = self.drafter.prepare_next_token_ids_padded(
                    spec_decode_common_attn_metadata,
                    sampled_token_ids,
                    self.requests,
                    self.input_batch,
                    self.discard_request_mask.gpu,
                )
                self._copy_valid_sampled_token_count(next_token_ids, valid_sampled_tokens_count)

        with record_function_or_nullcontext("gpu_model_runner: bookkeep"):
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
                scheduler_output.total_num_scheduled_tokens,
                spec_decode_metadata,
            )

        if self.speculative_config and not use_padded_batch_for_eagle and input_fits_in_drafter:
            propose_draft_token_ids(valid_sampled_token_ids)

        with record_function_or_nullcontext("gpu_model_runner: eplb"):
            self.eplb_step()

        self._process_additional_information_updates(multimodal_outputs)

        hidden_states_cpu = hidden_states.detach().to("cpu").contiguous()
        num_scheduled_tokens_np = getattr(self, "_omni_num_scheduled_tokens_np", None)
        if num_scheduled_tokens_np is None:
            req_ids = self.input_batch.req_ids
            num_scheduled_tokens_np = np.array(
                [scheduler_output.num_scheduled_tokens[rid] for rid in req_ids],
                dtype=np.int32,
            )

        pooler_output: list[dict[str, object]] = []
        for rid in req_ids_output_copy:
            idx = req_id_to_index_output_copy[rid]
            start = int(self.query_start_loc.cpu[idx])
            sched = int(num_scheduled_tokens_np[idx])
            end = start + sched
            hidden_slice = hidden_states_cpu[start:end]
            payload: dict[str, object] = {"hidden": hidden_slice}
            if isinstance(multimodal_outputs, dict) and multimodal_outputs:
                mm_payload: dict[str, object] = {}
                for k, v in multimodal_outputs.items():
                    try:
                        if isinstance(v, torch.Tensor) and v.shape[0] == hidden_states_cpu.shape[0]:
                            mm_payload[k] = v.detach().to("cpu")[start:end].contiguous()
                        elif isinstance(v, dict):
                            sub_dict: dict[str, torch.Tensor] = {}
                            for sk, sv in v.items():
                                if isinstance(sv, torch.Tensor) and sv.shape[0] == hidden_states_cpu.shape[0]:
                                    sub_dict[str(sk)] = sv.detach().to("cpu")[start:end].contiguous()
                            if sub_dict:
                                mm_payload[k] = sub_dict
                        elif isinstance(v, list):
                            element = v[0]
                            if isinstance(element, torch.Tensor):
                                element = element.detach().to("cpu").contiguous()
                            mm_payload[k] = element
                    except Exception as e:
                        logger.error(f"Error in merge multimodal outputs: {e}")
                if mm_payload:
                    payload.update(mm_payload)
            pooler_output.append(payload)
        with record_function_or_nullcontext("gpu_model_runner: ModelRunnerOutput"):
            output = OmniModelRunnerOutput(
                req_ids=req_ids_output_copy,
                req_id_to_index=req_id_to_index_output_copy,
                sampled_token_ids=valid_sampled_token_ids,
                logprobs=logprobs_lists,
                prompt_logprobs_dict=prompt_logprobs_dict,
                pooler_output=(pooler_output if self.vllm_config.model_config.engine_output_type != "text" else None),
                kv_connector_output=kv_connector_output,
                ec_connector_output=ec_connector_output if self.supports_mm_inputs else None,
                num_nans_in_logits=num_nans_in_logits,
            )
            output.kv_extracted_req_ids = kv_extracted_req_ids

        if not self.use_async_scheduling:
            return output
        with record_function_or_nullcontext("gpu_model_runner: AsyncGPUModelRunnerOutput"):
            async_output = AsyncGPUModelRunnerOutput(
                model_runner_output=output,
                sampled_token_ids=sampler_output.sampled_token_ids,
                logprobs_tensors=sampler_output.logprobs_tensors,
                invalid_req_indices=invalid_req_indices,
                async_output_copy_stream=self.async_output_copy_stream,
                vocab_size=self.input_batch.vocab_size,
            )
        with record_function_or_nullcontext("gpu_model_runner: set_async_sampled_token_ids"):
            # Save ref of sampled_token_ids CPU tensor if the batch contains
            # any requests with sampling params that require output ids.
            self.input_batch.set_async_sampled_token_ids(
                async_output.sampled_token_ids_cpu,
                async_output.async_copy_ready_event,
            )

        return async_output

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
                merged[k] = [
                    (item.detach().to("cpu").contiguous() if isinstance(item, torch.Tensor) else item) for item in v
                ]
            else:
                merged[k] = v
        setattr(req_state, "additional_information_cpu", merged)

    def _handle_finished_requests_kv_transfer(self, scheduler_output: SchedulerOutput) -> list[str]:
        """Handle KV cache transfer for finished requests asynchronously."""
        # Get finished requests that need KV transfer
        finished_reqs = getattr(scheduler_output, "finished_requests_needing_kv_transfer", {})
        if not finished_reqs:
            return []

        logger.debug(f"Processing KV cache transfer for finished requests: {finished_reqs.keys()}")

        extracted_ids = []
        transfer_data_list = []

        # 1. Sync Phase: Copy from GPU to CPU
        # Must be done in the current execution window while blocks are valid
        # because the Scheduler is waiting for our confirmation to free them.
        for req_id, data in finished_reqs.items():
            try:
                # Extract data (GPU -> CPU)
                # finished_reqs is now {req_id: {"seq_len": int, "block_ids": list[int]}}
                kv_data_dict = self._extract_kv_cache_for_requests({req_id: data})

                if req_id in kv_data_dict:
                    kv_data = kv_data_dict[req_id]
                    # Ensure all tensors in kv_data are on CPU
                    self._move_kv_data_to_cpu(kv_data)

                    transfer_data_list.append(kv_data)
            except Exception as e:
                logger.error(f"Failed to extract KV for {req_id}: {e}")
                import traceback

                traceback.print_exc()
            finally:
                # Always mark as extracted so Scheduler can free blocks.
                # Even if extraction failed, we can't let the block leak.
                extracted_ids.append(req_id)

        # 2. Transfer Phase: Network Transfer (CPU -> Remote)
        # Use synchronous transfer to ensure process doesn't exit before transfer completes
        if transfer_data_list:
            self._async_batch_transfer(transfer_data_list)

        return extracted_ids

    def _move_kv_data_to_cpu(self, kv_data: KVCacheTransferData) -> None:
        """Ensure all tensors in KV data are on CPU."""
        new_layer_blocks = {}
        for layer_name, blocks in kv_data.layer_blocks.items():
            if isinstance(blocks, torch.Tensor) and blocks.is_cuda:
                new_layer_blocks[layer_name] = blocks.detach().cpu().contiguous()
            else:
                new_layer_blocks[layer_name] = blocks
        kv_data.layer_blocks = new_layer_blocks

    def _async_batch_transfer(self, data_list: list[KVCacheTransferData]) -> None:
        """Worker function for asynchronous KV cache transfer."""
        for kv_data in data_list:
            try:
                # Transfer via OmniConnector
                self._transfer_kv_cache_via_omni({kv_data.request_id: kv_data})
            except Exception as e:
                logger.error(f"Error in async KV transfer for {kv_data.request_id}: {e}")

    def _async_kv_transfer_worker(self, req_ids: set[str]) -> None:
        # Deprecated, kept for compatibility if needed or removed
        pass

    def _extract_kv_cache_for_requests(self, req_data: dict[str, dict]) -> dict[str, any]:
        # TODO(wzliu)! Optimize kv cache transfer using rdma
        """Extract KV cache data for specific requests using provided block IDs."""
        result = {}

        for req_id, data in req_data.items():
            if isinstance(data, int):
                logger.error(f"Legacy call to _extract_kv_cache_for_requests for {req_id} not supported")
                continue

            seq_len = data.get("seq_len", 0)
            block_ids = data.get("block_ids", [])

            if not block_ids:
                logger.warning(f"Request {req_id} has no block IDs, skipping KV transfer")
                continue

            # Extract KV cache blocks for this request
            layer_blocks = {}
            for layer_idx, kv_cache_item in enumerate(self.kv_caches):
                try:
                    # with shape [2, num_blocks, block size, kv heads, head size]
                    if isinstance(kv_cache_item, torch.Tensor) and kv_cache_item.dim() == 5:
                        # return [2, seq_len, n_heads, head_dim]
                        combined_kv = self._extract_blocks_from_kv_tensor(kv_cache_item, block_ids, seq_len)
                        layer_blocks[f"{layer_idx}_k"] = combined_kv[0]  # [seq_len, 4, 128]
                        layer_blocks[f"{layer_idx}_v"] = combined_kv[1]  # [seq_len, 4, 128]
                    # for kv list (k_cache, v_cache)
                    elif isinstance(kv_cache_item, (tuple, list)) and len(kv_cache_item) == 2:
                        k_data = self._extract_blocks_from_kv_tensor(kv_cache_item[0], block_ids, seq_len)
                        v_data = self._extract_blocks_from_kv_tensor(kv_cache_item[1], block_ids, seq_len)
                        layer_blocks[f"{layer_idx}_k"] = k_data
                        layer_blocks[f"{layer_idx}_v"] = v_data
                    else:
                        logger.warning(f"Unexpected kv_cache structure at layer {layer_idx}: {type(kv_cache_item)}")
                        continue

                except Exception as e:
                    logger.error(f"Failed to extract KV blocks for layer {layer_idx}, request {req_id}: {e}")
                    continue

            if layer_blocks:
                metadata = self._get_kv_cache_metadata()
                metadata.update(
                    {
                        "kv_lens": [seq_len],
                        "ropes": [0],
                        "seq_len": seq_len,
                    }
                )

                kv_data = KVCacheTransferData(
                    request_id=req_id,
                    layer_blocks=layer_blocks,
                    block_ids=block_ids,
                    metadata=metadata,
                )
                result[req_id] = kv_data
                logger.debug(f"Extracted KV cache for {req_id}, {len(layer_blocks)} items, len={seq_len}")

        return result

    def _extract_blocks_from_kv_tensor(
        self, kv_tensor: torch.Tensor, block_ids: list[int], seq_len: int
    ) -> torch.Tensor:
        """Extract specific blocks and reconstruct them into a logical sequence."""

        # 5D: [2, num_blocks, block_size, n_heads, head_dim] -> current shape
        # 4D: [num_blocks, block_size, n_heads, head_dim] ->  single KV

        is_5d = kv_tensor.dim() == 5
        block_dim = 1 if is_5d else 0

        max_block_id = kv_tensor.shape[block_dim] - 1
        valid_block_ids = [bid for bid in block_ids if 0 <= bid <= max_block_id]

        if not valid_block_ids:
            raise ValueError(f"No valid block IDs. Max ID: {max_block_id}, Requested: {block_ids}")

        # 1. extract blocks
        if is_5d:
            # result shape: [2, len(valid_block_ids), 16, 4, 128]
            selected = kv_tensor[:, valid_block_ids]
        else:
            # result shape: [len(valid_block_ids), 16, 4, 128]
            selected = kv_tensor[valid_block_ids]

        # 2. reshape (Flatten)
        if is_5d:
            # [2, n_blocks, 16, 4, 128] -> [2, n_blocks * 16, 4, 128]
            n_kv, n_blks, blk_sz, n_heads, d_head = selected.shape
            flat = selected.reshape(n_kv, n_blks * blk_sz, n_heads, d_head)
            # get actual seq_len
            if seq_len <= flat.shape[1]:
                flat = flat[:, :seq_len]
        else:
            # [n_blocks, 16, 4, 128] -> [n_blocks * 16, 4, 128]
            n_blks, blk_sz, n_heads, d_head = selected.shape
            flat = selected.reshape(n_blks * blk_sz, n_heads, d_head)
            if seq_len <= flat.shape[0]:
                flat = flat[:seq_len]

        return flat.contiguous()

    def _get_kv_cache_metadata(self) -> dict[str, Any]:
        """Get metadata about the KV cache for transfer."""
        return {
            "block_size": self.cache_config.block_size,
            "num_layers": len(self.kv_caches),
            "dtype": str(self.cache_config.cache_dtype),
            "device": str(self.device),
        }

    def _transfer_kv_cache_via_omni(self, kv_transfer_data: dict[str, KVCacheTransferData]) -> None:
        """Transfer KV cache data via OmniConnector."""
        try:
            # Import here to avoid circular imports
            from vllm_omni.distributed.omni_connectors.factory import OmniConnectorFactory
            from vllm_omni.distributed.omni_connectors.utils.config import ConnectorSpec

            # Get connector configuration from system config
            connector_config = self._get_omni_connector_config()
            if not connector_config:
                logger.warning("No OmniConnector config found, skipping KV transfer")
                return

            if isinstance(connector_config, dict):
                c_type = connector_config.get("type")
                if not c_type:
                    logger.error("OmniConnector config missing 'type' field")
                    return
                c_extra = {k: v for k, v in connector_config.items() if k != "type"}
                connector_spec = ConnectorSpec(name=c_type, extra=c_extra)
            else:
                # Should not happen based on _get_omni_connector_config type hint
                logger.error(f"Unexpected OmniConnector config type: {type(connector_config)}")
                return

            connector = (
                self.omni_connector if self.omni_connector else OmniConnectorFactory.create_connector(connector_spec)
            )

            for req_id, kv_data in kv_transfer_data.items():
                logger.info(f"Transferring KV cache for request {req_id}")

                # Convert to dict for serialization
                data_dict = kv_data.to_dict()

                # Detect stages and send via OmniConnector with retry
                from_stage, to_stage = self._detect_transfer_stages()
                success, size, metadata = self._transfer_with_retry(
                    connector, from_stage, to_stage, f"kv_cache_{req_id}", data_dict
                )

                if success:
                    logger.info(f"Successfully transferred KV cache for {req_id}, size: {size} bytes")
                else:
                    logger.error(f"Failed to transfer KV cache for {req_id} after retries")

        except Exception as e:
            logger.error(f"Error during KV cache transfer: {e}")
            import traceback

            traceback.print_exc()

    def _get_omni_connector_config(self) -> dict[str, Any] | None:
        # TODO(wzliu)! get real connector from yaml file instead of hardcode
        """Get OmniConnector configuration from system config."""
        try:
            # Try to get from vLLM config first (if configured for KV transfer)
            if hasattr(self.vllm_config, "kv_transfer_config") and self.vllm_config.kv_transfer_config:
                kv_config = self.vllm_config.kv_transfer_config
                if hasattr(kv_config, "omni_connector_config"):
                    return kv_config.omni_connector_config

            # Fallback: try to get from environment or default config
            # TODO: Implement proper config loading from stage configuration
            import os

            if os.getenv("OMNI_CONNECTOR_TYPE"):
                return {
                    "type": os.getenv("OMNI_CONNECTOR_TYPE"),
                    "host": os.getenv("OMNI_CONNECTOR_HOST", "127.0.0.1"),
                    "metadata_server": os.getenv("OMNI_CONNECTOR_METADATA_SERVER", "http://127.0.0.1:8080/metadata"),
                    "master": os.getenv("OMNI_CONNECTOR_MASTER", "127.0.0.1:50051"),
                }

            # Default fallback for testing
            logger.warning("Using default OmniConnector config for testing")
            return {
                "type": "MooncakeConnector",  # Use Mooncake for testing
                "host": "127.0.0.1",
                "metadata_server": "http://127.0.0.1:8080/metadata",
                "master": "127.0.0.1:50051",
            }

        except Exception as e:
            logger.error(f"Error getting OmniConnector config: {e}")
            return None

    def _detect_transfer_stages(self) -> tuple[str, str]:
        """Detect the source and target stages for KV transfer."""
        try:
            # Try to detect from KV transfer config
            if hasattr(self.vllm_config, "kv_transfer_config") and self.vllm_config.kv_transfer_config:
                kv_config = self.vllm_config.kv_transfer_config
                kv_role = getattr(kv_config, "kv_role", None)
                if kv_role == "kv_producer":
                    return "prefill", "decode"
                elif kv_role == "kv_consumer":
                    return "decode", "prefill"
                elif kv_role == "kv_both":
                    # For kv_both, we need to determine direction based on context
                    # TODO: Implement smarter stage detection
                    return "prefill", "decode"

            # Fallback based on environment or simple heuristics
            import os

            from_stage = os.getenv("VLLM_STAGE", "prefill")
            if from_stage == "prefill":
                to_stage = "decode"
            else:
                to_stage = "prefill"

            return from_stage, to_stage

        except Exception as e:
            logger.error(f"Error detecting transfer stages: {e}")
            return "prefill", "decode"  # Default fallback

    def _transfer_with_retry(
        self,
        connector: Any,
        from_stage: str,
        to_stage: str,
        request_id: str,
        data: dict[str, Any],
        max_retries: int = 3,
        retry_delay: float = 0.1,
    ) -> tuple[bool, int, dict[str, Any] | None]:
        """Transfer data with retry mechanism."""
        import time

        for attempt in range(max_retries):
            try:
                success, size, metadata = connector.put(
                    from_stage=from_stage, to_stage=to_stage, request_id=request_id, data=data
                )
                # TODO(wzliu)! in offline mode + mooncake connectorif no sleep,
                # data actually not stored due to the exit of process
                time.sleep(20)

                if success:
                    return success, size, metadata
                else:
                    logger.warning(f"Transfer attempt {attempt + 1} failed for {request_id}")

            except Exception as e:
                logger.warning(f"Transfer attempt {attempt + 1} exception for {request_id}: {e}")

            # Wait before retry (exponential backoff)
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (2**attempt))

        return False, 0, None
