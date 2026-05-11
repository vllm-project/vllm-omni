# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import time
from copy import copy, deepcopy
from typing import Any, NamedTuple

import numpy as np
import torch
from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.config import CUDAGraphMode
from vllm.distributed.ec_transfer import get_ec_transfer, has_ec_transfer
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.forward_context import BatchDescriptor
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import RoutedExpertsCapturer
from vllm.sequence import IntermediateTensors
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    AsyncModelRunnerOutput,
    ECConnectorOutput,
    make_empty_encoder_model_runner_output,
)
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.structured_output.utils import apply_grammar_bitmask
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.worker import mamba_utils
from vllm.v1.worker.gpu_model_runner import AsyncGPUModelRunnerOutput, PerLayerAttnMetadata
from vllm.v1.worker.ubatch_utils import maybe_create_ubatch_slices
from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.compilation.acl_graph import ACLGraphWrapper

# yapf conflicts with isort for this block
# yapf: disable
from vllm_ascend.ops.rotary_embedding import update_cos_sin
from vllm_ascend.utils import enable_sp, global_stream
from vllm_ascend.worker.model_runner_v1 import graph_capture

from vllm_omni.data_entry_keys import flatten_payload
from vllm_omni.distributed.omni_connectors.kv_transfer_manager import OmniKVTransferManager
from vllm_omni.outputs import OmniModelRunnerOutput
from vllm_omni.platforms.npu.worker.npu_model_runner import OmniNPUModelRunner
from vllm_omni.utils.mm_outputs import build_mm_cpu, to_payload_element


class ExecuteModelState(NamedTuple):
    """Ephemeral cached state transferred between execute_model() and
    sample_tokens(), after execute_model() returns None."""

    scheduler_output: SchedulerOutput
    logits: torch.Tensor
    spec_decode_metadata: SpecDecodeMetadata | None
    spec_decode_common_attn_metadata: AscendCommonAttentionMetadata | None
    hidden_states: torch.Tensor
    sample_hidden_states: torch.Tensor
    aux_hidden_states: list[torch.Tensor] | None
    attn_metadata: PerLayerAttnMetadata
    positions: torch.Tensor
    ec_connector_output: ECConnectorOutput | None
    cudagraph_stats: CUDAGraphStat | None
    batch_desc: BatchDescriptor
    multimodal_outputs: Any # Omni-Specific

class NPUARModelRunner(OmniNPUModelRunner):
    """Autoregressive NPU model runner that returns hidden states per request."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_ids = self._make_buffer(self.max_num_tokens, dtype=torch.int32)
        # each model stage has their own hidden size
        self.hidden_size = self.model_config.hf_text_config.hidden_size
        self.inputs_embeds = self._make_buffer(self.max_num_tokens, self.hidden_size, dtype=self.dtype, numpy=False)
        # Initialize KV cache manager (preserve vllm_config fallback behavior)
        self.kv_transfer_manager = OmniKVTransferManager.from_vllm_config(self.vllm_config, self.model_config)
        self._downstream_payload_cache: dict[str, bool] = {}

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

    #  -------------------------------------- Omni-new -------------------------------------------------
    def capture_model(self) -> int:
        npugraph_memory_bytes = super().capture_model()
        self._capture_talker_mtp_graphs()
        return npugraph_memory_bytes

    def _capture_talker_mtp_graphs(self) -> None:
        if not self.has_talker_mtp or not isinstance(self.talker_mtp, ACLGraphWrapper):
            return

        from vllm.compilation.monitor import set_cudagraph_capturing_enabled

        capture_sizes = sorted(self.compilation_config.cudagraph_capture_sizes, reverse=True)
        num_warmups = self.compilation_config.cudagraph_num_of_warmups
        logger.info("Capturing talker_mtp graphs for sizes %s", capture_sizes)

        set_cudagraph_capturing_enabled(True)
        try:
            with torch.inference_mode(), graph_capture(device=self.device):
                for bsz in capture_sizes:
                    _, batch_desc, _, _, _ = self._determine_batch_execution_and_padding(
                        num_tokens=bsz,
                        num_reqs=bsz,
                        num_scheduled_tokens_np=np.ones(bsz, dtype=np.int32),
                        max_num_scheduled_tokens=1,
                        use_cascade_attn=False,
                    )
                    n = batch_desc.num_tokens
                    ids = self.talker_mtp_input_ids.gpu[:n]
                    emb = self.talker_mtp_inputs_embeds.gpu[:n]
                    hid = self.last_talker_hidden.gpu[:n]
                    ts = self.text_step.gpu[:n]

                    for _ in range(num_warmups):
                        with set_ascend_forward_context(
                            None,
                            self.vllm_config,
                            aclgraph_runtime_mode=CUDAGraphMode.NONE,
                            batch_descriptor=batch_desc,
                        ):
                            self.talker_mtp(ids, emb, hid, ts)

                    with set_ascend_forward_context(
                        None,
                        self.vllm_config,
                        aclgraph_runtime_mode=CUDAGraphMode.FULL,
                        batch_descriptor=batch_desc,
                    ):
                        self.talker_mtp(ids, emb, hid, ts)
                    torch.npu.synchronize()

            logger.info("Captured talker_mtp graphs for %d sizes", len(capture_sizes))
        except RuntimeError as e:
            raise RuntimeError(
                f"talker_mtp graph capture failed for a model that declared talker_mtp_graph_safe=True: {e}"
            ) from e
        finally:
            set_cudagraph_capturing_enabled(False)

    def _maybe_update_prefix_cache(
        self,
        hidden_states: torch.Tensor,
        multimodal_outputs: dict,
        num_tokens_unpadded: int,
        num_tokens_padded: int,
    ):
        if self.omni_prefix_cache is not None and get_pp_group().is_last_rank:
            if multimodal_outputs is not None and not isinstance(multimodal_outputs, dict):
                logger.warning_once(
                    "prefix caching expects mm outputs to be a dict, but got %s",
                    type(multimodal_outputs),
                )

            self.omni_prefix_cache.update_omni_tensor_prefix_cache(
                hidden_states=hidden_states,
                multimodal_outputs=multimodal_outputs,
                num_tokens_unpadded=num_tokens_unpadded,
                slot_mapping=self.input_batch.block_table[0].slot_mapping.cpu,
                num_tokens_padded=num_tokens_padded,
            )

    def _maybe_get_combined_prefix_cache_tensors(
        self,
        hidden_states: torch.Tensor,
        multimodal_outputs: dict,
        num_scheduled_tokens: dict[str, int],
    ) -> tuple[dict[str, torch.Tensor] | None, dict | None]:
        combined_hidden_states, combined_multimodal_outputs = None, None
        if self.omni_prefix_cache is not None:
            combined_hidden_states = self.omni_prefix_cache.get_merged_hidden_states(
                query_start_loc=self.query_start_loc.cpu,
                input_batch=self.input_batch,
                hidden_states=hidden_states,
                num_scheduled_tokens=num_scheduled_tokens,
            )
            combined_multimodal_outputs = self.omni_prefix_cache.get_merged_multimodal_states(
                query_start_loc=self.query_start_loc.cpu,
                input_batch=self.input_batch,
                multimodal_outputs=multimodal_outputs,
                num_scheduled_tokens=num_scheduled_tokens,
            )
        return combined_hidden_states, combined_multimodal_outputs

    @staticmethod
    def _resolve_req_hidden_states(
        hidden_states_cpu: torch.Tensor,
        combined_hidden_states: dict[str, torch.Tensor] | None,
        rid: str,
        start: int,
        end: int,
    ):
        if combined_hidden_states is not None:
            if rid not in combined_hidden_states:
                raise RuntimeError("Request IDs in the batch are missing from the merged states!")
            return combined_hidden_states[rid]
        return hidden_states_cpu[start:end]

    def _request_final_stage_id(self, req_id: str) -> int | None:
        info = self.model_intermediate_buffer.get(req_id)
        if not isinstance(info, dict):
            req_state = self.requests.get(req_id)
            info = getattr(req_state, "additional_information_cpu", None)
        if not isinstance(info, dict):
            return None
        val = info.get("omni_final_stage_id")
        try:
            return int(val)
        except (TypeError, ValueError):
            return None

    def _request_needs_downstream_stage_payload(self, req_id: str) -> bool:
        cached = self._downstream_payload_cache.get(req_id)
        if cached is not None:
            return cached
        final_stage_id = self._request_final_stage_id(req_id)
        needs_payload = final_stage_id is None or final_stage_id > 0
        self._downstream_payload_cache[req_id] = needs_payload
        return needs_payload

    def _resolve_pooler_payload_req_ids(self, req_ids_output_copy: list[str]) -> tuple[str, list[str]]:
        downstream_req_ids = [rid for rid in req_ids_output_copy if self._request_needs_downstream_stage_payload(rid)]
        engine_output_type = (self.vllm_config.model_config.engine_output_type or "").lower()
        if engine_output_type == "audio" and not downstream_req_ids:
            downstream_req_ids = req_ids_output_copy
        return engine_output_type, downstream_req_ids
    #  -------------------------------------- Omni-new -------------------------------------------------

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> OmniModelRunnerOutput | IntermediateTensors | None:
        if self.vllm_config.model_config.enable_return_routed_experts:
            capturer = RoutedExpertsCapturer.get_instance()
            if capturer is not None:
                capturer.clear_buffer()
            else:
                logger.warning("RoutedExpertsCapturer is not initialized.")
        if self.ascend_config.profiling_chunk_config.enabled:
            self._sync_device()
            self._execution_start_time = time.perf_counter()
        if self.execute_model_state is not None:
            raise RuntimeError("State error: sample_tokens() must be called after execute_model() returns None.")

        #  -------------------------------------- Omni-new -------------------------------------------------
        # [Omni] Handle KV transfer BEFORE updating states (which removes finished requests)
        if not getattr(self, "_warmup_state_cleared", False):
            self._warmup_state_cleared = True
            if hasattr(self.model, "_clear_warmup_state"):
                self.model._clear_warmup_state()

        # [Omni] Handle KV transfer BEFORE updating states (which removes finished requests)
        finished_reqs = getattr(scheduler_output, "finished_requests_needing_kv_transfer", {})
        if finished_reqs and hasattr(self.model, "get_kv_transfer_metadata"):
            for req_id, data in finished_reqs.items():
                try:
                    req_idx = self.input_batch.req_id_to_index.get(req_id)
                    num_computed = (
                        int(self.input_batch.num_computed_tokens_cpu[req_idx]) if req_idx is not None else None
                    )
                    model_meta = self.model.get_kv_transfer_metadata(
                        req_id,
                        num_computed_tokens=num_computed,
                    )
                    if model_meta:
                        existing = data.get("custom_metadata") or {}
                        existing.update(model_meta)
                        data["custom_metadata"] = existing
                except Exception as e:
                    logger.warning(f"Failed to get custom metadata from model for {req_id}: {e}")
        self.kv_extracted_req_ids = self.kv_transfer_manager.handle_finished_requests_kv_transfer(
            finished_reqs=finished_reqs,
            kv_caches=self.kv_caches,
            block_size=self.cache_config.block_size,
            cache_dtype=str(self.cache_config.cache_dtype),
            request_id_resolver=self._resolve_global_request_id,
        )
        #  -------------------------------------- Omni-new -------------------------------------------------
        # self._draft_token_ids is None when `input_fits_in_drafter=False`
        # and there is no draft tokens scheduled. so it need to update the
        # spec_decoding info in scheduler_output with async_scheduling.
        # use deepcopy to avoid the modification has influence on the
        # scheduler_output in engine core process.
        # TODO(Ronald1995): deepcopy is expensive when there is a large
        # number of requests, optimize it later.
        if ((
            self.use_async_scheduling and self.num_spec_tokens and self._draft_token_ids is None  # type: ignore[has-type]
        ) or (
            # NOTE: This branch specifically triggers a deepcopy during the prefill phase
            # only for PCP (Parallel Context Processing) + Multi-Modal (MM) scenarios.
            # It does not affect other use cases. This is a temporary workaround and
            # will be removed once upstream vLLM provides native support for PCP + MM.
            self.pcp_size > 1
            and self.supports_mm_inputs
            and get_pp_group().is_first_rank
            and not self.model_config.is_encoder_decoder
        )):
            scheduler_output = deepcopy(scheduler_output)

        #  -------------------------------------- Omni-new -------------------------------------------------
        if has_kv_transfer_group():
            kv_connector_metadata = scheduler_output.kv_connector_metadata
            if kv_connector_metadata is not None:
                get_kv_transfer_group().handle_preemptions(kv_connector_metadata)
        #  -------------------------------------- Omni-new -------------------------------------------------

        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        with record_function_or_nullcontext("prepare input"):
            with self.synchronize_input_prep():
                # Update persistent batch states.
                deferred_state_corrections_fn = self._update_states(scheduler_output)

                #  -------------------------------------- Omni-new -------------------------------------------------
                if scheduler_output.finished_req_ids and hasattr(self.model, "on_requests_finished"):
                    self.model.on_requests_finished(scheduler_output.finished_req_ids)
                #  -------------------------------------- Omni-new -------------------------------------------------

                if has_ec_transfer() and get_ec_transfer().is_producer:
                    with self.maybe_get_ec_connector_output(
                        scheduler_output,
                        encoder_cache=self.encoder_cache,
                    ) as ec_connector_output:
                        self._execute_mm_encoder(scheduler_output)

                        kv_ids = self.kv_extracted_req_ids
                        self.kv_extracted_req_ids = None

                        output = make_empty_encoder_model_runner_output(scheduler_output)
                        if kv_ids:
                            output = copy(output)
                            output.kv_extracted_req_ids = kv_ids
                        return output

                if not num_scheduled_tokens:
                    if (
                        self.parallel_config.distributed_executor_backend == "external_launcher"
                        and self.parallel_config.data_parallel_size > 1
                    ):
                        # this is a corner case when both external launcher
                        # and DP are enabled, num_scheduled_tokens could be
                        # 0, and has_unfinished_requests in the outer loop
                        # returns True. before returning early here we call
                        # dummy run to ensure coordinate_batch_across_dp
                        # is called into to avoid out of sync issues.
                        self._dummy_run(1)

                    kv_ids = self.kv_extracted_req_ids
                    self.kv_extracted_req_ids = None

                    if not has_kv_transfer_group():
                        output = EMPTY_MODEL_RUNNER_OUTPUT
                    else:
                        output = self.kv_connector_no_forward(scheduler_output, self.vllm_config)

                    if kv_ids:
                        output = copy(output)
                        output.kv_extracted_req_ids = kv_ids

                    return output
                if self.cache_config.kv_sharing_fast_prefill:
                    assert not self.num_prompt_logprobs, (
                        "--kv-sharing-fast-prefill produces incorrect "
                        "logprobs for prompt tokens, tokens, please disable "
                        "it when the requests need prompt logprobs"
                    )

                num_reqs = self.input_batch.num_reqs
                req_ids = self.input_batch.req_ids
                tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
                num_scheduled_tokens_np = np.array(tokens, dtype=np.int32)
                max_num_scheduled_tokens = int(num_scheduled_tokens_np.max())

                (
                    logits_indices,
                    spec_decode_metadata,
                    total_num_scheduled_tokens,
                ) = self._prepare_inputs(
                    scheduler_output,
                    num_scheduled_tokens_np,
                )

                num_tokens_unpadded = scheduler_output.total_num_scheduled_tokens
                if self.pcp_size > 1:
                    num_tokens_unpadded = self.pcp_manager.total_num_sampled_tokens_pcp
                cascade_attn_prefix_lens = None
                # Disable cascade attention when using microbatching (DBO)
                if self.cascade_attn_enabled and not self.parallel_config.enable_dbo:
                    # Pre-compute cascade attention prefix lengths
                    cascade_attn_prefix_lens = self._compute_cascade_attn_prefix_lens(
                        num_scheduled_tokens_np,
                        self.input_batch.num_computed_tokens_cpu[:num_reqs],
                        scheduler_output.num_common_prefix_blocks,
                    )

                (
                    cudagraph_mode,
                    batch_desc,
                    should_ubatch,
                    num_tokens_across_dp,
                    cudagraph_stats,
                ) = self._determine_batch_execution_and_padding(
                    num_tokens=num_tokens_unpadded,
                    num_reqs=num_reqs,
                    num_scheduled_tokens_np=num_scheduled_tokens_np,
                    max_num_scheduled_tokens=max_num_scheduled_tokens,
                    use_cascade_attn=cascade_attn_prefix_lens is not None,
                    force_eager=self.model_config.enforce_eager,
                    num_encoder_reqs=len(scheduler_output.scheduled_encoder_inputs),
                )

                logger.debug(
                    "Running batch with cudagraph_mode: %s, batch_descriptor: %s, "
                    "should_ubatch: %s, num_tokens_across_dp: %s",
                    cudagraph_mode,
                    batch_desc,
                    should_ubatch,
                    num_tokens_across_dp,
                )

                num_tokens_padded = batch_desc.num_tokens
                num_reqs_padded = batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs
                ubatch_slices, ubatch_slices_padded = maybe_create_ubatch_slices(
                    should_ubatch,
                    num_scheduled_tokens_np,
                    num_tokens_padded,
                    num_reqs_padded,
                    self.parallel_config.num_ubatches,
                )

                pad_attn = cudagraph_mode == CUDAGraphMode.FULL

                # NOTE(Angazenn): According to https://github.com/vllm-project/vllm/pull/30877,
                # there should be a corresponding 'postprocess_mamba'. However, it is called inside
                # '_update_states_after_model_execute', which is not overridden in vLLM-Ascend.
                # We simply utilize the implementation in vLLM.
                if self.cache_config.mamba_cache_mode == "align":
                    # preprocess_mamba reads req_state.num_computed_tokens (CPU)
                    # to decide copy operations, so we must apply deferred
                    # corrections before it runs.
                    if deferred_state_corrections_fn:
                        deferred_state_corrections_fn()
                        deferred_state_corrections_fn = None
                    mamba_utils.preprocess_mamba(
                        scheduler_output,
                        self.kv_cache_config,
                        self.cache_config,
                        self.mamba_state_idx,
                        self.input_batch,
                        self.requests,
                        self.compilation_config.static_forward_context,
                        self.model.get_mamba_state_copy_func(),
                        self._get_mamba_copy_bufs(),
                    )
                    # preprocess_mamba resets num_accepted_tokens_cpu to 1
                    # for requests whose state was copied to a new block.
                    # Re-sync to GPU so the mamba kernel reads from the
                    # correct initial state slot (init_token_idx = 0).
                    self.num_accepted_tokens.np[:num_reqs] = self.input_batch.num_accepted_tokens_cpu[:num_reqs]
                    self.num_accepted_tokens.copy_to_gpu(num_reqs)

                use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
                ubatch_slices_attn = ubatch_slices_padded if pad_attn else ubatch_slices

                if (
                    cudagraph_mode == CUDAGraphMode.FULL
                    or (enable_sp() and not self.model_config.use_mla)
                    and self.pcp_size * self.dcp_size == 1
                ):
                    # Currently, Graph Mode and SP will both pad num_tokens,
                    # Another possible condition is num_tokens_padded != num_tokens_unpadded
                    # but this scope is way too big and the consequences are unpredictable
                    num_reqs_padded = self._pad_query_start_loc_for_fia(
                        num_tokens_padded, num_reqs_padded, num_reqs, cudagraph_mode, batch_desc.num_reqs
                    )

                (attn_metadata, spec_decode_common_attn_metadata) = self._build_attention_metadata(
                    num_tokens=num_tokens_unpadded
                    if not (self.use_cp and self.pcp_manager.pcp_use_hybrid_attn)
                    else total_num_scheduled_tokens,
                    num_tokens_padded=num_tokens_padded,
                    num_reqs=num_reqs,
                    num_reqs_padded=num_reqs_padded,
                    max_query_len=max_num_scheduled_tokens,
                    ubatch_slices=ubatch_slices_attn,
                    logits_indices=logits_indices,
                    use_spec_decode=use_spec_decode,
                    num_scheduled_tokens=scheduler_output.num_scheduled_tokens,
                    num_scheduled_tokens_np=num_scheduled_tokens_np,
                    cascade_attn_prefix_lens=cascade_attn_prefix_lens,
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
                num_tokens_padded
                if not (self.use_cp and self.pcp_manager.pcp_use_hybrid_attn)
                else total_num_scheduled_tokens,
                intermediate_tensors,
            )

            #  -------------------------------------- Omni-new -------------------------------------------------
            if hasattr(self.model, "prepare_runner_inputs"):
                input_ids, positions = self.model.prepare_runner_inputs(
                    input_ids=input_ids,
                    positions=positions,
                    inputs_embeds=inputs_embeds,
                    req_ids=req_ids[:num_reqs],
                    num_computed_tokens=[int(self.input_batch.num_computed_tokens_cpu[i]) for i in range(num_reqs)],
                    num_scheduled_tokens=[int(num_scheduled_tokens_np[i]) for i in range(num_reqs)],
                    input_ids_buffer=self.input_ids.gpu[:num_tokens_padded],
                )
            #  -------------------------------------- Omni-new -------------------------------------------------

            # update global cos, sin
            update_cos_sin(positions)

        if self.dynamic_eplb:
            with record_function_or_nullcontext("EPLB weight D2D"):
                self.eplb_updator.forward_before()

        # Set cudagraph mode to none if calc_kv_scales is true.
        # KV scales calculation involves dynamic operations that are incompatible
        # with CUDA graph capture.
        if self.calculate_kv_scales:  # type: ignore[has-type]
            cudagraph_mode = CUDAGraphMode.NONE
            # Mark KV scales as calculated after the first forward pass
            self.calculate_kv_scales = False  # type: ignore[has-type]
        # prevent debugger is None
        if self.debugger is not None:
            dbg_cfg = getattr(self.debugger, "config", None)
            dump_level = str(getattr(dbg_cfg, "level", "L1")).upper() if dbg_cfg is not None else "L1"
            if dump_level in ("L0", "MIX"):
                self.debugger.start(model=self.model)
            else:
                self.debugger.start()
        if self.ascend_config.enable_async_exponential:
            self.sampler.do_async_exponential(
                b_s=logits_indices.shape[0],
                head_dim=self.model_config.get_vocab_size(),
                generators=self.input_batch.sampling_metadata.generators,
            )

        # Encoder-decoder models can only compile the pure decode steps where no
        # encoder inputs are present. Use eager for the first pass.
        num_encoder_reqs = len(scheduler_output.scheduled_encoder_inputs)
        has_encoder_input = self.model_config.is_encoder_decoder and num_encoder_reqs > 0

        # Run forward pass
        clear_kv_metadata = self.speculative_config is None
        with (
            record_function_or_nullcontext("forward"),
            set_ascend_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_tokens_padded,
                num_tokens_across_dp=num_tokens_across_dp,
                aclgraph_runtime_mode=cudagraph_mode,
                batch_descriptor=batch_desc,
                num_actual_tokens=scheduler_output.total_num_scheduled_tokens,
                model_instance=self.model,
                max_tokens_across_pcp=0 if self.pcp_size == 1 else self.pcp_manager.max_num_tokens_across_pcp,
                skip_compiled=has_encoder_input,
            ),
            self.maybe_get_kv_connector_output(
                scheduler_output,
                **(
                    {"defer_finalize": not clear_kv_metadata}
                ),
            ) as kv_connector_output,
        ):
            hidden_states = self._model_forward(
                num_tokens_padded, input_ids, positions, intermediate_tensors, inputs_embeds, **model_kwargs
            )
        with record_function_or_nullcontext("post process"):
            #  -------------------------------------- Omni-new -------------------------------------------------
            # [Omni] Map pending ropes metadata to req_ids.
            if hasattr(self.model, "flush_pending_metadata"):
                self.model.flush_pending_metadata(list(req_ids))

            hidden_states, multimodal_outputs = self.extract_multimodal_outputs(hidden_states)

            if multimodal_outputs is not None:
                keys_or_type = (
                    list(multimodal_outputs.keys())
                    if isinstance(multimodal_outputs, dict)
                    else type(multimodal_outputs)
                )
                logger.debug(f"[AR] execute_model: multimodal_outputs keys = {keys_or_type}")
            else:
                logger.debug("[AR] execute_model: multimodal_outputs is None")
            #  -------------------------------------- Omni-new -------------------------------------------------
            aux_hidden_states = None
            if self.use_aux_hidden_state_outputs:
                hidden_states, aux_hidden_states = hidden_states
            if self.pcp_size > 1:
                # NOTE we must `slice` hidden_states because pcp_allgather_restore_idx
                # ignores the padding from CUDA Graph.
                hidden_states = self.pcp_manager.get_restore_hidden_states(hidden_states)
                if aux_hidden_states is not None:
                    aux_hidden_states = [
                        self.pcp_manager.get_restore_hidden_states(aux_hidden_states_pcp)
                        for aux_hidden_states_pcp in aux_hidden_states
                    ]

            #  -------------------------------------- Omni-new -------------------------------------------------
            self._maybe_update_prefix_cache(
                hidden_states=hidden_states,
                multimodal_outputs=multimodal_outputs,
                num_tokens_unpadded=num_tokens_unpadded,
                num_tokens_padded=num_tokens_padded,
            )
            #  -------------------------------------- Omni-new -------------------------------------------------

            if not self.broadcast_pp_output:
                # Common case.
                if not get_pp_group().is_last_rank:
                    # Return the intermediate tensors.
                    assert isinstance(hidden_states, IntermediateTensors)
                    hidden_states.kv_connector_output = kv_connector_output
                    self.kv_connector_output = kv_connector_output
                    if self.debugger is not None:
                        self.debugger.stop()
                        self.debugger.step()
                    return hidden_states
                if self.is_pooling_model:
                    # Return the pooling output.
                    output = self._pool(
                        hidden_states, num_scheduled_tokens, num_scheduled_tokens_np, kv_connector_output
                    )
                    output.kv_connector_output = kv_connector_output
                    if self.debugger is not None:
                        self.debugger.stop()
                        self.debugger.step()
                    return output

                sample_hidden_states = hidden_states[logits_indices]
                #  -------------------------------------- Omni-new -------------------------------------------------
                # Try with sampling_metadata first; fall back to without for models that don't support it
                try:
                    logits = self.model.compute_logits(
                        sample_hidden_states, sampling_metadata=self.input_batch.sampling_metadata
                    )
                except TypeError:
                    logits = self.model.compute_logits(sample_hidden_states)
                #  -------------------------------------- Omni-new -------------------------------------------------
            else:
                # Rare case.
                assert not self.is_pooling_model

                if not get_pp_group().is_last_rank:
                    sample_hidden_states = hidden_states[logits_indices]
                    get_pp_group().send_tensor_dict(hidden_states.tensors, all_gather_group=get_tp_group())
                    logits = None
                else:
                    sample_hidden_states = hidden_states[logits_indices]
                    #  -------------------------------------- Omni-new -------------------------------------------------
                    # Try with sampling_metadata first; fall back to without for models that don't support it
                    try:
                        logits = self.model.compute_logits(
                            sample_hidden_states, sampling_metadata=self.input_batch.sampling_metadata
                        )
                    except TypeError:
                        logits = self.model.compute_logits(sample_hidden_states)
                    #  -------------------------------------- Omni-new -------------------------------------------------

                model_output_broadcast_data: dict[str, Any] = {}
                if logits is not None:
                    model_output_broadcast_data["logits"] = logits.contiguous()
                broadcasted = get_pp_group().broadcast_tensor_dict(
                    model_output_broadcast_data, src=len(get_pp_group().ranks) - 1
                )
                assert broadcasted is not None
                logits = broadcasted["logits"]

            # Apply structured output bitmasks if present
            self.execute_model_state = ExecuteModelState(
                scheduler_output,
                logits,
                spec_decode_metadata,
                spec_decode_common_attn_metadata,
                hidden_states,
                sample_hidden_states,
                aux_hidden_states,
                attn_metadata,
                positions,
                ec_connector_output,
                cudagraph_stats,
                batch_desc,
                multimodal_outputs, # Omni-specific
            )
            self.kv_connector_output = kv_connector_output

        # Now the batch has been launched we can wait for corrections from the
        # previous model forward without breaking async scheduling.
        if deferred_state_corrections_fn:
            deferred_state_corrections_fn()
        return None

    @torch.inference_mode()
    def sample_tokens(
        self, grammar_output: GrammarOutput | None
    ) -> OmniModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors:
        kv_connector_output = self.kv_connector_output
        self.kv_connector_output = None

        #  -------------------------------------- Omni-new -------------------------------------------------
        kv_extracted_req_ids = getattr(self, "kv_extracted_req_ids", None)
        self.kv_extracted_req_ids = None
        combined_hidden_states = None
        combined_multimodal_outputs = None
        mm_cpu = {}
        #  -------------------------------------- Omni-new -------------------------------------------------


        if self.execute_model_state is None:
            # Nothing to do (PP non-final rank case), output isn't used.
            # receive sampled token ids from the last PP rank when using
            # async scheduling + pipeline parallelism so downstream code
            # (e.g., PCP input preparation) can access them.
            if self.use_async_scheduling and get_pp_group().world_size > 1:
                self._pp_receive_prev_sampled_token_ids_to_input_batch()
            if not kv_connector_output:
                return None  # noqa
            # In case of PP with kv transfer, we need to pass through the
            # kv_connector_output
            if kv_connector_output.is_empty():
                return EMPTY_MODEL_RUNNER_OUTPUT

            output = copy(EMPTY_MODEL_RUNNER_OUTPUT)
            output.kv_connector_output = kv_connector_output
            return output

        # Unpack ephemeral state.
        (
            scheduler_output,
            logits,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            attn_metadata,
            positions,
            ec_connector_output,
            cudagraph_stats,
            batch_desc,
            multimodal_outputs, # Omni-Specific
        ) = self.execute_model_state
        # Clear ephemeral state.
        self.execute_model_state = None
        seq_len = hidden_states.shape[0]

        # Apply structured output bitmasks if present.
        if grammar_output is not None:
            # here we are different from gpu_model_runner,
            # the apply_grammar_bitmask uses torch.compile to optimize this,ascend does not support it now
            logits_dtype = logits.dtype
            logits = logits.to("cpu").float()
            apply_grammar_bitmask(scheduler_output, grammar_output, self.input_batch, logits)
            logits = logits.to(self.device).to(logits_dtype)

        #  -------------------------------------- Omni-new -------------------------------------------------
        # Correct padding values of prompt_token_ids to match the logits vocabulary size.
        if logits is not None and not self.input_batch.sampling_metadata.no_penalties:
            smd = self.input_batch.sampling_metadata
            if smd.prompt_token_ids is not None:
                logits_vocab = logits.shape[-1]
                if self.input_batch.vocab_size > logits_vocab:
                    smd.prompt_token_ids = smd.prompt_token_ids.clamp(max=logits_vocab)
        #  -------------------------------------- Omni-new -------------------------------------------------


        with record_function_or_nullcontext("sample_token"):
            sampler_output = self._sample(logits, spec_decode_metadata)

        if self.need_accepted_tokens:
            if self.sampling_done_event is None:
                self.sampling_done_event = torch.npu.Event()

            assert self.sampling_done_event is not None
            self.sampling_done_event.record()

        self.valid_sampled_token_count_gpu: torch.Tensor | None = None # type: ignore[no-redef]

        def propose_draft_token_ids(sampled_token_ids):
            assert spec_decode_common_attn_metadata is not None
            self._draft_token_ids = self.propose_draft_token_ids(
                sampled_token_ids,
                self.input_batch.sampling_metadata,
                scheduler_output,
                spec_decode_metadata,
                spec_decode_common_attn_metadata,
                positions,
                scheduler_output.total_num_scheduled_tokens,
                hidden_states,
                aux_hidden_states,
                sample_hidden_states,
                batch_desc,
            )
            self._copy_draft_token_ids_to_cpu(scheduler_output)

        (
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

        with record_function_or_nullcontext("draft_token"):
            if self.speculative_config:
                use_padded_batch = (
                    self.speculative_config
                    and (self.speculative_config.use_eagle() or self.speculative_config.uses_draft_model())
                    and not self.speculative_config.disable_padded_drafter_batch
                )
                if use_padded_batch:
                    # EAGLE speculative decoding can use the GPU sampled tokens
                    # as inputs, and does not need to wait for bookkeeping to finish.
                    propose_draft_token_ids(sampler_output.sampled_token_ids)
                if self.speculative_config and not use_padded_batch:
                    # ngram and other speculative decoding methods use the sampled
                    # tokens on the CPU, so they are run after bookkeeping.
                    propose_draft_token_ids(valid_sampled_token_ids)

            # vLLM v0.18 defers KV connector finalization during target-model
            # forward when speculative decoding is enabled. Finalize here after
            # draft model runs so KV pool save/put can complete.
            if self.speculative_config is not None:
                self.finalize_kv_connector()

        if self.model_config.enable_return_routed_experts:
            capturer = RoutedExpertsCapturer.get_instance()
            if capturer is not None:
                capturer.save_captured_experts(indices=self.cpu_slot_mapping)
            else:
                logger.warning("RoutedExpertsCapturer is not initialized.")

        #  -------------------------------------- Omni-new -------------------------------------------------
        engine_output_type, downstream_req_ids = self._resolve_pooler_payload_req_ids(req_ids_output_copy)
        needs_pooler_payload = len(downstream_req_ids) > 0
        downstream_req_id_set = set(downstream_req_ids)
        hidden_states_cpu = None
        req_hidden_states_cpu: dict[str, torch.Tensor] | None = None
        if needs_pooler_payload:
            num_valid_tokens = min(
                int(scheduler_output.total_num_scheduled_tokens),
                int(hidden_states.shape[0]),
            )
            if len(downstream_req_ids) == len(req_ids_output_copy):
                hidden_states_cpu = hidden_states[:num_valid_tokens].detach().to("cpu").contiguous()
            else:
                req_hidden_states_cpu = {}
        num_scheduled_tokens_np = getattr(self, "_omni_num_scheduled_tokens_np", None)
        if num_scheduled_tokens_np is None:
            req_ids = self.input_batch.req_ids
            num_scheduled_tokens_np = np.array(
                [scheduler_output.num_scheduled_tokens[rid] for rid in req_ids],
                dtype=np.int32,
            )
        query_start_loc_cpu = self.query_start_loc.cpu

        pooler_output: list[dict[str, object]] | None = None
        if needs_pooler_payload:
            if self.omni_prefix_cache is not None:
                (
                    combined_hidden_states,
                    combined_multimodal_outputs,
                ) = self._maybe_get_combined_prefix_cache_tensors(
                    hidden_states,
                    multimodal_outputs,
                    scheduler_output.num_scheduled_tokens,
                )
            else:
                mm_cpu = build_mm_cpu(flatten_payload(multimodal_outputs))

            self._process_additional_information_updates(
                hidden_states,
                multimodal_outputs,
                num_scheduled_tokens_np,
                scheduler_output,
                combined_hidden_states,
                combined_multimodal_outputs,
                req_ids_filter=downstream_req_id_set,
            )

            if req_hidden_states_cpu is not None and combined_hidden_states is None:
                for rid in downstream_req_ids:
                    idx = req_id_to_index_output_copy[rid]
                    start = int(query_start_loc_cpu[idx])
                    sched = int(num_scheduled_tokens_np[idx])
                    end = start + sched
                    req_hidden_states_cpu[rid] = hidden_states[start:end].detach().to("cpu").contiguous()

            pooler_output = []
            for rid in req_ids_output_copy:
                if rid not in downstream_req_id_set:
                    pooler_output.append({})
                    continue
                idx = req_id_to_index_output_copy[rid]
                start = int(query_start_loc_cpu[idx])
                sched = int(num_scheduled_tokens_np[idx])
                end = start + sched
                if req_hidden_states_cpu is not None and combined_hidden_states is None:
                    req_hidden_states = req_hidden_states_cpu[rid]
                else:
                    req_hidden_states = self._resolve_req_hidden_states(
                        hidden_states_cpu,
                        combined_hidden_states,
                        rid,
                        start,
                        end,
                    )
                payload: dict[str, object] = {"hidden": req_hidden_states}
                mm_payload: dict[str, object] = {}
                if combined_multimodal_outputs or mm_cpu:
                    if combined_multimodal_outputs:
                        for mm_key in combined_multimodal_outputs.keys():
                            value = combined_multimodal_outputs[mm_key][rid]
                            if isinstance(value, list):
                                mm_payload[mm_key] = value[idx] if idx < len(value) else value[0]
                            else:
                                mm_payload[mm_key] = value
                    else:
                        for mm_key, mm_val in mm_cpu.items():
                            mm_payload[mm_key] = to_payload_element(
                                element=mm_val,
                                idx=idx,
                                start=start,
                                end=end,
                                pass_lists_through=False,
                                seq_len=seq_len,
                            )
                    payload.update(mm_payload)
                pooler_output.append(flatten_payload(payload))

        model_runner_output = OmniModelRunnerOutput(
            req_ids=req_ids_output_copy,
            req_id_to_index=req_id_to_index_output_copy,
            sampled_token_ids=valid_sampled_token_ids,
            logprobs=logprobs_lists,
            prompt_logprobs_dict=prompt_logprobs_dict,
            pooler_output=(pooler_output if engine_output_type != "text" and needs_pooler_payload else None),
            kv_connector_output=kv_connector_output,
            ec_connector_output=ec_connector_output if self.supports_mm_inputs else None,
            cudagraph_stats=cudagraph_stats,
        )
        model_runner_output.kv_extracted_req_ids = kv_extracted_req_ids
        #  -------------------------------------- Omni-new -------------------------------------------------

        if self.ascend_config.profiling_chunk_config.enabled and hasattr(self, "_execution_start_time"):
            self._sync_device()
            model_runner_output.execution_time_ms = (time.perf_counter() - self._execution_start_time) * 1000.0

        if self.dynamic_eplb:
            with record_function_or_nullcontext("EPLB update"):
                self.eplb_updator.forward_end()

        if self.debugger is not None:
            self.debugger.stop()
            self.debugger.step()

        if self.need_accepted_tokens:
            assert self.sampling_done_event is not None
            with (
                record_function_or_nullcontext("async_state_update"),
                torch.npu.stream(global_stream()),
            ):
                global_stream().wait_event(self.sampling_done_event)
                self._update_states_after_model_execute(sampler_output.sampled_token_ids, scheduler_output)

        # In async scheduling + PP, broadcast sampled token ids from the
        # last PP rank so other PP ranks can receive them without going
        # through the scheduler/engine IPC path.
        if self.use_async_scheduling:
            pp = get_pp_group()
            if pp.world_size > 1 and pp.is_last_rank:
                self._pp_broadcast_prev_sampled_token_ids(sampler_output.sampled_token_ids)

        if not self.use_async_scheduling:
            return model_runner_output
        async_output = AsyncGPUModelRunnerOutput(
            model_runner_output=model_runner_output,
            sampled_token_ids=sampler_output.sampled_token_ids,
            logprobs_tensors=sampler_output.logprobs_tensors,
            invalid_req_indices=invalid_req_indices,
            async_output_copy_stream=self.async_output_copy_stream,
            vocab_size=self.input_batch.vocab_size,
        )
        self.input_batch.set_async_sampled_token_ids(
            async_output.sampled_token_ids_cpu,
            async_output.async_copy_ready_event,
        )
        return async_output

    #  -------------------------------------- Omni-new -------------------------------------------------
    def _resolve_global_request_id(self, req_id: str) -> str:
        """Resolve global request ID from request state."""
        req_state = self.requests.get(req_id)
        if not req_state:
            return req_id

        add_info = self.model_intermediate_buffer.get(req_id, {})
        global_id = add_info.get("global_request_id")
        if global_id:
            if isinstance(global_id, list) and global_id:
                global_id = global_id[0]
            if isinstance(global_id, bytes):
                return global_id.decode("utf-8")
            return str(global_id)
        return req_id
    #  -------------------------------------- Omni-new -------------------------------------------------
