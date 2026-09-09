# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from copy import copy
from dataclasses import replace
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
from vllm_omni.model_executor.duplex_sampling import DuplexSamplingRunnerMixin
from vllm_omni.outputs import OmniModelRunnerOutput
from vllm_omni.platforms.npu.worker.npu_model_runner import OmniNPUModelRunner
from vllm_omni.worker.async_omni_output import (
    AsyncOmniOutputRunnerMixin,
    OmniAsyncGPUModelRunnerOutput,
)
from vllm_omni.worker.omni_connector_model_runner_mixin import (
    OmniConnectorModelRunnerMixin,
    needs_omni_connector,
)
from vllm_omni.worker.sampling_utils import sanitize_min_tokens_stop_ids


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

class NPUARModelRunner(
    OmniNPUModelRunner,
    OmniConnectorModelRunnerMixin,
    DuplexSamplingRunnerMixin,
    AsyncOmniOutputRunnerMixin,
):
    """Autoregressive NPU model runner that returns hidden states per request."""

    kv_extracted_req_ids: list[str] | None
    sampling_done_event: Any

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kv_extracted_req_ids = None
        self.input_ids = self._make_buffer(self.max_num_tokens, dtype=torch.int32)
        # each model stage has their own hidden size
        self.hidden_size = self.model_config.hf_text_config.hidden_size
        self.inputs_embeds = self._make_buffer(self.max_num_tokens, self.hidden_size, dtype=self.dtype, numpy=False)
        # Initialize KV cache manager (preserve vllm_config fallback behavior)
        self.kv_transfer_manager = OmniKVTransferManager.from_vllm_config(self.vllm_config, self.model_config)
        self._async_chunk = getattr(self.model_config, "async_chunk", False)
        if needs_omni_connector(self.model_config):
            self.init_omni_connectors(
                model_config=self.model_config,
                kv_transfer_manager=self.kv_transfer_manager,
            )
        self._downstream_payload_cache: dict[str, bool] = {}
        self._init_duplex_sampling_state()
        #  -------------------------------------- Omni-new -------------------------------------------------

    def load_model(self, *args, **kwargs) -> None:
        super().load_model(*args, **kwargs)
        self._resolve_duplex_sampling_hook(force=True)

    def _update_states(self, scheduler_output: SchedulerOutput):
        deferred_state_corrections_fn = super()._update_states(scheduler_output)
        self._update_duplex_sampling_states(scheduler_output)
        return deferred_state_corrections_fn

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

    def _model_needs_full_prefix_hidden_states(self) -> bool:
        """See gpu_ar_model_runner._model_needs_full_prefix_hidden_states."""
        model = getattr(self, "model", None)
        return bool(getattr(model, "requires_full_prefix_cached_hidden_states", True))

    def _deferred_prefix_cache_mm_keys(self) -> set[str]:
        """Model-declared multimodal keys whose prefix-cache writes are deferred."""
        model = getattr(self, "model", None)
        keys = getattr(model, "deferred_prefix_cache_mm_keys", ())
        return set(keys or ())

    def _stage_deferred_prefix_cache_mm_outputs(
        self,
        *,
        scheduler_output: SchedulerOutput,
        multimodal_outputs: Any,
        query_start_loc_cpu: Any,
    ) -> None:
        """See gpu_ar_model_runner._stage_deferred_prefix_cache_mm_outputs."""
        if self.omni_prefix_cache is None:
            return

        deferred_mm_cache_keys = self._deferred_prefix_cache_mm_keys()
        if not deferred_mm_cache_keys:
            return

        self.omni_prefix_cache.stage_deferred_mm_outputs(
            query_start_loc=query_start_loc_cpu,
            input_batch=self.input_batch,
            multimodal_outputs=flatten_payload(multimodal_outputs) if multimodal_outputs else multimodal_outputs,
            num_scheduled_tokens=scheduler_output.num_scheduled_tokens,
            deferred_mm_cache_keys=deferred_mm_cache_keys,
        )

    def _maybe_update_prefix_cache(
        self,
        hidden_states: torch.Tensor,
        multimodal_outputs: dict,
        num_tokens_unpadded: int,
        num_tokens_padded: int,
    ):
        if self.omni_prefix_cache is not None and get_pp_group().is_last_rank:
            hs_for_cache = hidden_states if self._model_needs_full_prefix_hidden_states() else None
            slot_mapping_gpu = self.input_batch.block_table[0].slot_mapping.gpu
            slot_mapping_cpu = slot_mapping_gpu[:num_tokens_padded].cpu()
            self.omni_prefix_cache.update_omni_tensor_prefix_cache(
                hidden_states=hs_for_cache,
                multimodal_outputs=flatten_payload(multimodal_outputs) if multimodal_outputs else multimodal_outputs,
                num_tokens_unpadded=num_tokens_unpadded,
                slot_mapping=slot_mapping_cpu,
                num_tokens_padded=num_tokens_padded,
                skip_mm_cache_keys=self._deferred_prefix_cache_mm_keys(),
            )

    def _maybe_get_combined_prefix_cache_tensors(
        self,
        hidden_states: torch.Tensor,
        hidden_states_cpu: torch.Tensor | None,  # GPU-compatible; unused on NPU merge path
        multimodal_outputs: dict,
        num_scheduled_tokens: dict[str, int],
    ) -> tuple[dict[str, torch.Tensor] | None, dict | None]:
        combined_hidden_states, combined_multimodal_outputs = None, None
        if self.omni_prefix_cache is not None:
            if (
                not self._model_needs_full_prefix_hidden_states()
                and not self.omni_prefix_cache.has_prefix_cached_new_req_ids()
            ):
                return None, None
            if self._model_needs_full_prefix_hidden_states():
                combined_hidden_states = self.omni_prefix_cache.get_merged_hidden_states(
                    query_start_loc=self.query_start_loc.cpu,
                    input_batch=self.input_batch,
                    hidden_states=hidden_states,
                    num_scheduled_tokens=num_scheduled_tokens,
                )
            combined_multimodal_outputs = self.omni_prefix_cache.get_merged_multimodal_states(
                query_start_loc=self.query_start_loc.cpu,
                input_batch=self.input_batch,
                multimodal_outputs=flatten_payload(multimodal_outputs) if multimodal_outputs else multimodal_outputs,
                num_scheduled_tokens=num_scheduled_tokens,
            )
        return combined_hidden_states, combined_multimodal_outputs

    #  -------------------------------------- Omni-new -------------------------------------------------

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> OmniModelRunnerOutput | IntermediateTensors | None:
        if self.vllm_config.model_config.enable_return_routed_experts:
            capturer = self.routed_experts_capturer
            if capturer is not None and hasattr(capturer, "finalize_pending_copy"):
                capturer.finalize_pending_copy()
        profiling_chunk_config = self.ascend_config.scheduler_config.profiling_chunk_config
        if profiling_chunk_config.enabled and profiling_chunk_config.need_timing:
            if getattr(scheduler_output, "disable_profiling_timing", False):
                profiling_chunk_config.need_timing = False
            else:
                self._sync_device()
                self._execution_start_time = time.perf_counter()
        if self.execute_model_state is not None:  # type: ignore[has-type]
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
        if hasattr(self, "_omni_connector"):
            for request in getattr(scheduler_output, "pending_input_registrations", []):
                self.register_chunk_recv(request)
            self.recv_full_payload_inputs(scheduler_output)
            if self._pending_full_payload_send:
                flush_ids = set(getattr(scheduler_output, "finished_req_ids", set()))
                flush_ids.update({rid for rid in self._pending_full_payload_send if rid not in self.requests})
                if flush_ids:
                    self.flush_full_payload_outputs(flush_ids)

        if self.omni_prefix_cache is not None and scheduler_output.finished_req_ids:
            self.omni_prefix_cache.commit_deferred_mm_outputs(
                set(scheduler_output.finished_req_ids),
                self.input_batch,
            )

        #  -------------------------------------- Omni-new -------------------------------------------------
        if self.speculative_config is not None and self.speculative_config.use_ngram_gpu():
            num_scheduled_tokens_copy = scheduler_output.num_scheduled_tokens.copy()
            spec_decode_tokens_copy = scheduler_output.scheduled_spec_decode_tokens.copy()
            scheduler_output = replace(
                scheduler_output,
                num_scheduled_tokens=num_scheduled_tokens_copy,
                scheduled_spec_decode_tokens=spec_decode_tokens_copy,
            )

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
                    self._start_dump_data(scheduled_tokens=scheduler_output.num_scheduled_tokens)
                    with self.maybe_get_ec_connector_output(
                        scheduler_output,
                        encoder_cache=self.encoder_cache,
                    ) as ec_connector_output:
                        self._execute_mm_encoder(scheduler_output)

                        kv_ids = self.kv_extracted_req_ids
                        self.kv_extracted_req_ids = None

                        self._finalize_dump_data()
                        output = make_empty_encoder_model_runner_output(scheduler_output)
                        if kv_ids:
                            output = copy(output)
                            output.kv_extracted_req_ids = kv_ids
                        return self.attach_omni_connector_output(output)

                # `<= 0`: upstream can schedule a negative span, which is truthy (#5196).
                if num_scheduled_tokens <= 0:
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

                    return self.attach_omni_connector_output(output)
                if self.cache_config.kv_sharing_fast_prefill:
                    assert not self.num_prompt_logprobs, (
                        "--kv-sharing-fast-prefill produces incorrect "
                        "logprobs for prompt tokens, tokens, please disable "
                        "it when the requests need prompt logprobs"
                    )

                self._start_dump_data(scheduled_tokens=scheduler_output.num_scheduled_tokens)
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

                if logger.isEnabledFor(logging.DEBUG):
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

                if self.dynamic_eplb:
                    self.update_eplb_heat_collection_status(num_tokens_padded)

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
                    mamba_bufs = self._get_mamba_bufs()
                    preprocess_bufs = mamba_bufs.preprocess
                    mamba_utils.preprocess_mamba(
                        scheduler_output,
                        self.kv_cache_config,
                        self.cache_config,
                        self.mamba_state_idx,
                        self.input_batch,
                        self.requests,
                        self.compilation_config.static_forward_context,
                        self.model.get_mamba_state_copy_func(),
                        preprocess_bufs,
                    )
                    # preprocess_mamba resets num_accepted_tokens_cpu to 1
                    # for requests whose state was copied to a new block.
                    # Re-sync to GPU so the mamba kernel reads from the
                    # correct initial state slot (init_token_idx = 0).
                    self.num_accepted_tokens.np[:num_reqs] = self.input_batch.num_accepted_tokens_cpu[:num_reqs]
                    self.num_accepted_tokens.copy_to_gpu(num_reqs)

                    if mamba_bufs.postprocess_align is not None:
                        mamba_utils.stage_postprocess_inputs_to_gpu(
                            mamba_bufs.postprocess_align,
                            scheduler_output,
                            self.input_batch.req_ids,
                            num_reqs,
                            self.requests,
                            self.mamba_state_idx,
                        )

                use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
                ubatch_slices_attn = ubatch_slices_padded if pad_attn else ubatch_slices

                if (
                    cudagraph_mode == CUDAGraphMode.FULL
                    or (enable_sp() and not self.model_config.use_mla)
                    and self.dcp_size == 1
                ):
                    # Currently, Graph Mode and SP will both pad num_tokens,
                    # Another possible condition is num_tokens_padded != num_tokens_unpadded
                    # but this scope is way too big and the consequences are unpredictable
                    num_reqs_padded = self._pad_query_start_loc_for_fia(
                        self.query_start_loc,
                        num_tokens_padded,
                        num_reqs_padded,
                        num_reqs,
                        cudagraph_mode,
                        batch_desc.num_reqs,
                    )

                (attn_metadata, spec_decode_common_attn_metadata) = self._build_attention_metadata(
                    num_tokens=num_tokens_unpadded,
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
                num_tokens_padded,
                intermediate_tensors,
            )

            #  -------------------------------------- Omni-new -------------------------------------------------
            if hasattr(self.model, "prepare_runner_inputs"):
                input_ids, positions = self.model.prepare_runner_inputs(
                    input_ids=input_ids,
                    positions=positions,
                    inputs_embeds=inputs_embeds,
                    req_ids=req_ids[:num_reqs],
                    num_computed_tokens=self.input_batch.num_computed_tokens_cpu[:num_reqs],
                    num_scheduled_tokens=num_scheduled_tokens_np[:num_reqs],
                    input_ids_buffer=self.input_ids.gpu[:num_tokens_padded],
                )
            #  -------------------------------------- Omni-new -------------------------------------------------

            # update global cos, sin
            update_cos_sin(positions)

        if self.dynamic_eplb:
            with record_function_or_nullcontext("EPLB weight D2D"):
                self.eplb_updator.forward_before()

        # Encoder-decoder models can only compile the pure decode steps where no
        # encoder inputs are present. Use eager for the first pass.
        num_encoder_reqs = len(scheduler_output.scheduled_encoder_inputs)
        has_encoder_input = self.model_config.is_encoder_decoder and num_encoder_reqs > 0

        # Run forward pass
        defer_kv_connector_finalize = self.speculative_config is not None and (
            get_pp_group().is_last_rank or self.broadcast_pp_output
        )
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
                skip_compiled=has_encoder_input,
                has_sinks=self._has_sinks,
                eplb_heat_collection_status=self.eplb_heat_collection_status if self.dynamic_eplb else False,
            ),
            self.maybe_get_kv_connector_output(
                scheduler_output,
                **(
                    {"defer_finalize": defer_kv_connector_finalize}
                ),
            ) as kv_connector_output,
        ):
            if self.cache_config.mamba_cache_mode == "align":
                mamba_utils.do_mamba_copy_block(preprocess_bufs)
            hidden_states = self._model_forward(
                num_tokens_padded, input_ids, positions, intermediate_tensors, inputs_embeds, **model_kwargs
            )
        with record_function_or_nullcontext("post process"):
            #  -------------------------------------- Omni-new -------------------------------------------------
            # [Omni] Map pending ropes metadata to req_ids.
            flush_pending_metadata = getattr(self.model, "flush_pending_metadata", None)
            if callable(flush_pending_metadata):
                flush_pending_metadata(req_ids[:num_reqs])

            # [Omni] Hand the model the batch's req_ids in logits order, for
            # models that gate logits per request. Mirrors gpu_ar_model_runner.
            # Only valid without spec decode: there logits_indices carries several
            # rows per request, so row i no longer corresponds to req_ids[i].
            if spec_decode_metadata is None:
                set_batch_req_ids = getattr(self.model, "set_batch_req_ids", None)
                if callable(set_batch_req_ids):
                    set_batch_req_ids(req_ids[:num_reqs])

            hidden_states, multimodal_outputs = self.extract_multimodal_outputs(hidden_states)

            if multimodal_outputs is not None:
                keys_or_type = (
                    list(multimodal_outputs.keys())
                    if isinstance(multimodal_outputs, Mapping)
                    else type(multimodal_outputs)
                )
                logger.debug(f"[AR] execute_model: multimodal_outputs keys = {keys_or_type}")
            else:
                logger.debug("[AR] execute_model: multimodal_outputs is None")
            #  -------------------------------------- Omni-new -------------------------------------------------
            aux_hidden_states = None
            if self.use_aux_hidden_state_outputs:
                hidden_states, aux_hidden_states = hidden_states

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
                    self._finalize_dump_data()
                    if self.dynamic_eplb:
                        self.eplb_updator.forward_end(self.eplb_heat_collection_status)
                    return hidden_states
                if self.is_pooling_model:
                    # Return the pooling output.
                    output = self._pool(
                        hidden_states, num_scheduled_tokens, num_scheduled_tokens_np, kv_connector_output
                    )
                    output.kv_connector_output = kv_connector_output
                    self._finalize_dump_data()
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

        if self.vllm_config.model_config.enable_return_routed_experts and hasattr(self, "_positions_cpu"):
            self._omni_routed_experts_d2h(scheduler_output)

        return None

    def _sample(
        self,
        logits: torch.Tensor | None,
        spec_decode_metadata: Any,
    ):
        sampling_metadata = self.input_batch.sampling_metadata
        if spec_decode_metadata is None:
            model_sample = getattr(self.model, "sample", None)
            self.input_batch.update_async_output_token_ids()
            if logits is not None and callable(model_sample) and getattr(self.model, "prefer_model_sampler", False):
                # Apply logit bias (min_tokens, allowed_token_ids) before
                # the custom model sampler — the standard GPU sampler does
                # this internally, but prefer_model_sampler bypasses it.
                if hasattr(self.sampler, "logit_bias_state"):
                    self.sampler.logit_bias_state.apply_logit_bias(
                        logits,
                        self.input_batch.expanded_idx_mapping,
                        self.input_batch.idx_mapping_np,
                        self.input_batch.positions[self.input_batch.logits_indices],
                    )
                prepared_sampling_metadata = self._sampling_metadata_for_model_sampler(sampling_metadata)
                self._apply_duplex_sampling(logits, prepared_sampling_metadata)
                sampler_output = model_sample(logits, prepared_sampling_metadata)
                if sampler_output is not None:
                    return sampler_output
            return self.sampler(
                logits=logits,
                sampling_metadata=sampling_metadata,
            )

        return super()._sample(logits, spec_decode_metadata)

    @torch.inference_mode()
    def sample_tokens(
        self, grammar_output: GrammarOutput | None
    ) -> OmniModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors:
        kv_connector_output = self.kv_connector_output
        self.kv_connector_output = None
        pp = get_pp_group()
        use_pp_spec_decode = self.speculative_config is not None and pp.world_size > 1

        #  -------------------------------------- Omni-new -------------------------------------------------
        kv_extracted_req_ids = getattr(self, "kv_extracted_req_ids", None)
        self.kv_extracted_req_ids = None
        #  -------------------------------------- Omni-new -------------------------------------------------


        if self.execute_model_state is None:
            # receive sampled token ids from the last PP rank.
            if self.use_async_scheduling and not pp.is_last_rank:
                self._pp_receive_prev_sampled_token_ids_to_input_batch()
            # Nothing to do (PP non-final rank case), output isn't used.
            if not kv_connector_output:
                return None  # noqa
            # In case of PP with kv transfer, we need to pass through the
            # kv_connector_output
            if kv_connector_output.is_empty():
                return self.attach_omni_connector_output(EMPTY_MODEL_RUNNER_OUTPUT)

            output = copy(EMPTY_MODEL_RUNNER_OUTPUT)
            output.kv_connector_output = kv_connector_output
            return self.attach_omni_connector_output(output)

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
        self.execute_model_state = None  # type: ignore[assignment]

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

        # Drop min-tokens stop ids the head cannot emit (e.g. the text
        # tokenizer EOS folded into all_stop_token_ids on a narrow codec
        # talker head); they would index_put_ out of bounds (#4962).
        if logits is not None:
            sanitize_min_tokens_stop_ids(
                self.input_batch.sampling_metadata.logitsprocs,
                logits.shape[-1],
            )
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

        output_spec_token_ids = None
        use_padded_batch = False
        early_pp_padded_drafter = False
        if self.speculative_config:
            use_padded_batch = (
                self.speculative_config.use_eagle()
                or self.speculative_config.uses_draft_model()
                or self.speculative_config.uses_extract_hidden_states()
                or self.speculative_config.use_ngram_gpu()
            ) and not self.speculative_config.disable_padded_drafter_batch
            early_pp_padded_drafter = use_pp_spec_decode and not self.use_async_scheduling and use_padded_batch
            if early_pp_padded_drafter:
                self._draft_token_ids = None
                self._draft_token_req_ids = None
                with record_function_or_nullcontext("draft_token"):
                    propose_draft_token_ids(sampler_output.sampled_token_ids)

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
                if not early_pp_padded_drafter:
                    self._draft_token_ids = None
                    self._draft_token_req_ids = None
                if use_padded_batch and not early_pp_padded_drafter:
                    # EAGLE speculative decoding can use the GPU sampled tokens
                    # as inputs, and does not need to wait for bookkeeping to finish.
                    propose_draft_token_ids(sampler_output.sampled_token_ids)
                if not use_padded_batch:
                    # ngram and other speculative decoding methods use the sampled
                    # tokens on the CPU, so they are run after bookkeeping.
                    propose_draft_token_ids(valid_sampled_token_ids)

            # vLLM v0.18 defers KV connector finalization during target-model
            # forward when speculative decoding is enabled. Finalize here after
            # draft model runs so KV pool save/put can complete.
            if self.speculative_config is not None:
                self.finalize_kv_connector()

            draft_token_ids = self._draft_token_ids if use_pp_spec_decode else None
            if draft_token_ids is not None:
                if isinstance(draft_token_ids, torch.Tensor):
                    num_draft_reqs = draft_token_ids.shape[0]
                    draft_ids_list = draft_token_ids[:num_draft_reqs].cpu().tolist()
                    draft_req_ids = self._draft_token_req_ids
                else:
                    draft_ids_list = draft_token_ids
                    draft_req_ids = self.input_batch.req_ids
                if draft_ids_list and draft_req_ids:
                    draft_by_req_id = dict(zip(draft_req_ids, draft_ids_list))
                    output_spec_token_ids = [draft_by_req_id.get(req_id, []) for req_id in req_ids_output_copy]

        routed_experts_lists = None
        if self.model_config.enable_return_routed_experts:
            capturer = self.routed_experts_capturer
            if capturer is not None and hasattr(self.input_batch, "num_tokens_no_spec"):
                routed_experts_lists = self._omni_extract_routed_experts(scheduler_output)

        #  -------------------------------------- Omni-new -------------------------------------------------
        scheduled_tokens = getattr(self, "_omni_num_scheduled_tokens_np", None)
        if scheduled_tokens is None:
            req_ids = self.input_batch.req_ids
            num_scheduled_tokens_np = np.array(
                [scheduler_output.num_scheduled_tokens[rid] for rid in req_ids],
                dtype=np.int32,
            )
        else:
            # The deferred builder must not observe the next step mutating this buffer.
            num_scheduled_tokens_np = np.asarray(scheduled_tokens, dtype=np.int32).copy()
        query_start_loc_cpu = self._snapshot_query_start_loc_cpu()

        # Async Omni output (PR #4476): move the per-request payload construction
        # off the AR decode critical path into a background builder, after
        # snapshotting hidden/mm tensors to CPU on a dedicated copy stream.
        # Disabled for the ascend profiling-chunk path, which needs the output
        # synchronously to record execution_time_ms, and when full-payload
        # accumulation would mutate request state from a background thread.
        profiling_need_timing = bool(
            self.ascend_config.scheduler_config.profiling_chunk_config.need_timing
            and hasattr(self, "_execution_start_time")
        )
        accumulate_needed = self._should_accumulate_full_payload_output()
        use_async_omni_output = (
            self._should_use_async_omni_output() and not profiling_need_timing and not accumulate_needed
        )

        # Snapshot mutable python state so the deferred builder cannot observe the
        # next decode step mutating shared runner structures (mirrors GPU AR runner).
        scheduler_output = self._snapshot_scheduler_output_for_async_omni_output(scheduler_output)
        req_ids_output_copy = list(req_ids_output_copy)
        req_id_to_index_output_copy = dict(req_id_to_index_output_copy)
        valid_sampled_token_ids = [list(token_ids) for token_ids in valid_sampled_token_ids]
        logprobs_lists = copy(logprobs_lists) if logprobs_lists is not None else None
        prompt_logprobs_dict = dict(prompt_logprobs_dict) if prompt_logprobs_dict is not None else {}

        # Models with a postprocess hook (e.g. talker) must run it on live device
        # tensors before the async D2H snapshot; the deferred builder then skips it.
        omni_postprocess_already_applied = False
        if use_async_omni_output:
            omni_postprocess_already_applied = self._maybe_run_eager_omni_postprocess_before_async_output(
                hidden_states=hidden_states,
                multimodal_outputs=multimodal_outputs,
                num_scheduled_tokens_np=num_scheduled_tokens_np,
                scheduler_output=scheduler_output,
                req_ids_output_copy=req_ids_output_copy,
                query_start_loc_cpu=query_start_loc_cpu,
            )

        output_tensor_snapshot = self._snapshot_omni_output_tensors_for_async_output(
            use_async_omni_output=use_async_omni_output,
            hidden_states=hidden_states,
            staged_hidden_states_cpu=None,
            multimodal_outputs=multimodal_outputs,
        )

        if self.dynamic_eplb:
            with record_function_or_nullcontext("EPLB update"):
                self.eplb_updator.forward_end(self.eplb_heat_collection_status)

        self._finalize_dump_data()

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
        # through the scheduler/engine IPC path. Skip when logits were
        # already broadcast (broadcast_pp_output), matching upstream vLLM.
        if self.use_async_scheduling:
            if not self.broadcast_pp_output and pp.world_size > 1 and pp.is_last_rank:
                self._pp_broadcast_prev_sampled_token_ids(sampler_output.sampled_token_ids)

        # Connector drain can HCCL-broadcast. Keep it on the decode thread.
        with record_function_or_nullcontext("omni_output_builder:get_omni_connector_output"):
            omni_connector_output = self.get_omni_connector_output()

        def output_builder() -> OmniModelRunnerOutput:
            if output_tensor_snapshot.async_payload is not None:
                with record_function_or_nullcontext("omni_async_output:wait_cpu_payload"):
                    output_tensor_snapshot.async_payload.wait()
            with record_function_or_nullcontext("omni_output_builder:total"):
                built_output = self._build_omni_model_runner_output_from_snapshot(
                    scheduler_output=scheduler_output,
                    hidden_states=output_tensor_snapshot.hidden_states,
                    staged_hidden_states_cpu=output_tensor_snapshot.staged_hidden_states_cpu,
                    multimodal_outputs=output_tensor_snapshot.multimodal_outputs,
                    req_ids_output_copy=req_ids_output_copy,
                    req_id_to_index_output_copy=req_id_to_index_output_copy,
                    valid_sampled_token_ids=valid_sampled_token_ids,
                    logprobs_lists=logprobs_lists,
                    prompt_logprobs_dict=prompt_logprobs_dict,
                    num_nans_in_logits=None,
                    kv_connector_output=kv_connector_output,
                    ec_connector_output=ec_connector_output,
                    cudagraph_stats=cudagraph_stats,
                    kv_extracted_req_ids=kv_extracted_req_ids,
                    num_scheduled_tokens_np=num_scheduled_tokens_np,
                    query_start_loc_cpu=query_start_loc_cpu,
                    postprocess_already_applied=omni_postprocess_already_applied,
                    omni_connector_output=omni_connector_output,
                    skip_accumulate_full_payload=use_async_omni_output,
                )
            if routed_experts_lists is not None:
                built_output.routed_experts = routed_experts_lists
            built_output.spec_token_ids = output_spec_token_ids
            return built_output

        if use_async_omni_output:
            async_output = OmniAsyncGPUModelRunnerOutput(
                model_runner_output_builder=output_builder,
                cuda_device=self.device,
                sampled_token_ids=sampler_output.sampled_token_ids,
                logprobs_tensors=sampler_output.logprobs_tensors,
                invalid_req_indices=invalid_req_indices,
                # Keep the sampled-token D2H on the runner's own copy stream.
                # The Omni payload stream already carries the (much larger)
                # hidden/mm snapshot, and the next decode step blocks on the
                # sampled-token event.
                async_output_copy_stream=self.async_output_copy_stream,
                vocab_size=self.input_batch.vocab_size,
            )
            self.input_batch.set_async_sampled_token_ids(
                async_output.sampled_token_ids_cpu,
                async_output.async_copy_ready_event,
            )
            return async_output

        model_runner_output = output_builder()
        #  -------------------------------------- Omni-new -------------------------------------------------

        if profiling_need_timing:
            self._sync_device()
            model_runner_output.execution_time_ms = (time.perf_counter() - self._execution_start_time) * 1000.0

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
