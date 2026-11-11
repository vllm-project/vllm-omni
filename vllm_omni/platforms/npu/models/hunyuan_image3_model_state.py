# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""HunyuanImage3 AR lifecycle, M-RoPE and image PrefixLM on Ascend MRv2."""

from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

import torch
import torch_npu
from vllm.config import CUDAGraphMode, VllmConfig, get_layers_from_vllm_config
from vllm.model_executor.layers.attention import Attention
from vllm.utils.torch_utils import get_kv_cache_torch_dtype
from vllm.v1.attention.backends.utils import compute_mm_prefix_ranges
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.interface import ModelSpecificAttnMetadata
from vllm.v1.worker.utils import AttentionGroup
from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl, AscendMetadata
from vllm_ascend.worker.v2.input_batch import AscendInputBatch
from vllm_ascend.worker.v2.model_states.default import AscendModelState

from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState


class _PrefixQuery(NamedTuple):
    req_index: int
    request_start: int
    query_start: int
    query_end: int
    kv_length: int


def _prefix_queries(
    query_start_loc: Sequence[int],
    seq_lens: Sequence[int],
    ranges: Mapping[int, Sequence[tuple[int, int]]],
) -> list[_PrefixQuery]:
    queries = []
    for req_index, intervals in ranges.items():
        request_start, request_end = query_start_loc[req_index : req_index + 2]
        seq_length = seq_lens[req_index]
        computed = seq_length - (request_end - request_start)
        for left, right in intervals:
            if left >= right:
                continue
            # Upstream ranges have inclusive ends. Cached image prefixes may
            # start before this chunk, but the whole image must be available.
            first, end = max(left, computed), min(right + 1, seq_length)
            if first < end:
                if right + 1 > seq_length:
                    raise ValueError(
                        "HunyuanImage3 NPU image prefix was split across prefill chunks. "
                        "Set disable_chunked_mm_input=True."
                    )
                queries.append(
                    _PrefixQuery(
                        req_index,
                        request_start,
                        request_start + first - computed,
                        request_start + end - computed,
                        end,
                    )
                )
    return queries


class NPUHunyuanImage3AttentionImpl(AscendAttentionBackendImpl):
    """Add image-prefix queries to the existing Ascend cache and attention path."""

    def forward_fused_infer_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = super().forward_fused_infer_attention(query, key, value, attn_metadata, output, kv_cache)
        prefix_queries: Sequence[_PrefixQuery] = getattr(attn_metadata, "_omni_prefix_queries", ())
        if not prefix_queries:
            return output
        keys, values, block_size, block_table, _ = self._get_fia_params(key, value, attn_metadata, kv_cache)
        for span in prefix_queries:
            selected_keys, selected_values = keys, values
            selected_table = None if block_table is None else block_table[span.req_index : span.req_index + 1]
            if block_table is None:
                selected_keys = keys[span.request_start : span.request_start + span.kv_length].contiguous()
                selected_values = values[span.request_start : span.request_start + span.kv_length].contiguous()
            # An image query sees every preceding token and its whole image.
            # Limit KV to the image end and use unmasked noncausal FIA. CANN's
            # paged path does not support an arbitrary dense PrefixLM mask.
            result, _ = torch_npu.npu_fused_infer_attention_score(
                query=query[span.query_start : span.query_end],
                key=selected_keys,
                value=selected_values,
                block_table=selected_table,
                input_layout="TND",
                block_size=block_size,
                actual_seq_lengths=[span.query_end - span.query_start],
                actual_seq_lengths_kv=[span.kv_length],
                num_key_value_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                scale=self.scale,
                sparse_mode=0,
                pre_tokens=2147483647,
                next_tokens=2147483647,
            )
            target = output[span.query_start : span.query_end]
            target.copy_(result.reshape_as(target))
        return output


class NPUHunyuanImage3ModelState(OmniModelState, AscendModelState):
    """Use Ascend attention metadata while preserving Omni inputs and sampling."""

    uses_ascend_attn_metadata = True

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: torch.nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        if not vllm_config.model_config.enforce_eager:
            raise ValueError("HunyuanImage3 NPU image-prefix attention requires enforce_eager=True")
        parallel = vllm_config.parallel_config
        if parallel.prefill_context_parallel_size != 1 or parallel.decode_context_parallel_size != 1:
            raise ValueError("HunyuanImage3 NPU image-prefix attention requires PCP/DCP=1")
        model_dtype = vllm_config.model_config.dtype
        if get_kv_cache_torch_dtype(vllm_config.cache_config.cache_dtype, model_dtype) != model_dtype:
            raise ValueError(
                "HunyuanImage3 NPU image-prefix attention requires unquantized KV matching the model dtype"
            )
        scheduler = vllm_config.scheduler_config
        if scheduler.enable_chunked_prefill and not scheduler.disable_chunked_mm_input:
            raise ValueError("HunyuanImage3 NPU chunked prefill requires disable_chunked_mm_input=True")

        layers = get_layers_from_vllm_config(vllm_config, Attention)
        for layer in layers.values():
            if type(layer.impl) not in (AscendAttentionBackendImpl, NPUHunyuanImage3AttentionImpl):
                raise ValueError("HunyuanImage3 NPU image-prefix attention requires the standard Ascend backend")
        super().__init__(vllm_config, model, encoder_cache, device)
        # Model-state initialization follows attention construction and weight
        # loading. Bind once here, preserving initialized Ascend backend state.
        for layer in layers.values():
            layer.impl.__class__ = NPUHunyuanImage3AttentionImpl

    def prepare_attn(
        self,
        input_batch: AscendInputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
        ubatch_idx: int = 0,
        model_specific_attn_metadata: ModelSpecificAttnMetadata | None = None,
    ) -> dict[str, Any]:
        metadata = super().prepare_attn(
            input_batch,
            cudagraph_mode,
            block_tables,
            slot_mappings,
            attn_groups,
            kv_cache_config,
            for_capture=for_capture,
            ubatch_idx=ubatch_idx,
            model_specific_attn_metadata=model_specific_attn_metadata,
        )
        queries = []
        if (
            metadata
            and not for_capture
            and not input_batch.is_dummy
            and input_batch.has_prefill
            and self.encoder_cache is not None
        ):
            ranges = compute_mm_prefix_ranges(
                [self.encoder_cache.mm_features.get(req_id, ()) for req_id in input_batch.req_ids],
                sliding_window=self.model_config.get_sliding_window(),
            )
            queries = _prefix_queries(
                input_batch.query_start_loc_np[: input_batch.num_reqs + 1].tolist(),
                input_batch.seq_lens_np[: input_batch.num_reqs].tolist(),
                ranges,
            )
        for layer_metadata in metadata.values():
            layer_metadata._omni_prefix_queries = queries
        return metadata
