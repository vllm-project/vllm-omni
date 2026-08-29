# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from vllm.config import AttentionConfig, CacheConfig, VllmConfig, set_current_vllm_config
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor
from vllm.v1.worker.gpu.attn_utils import init_attn_backend, init_kv_cache
from vllm.v1.worker.gpu.block_table import BlockTables

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionImpl
from vllm_omni.diffusion.diffusion_kv.paged_attention_adapter import (
    DiffusionPagedAttentionAdapter,
    DiffusionPagedAttentionLayerAdapter,
    DiffusionPagedAttentionRow,
    DiffusionPagedAttentionRowBinding,
)
from vllm_omni.diffusion.vllm_config import _DiffusionVllmModelConfig

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.gpu]

_LAYER_NAME = "model.layers.0.attn"
_NUM_HEADS = 2
_HEAD_SIZE = 64
_BLOCK_SIZE = 16


class _SmokeDiffusionAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_heads = _NUM_HEADS
        self.num_kv_heads = _NUM_HEADS
        self.head_size = _HEAD_SIZE
        self.softmax_scale = _HEAD_SIZE**-0.5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for native paged attention")
def test_adapter_executes_native_paged_attention_on_non_contiguous_blocks() -> None:
    device = torch.device("cuda", torch.accelerator.current_device_index())
    vllm_config = VllmConfig(
        cache_config=CacheConfig(
            block_size=_BLOCK_SIZE,
            cache_dtype="float16",
            enable_prefix_caching=False,
        ),
        attention_config=AttentionConfig(use_non_causal=True),
    )
    model_config = _DiffusionVllmModelConfig(
        model=None,
        dtype=torch.float16,
        max_model_len=32,
        original_max_model_len=32,
        hf_config=SimpleNamespace(model_type="diffusion_paged_attention_smoke"),
    )
    vllm_config.model_config = model_config
    model_config.set_attention_geometry(
        num_heads=_NUM_HEADS,
        num_kv_heads=_NUM_HEADS,
        head_size=_HEAD_SIZE,
    )
    vllm_config.compilation_config.static_forward_context.clear()

    diffusion_layer = _SmokeDiffusionAttention().to(device)
    spec = FullAttentionSpec(
        block_size=_BLOCK_SIZE,
        num_kv_heads=_NUM_HEADS,
        head_size=_HEAD_SIZE,
        dtype=torch.float16,
        non_causal=True,
    )
    with set_current_vllm_config(vllm_config):
        native_layer = DiffusionPagedAttentionLayerAdapter(
            layer_name=_LAYER_NAME,
            layer=diffusion_layer,
            spec=spec,
            vllm_config=vllm_config,
            device=device,
        )
    vllm_config.compilation_config.static_forward_context[_LAYER_NAME] = native_layer

    canonical_spec = native_layer.spec
    kv_cache_config = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[
            KVCacheTensor(
                size=canonical_spec.page_size_bytes * 4,
                shared_by=[_LAYER_NAME],
            )
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=[_LAYER_NAME],
                kv_cache_spec=canonical_spec,
            )
        ],
    )
    with set_current_vllm_config(vllm_config):
        attn_groups, _, kernel_block_sizes = init_attn_backend(
            kv_cache_config,
            vllm_config,
            device,
        )
        runner_kv_caches: list[torch.Tensor | list[torch.Tensor]] = []
        init_kv_cache(
            runner_kv_caches,
            vllm_config.compilation_config.static_forward_context,
            kv_cache_config,
            attn_groups,
            device,
            vllm_config.cache_config.cache_dtype,
            kernel_block_sizes,
            vllm_config,
        )
        block_tables = BlockTables(
            block_sizes=[_BLOCK_SIZE],
            max_num_reqs=1,
            max_num_batched_tokens=32,
            max_num_blocks_per_group=[2],
            device=device,
            kernel_block_sizes=kernel_block_sizes,
        )

    block_tables.append_block_ids(0, ([3, 1],), overwrite=True)
    block_tables.apply_staged_writes()
    adapter = DiffusionPagedAttentionAdapter(
        vllm_config=vllm_config,
        device=device,
        kv_cache_config=kv_cache_config,
        block_tables=block_tables,
        attn_groups=attn_groups,
        layers={_LAYER_NAME: native_layer},
        resolve_row=lambda _request_id, _sequence_id, _context_id: DiffusionPagedAttentionRowBinding(
            row_index=0,
            max_seq_len=19,
        ),
    )
    prefix_batch = adapter.prepare_batch(
        [
            DiffusionPagedAttentionRow(
                request_id="req-0",
                sequence_id=0,
                query_len=17,
                seq_len=17,
            )
        ]
    )

    torch.manual_seed(1)
    query = torch.randn(19, _NUM_HEADS, _HEAD_SIZE, dtype=torch.float16, device=device)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    omni_backend = FlashAttentionImpl(
        num_heads=_NUM_HEADS,
        head_size=_HEAD_SIZE,
        softmax_scale=diffusion_layer.softmax_scale,
        num_kv_heads=_NUM_HEADS,
    )
    with adapter.activate(prefix_batch):
        prefix_context = adapter.prepare_layer_context(_LAYER_NAME, query[:17], key[:17], value[:17])
        prefix_output = omni_backend.forward_paged(prefix_context)
    prefix_slot_mappings = prefix_batch.slot_mappings.clone()

    suffix_batch = adapter.prepare_batch(
        [
            DiffusionPagedAttentionRow(
                request_id="req-0",
                sequence_id=0,
                query_len=2,
                seq_len=19,
                kv_start_pos=17,
            )
        ]
    )
    with adapter.activate(suffix_batch):
        suffix_context = adapter.prepare_layer_context(_LAYER_NAME, query[17:], key[17:], value[17:])
        suffix_output = omni_backend.forward_paged(suffix_context)
    suffix_slot_mappings = suffix_batch.slot_mappings.clone()

    piecewise_batch = adapter.prepare_batch(
        [
            DiffusionPagedAttentionRow(
                request_id="req-0",
                sequence_id=0,
                query_len=19,
                seq_len=19,
            )
        ]
    )
    mixed_mask = torch.ones(19, 19, dtype=torch.bool, device=device).tril()
    mixed_mask[5:10, 5:10] = True
    mixed_metadata = AttentionMetadata(
        attn_mask=mixed_mask.unsqueeze(0).unsqueeze(0),
        full_attn_spans=[[(5, 10)]],
    )
    with adapter.activate(piecewise_batch):
        piecewise_context = adapter.prepare_layer_context(
            _LAYER_NAME,
            query,
            key,
            value,
            omni_attn_metadata=mixed_metadata,
        )
        piecewise_output = omni_backend.forward_paged(piecewise_context)

    prefix_reference = (
        F.scaled_dot_product_attention(
            query[:17].transpose(0, 1).unsqueeze(0),
            key[:17].transpose(0, 1).unsqueeze(0),
            value[:17].transpose(0, 1).unsqueeze(0),
            is_causal=False,
            scale=diffusion_layer.softmax_scale,
        )
        .squeeze(0)
        .transpose(0, 1)
    )
    suffix_reference = (
        F.scaled_dot_product_attention(
            query[17:].transpose(0, 1).unsqueeze(0),
            key.transpose(0, 1).unsqueeze(0),
            value.transpose(0, 1).unsqueeze(0),
            is_causal=False,
            scale=diffusion_layer.softmax_scale,
        )
        .squeeze(0)
        .transpose(0, 1)
    )
    piecewise_reference = (
        F.scaled_dot_product_attention(
            query.transpose(0, 1).unsqueeze(0),
            key.transpose(0, 1).unsqueeze(0),
            value.transpose(0, 1).unsqueeze(0),
            attn_mask=mixed_mask.unsqueeze(0).unsqueeze(0),
            scale=diffusion_layer.softmax_scale,
        )
        .squeeze(0)
        .transpose(0, 1)
    )
    torch.accelerator.synchronize(device)

    assert prefix_slot_mappings.cpu().tolist() == [list(range(3 * _BLOCK_SIZE, 4 * _BLOCK_SIZE)) + [_BLOCK_SIZE]]
    assert suffix_slot_mappings.cpu().tolist() == [[_BLOCK_SIZE + 1, _BLOCK_SIZE + 2]]
    torch.testing.assert_close(prefix_output, prefix_reference, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(suffix_output, suffix_reference, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(piecewise_output, piecewise_reference, rtol=2e-2, atol=2e-2)
