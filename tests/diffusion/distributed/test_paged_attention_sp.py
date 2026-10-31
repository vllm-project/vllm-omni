# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real-kernel coverage for Scheduler-paged sequence-parallel attention."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from vllm.config import AttentionConfig, CacheConfig, VllmConfig, set_current_vllm_config
from vllm.utils.network_utils import get_file_store_init_method
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.worker import block_table as native_block_table
from vllm.v1.worker.gpu.attn_utils import init_attn_backend, init_kv_cache
from vllm.v1.worker.gpu.block_table import BlockTables

from tests.helpers.kv_layout import build_kv_cache_tensor
from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionImpl
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.attention.parallel.allgather_kv import (
    AllGatherKVParallelAttention,
)
from vllm_omni.diffusion.attention.parallel.ulysses_allgather import (
    UlyssesAllGatherKVParallelAttention,
)
from vllm_omni.diffusion.diffusion_kv.layout import resolve_diffusion_kv_cache_layout
from vllm_omni.diffusion.diffusion_kv.paged_attention_adapter import (
    DiffusionPagedAttentionAdapter,
    DiffusionPagedAttentionLayerAdapter,
    DiffusionPagedAttentionRow,
    DiffusionPagedAttentionRowBinding,
)
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    get_sp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.vllm_config import _DiffusionVllmModelConfig
from vllm_omni.platforms import current_omni_platform

_LAYER_NAME = "model.layers.0.attn"
_NUM_HEADS = 8
_HEAD_SIZE = 64
_BLOCK_SIZE = 16
_GLOBAL_TOKENS = 16


class _SmokeAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_heads = _NUM_HEADS
        self.num_kv_heads = _NUM_HEADS
        self.head_size = _HEAD_SIZE
        self.softmax_scale = _HEAD_SIZE**-0.5


def _build_real_adapter(device: torch.device, *, ulysses_degree: int):
    local_heads = _NUM_HEADS // ulysses_degree
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
        max_model_len=_GLOBAL_TOKENS,
        original_max_model_len=_GLOBAL_TOKENS,
        hf_config=SimpleNamespace(model_type="diffusion_paged_allgather_test"),
    )
    vllm_config.model_config = model_config
    model_config.set_attention_geometry(
        num_heads=local_heads,
        num_kv_heads=local_heads,
        head_size=_HEAD_SIZE,
    )
    vllm_config.compilation_config.static_forward_context.clear()

    diffusion_layer = _SmokeAttention().to(device)
    full_spec = FullAttentionSpec(
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
            spec=full_spec,
            vllm_config=vllm_config,
            device=device,
            ulysses_degree=ulysses_degree,
        )
    assert native_layer.num_heads == local_heads
    assert native_layer.num_kv_heads == local_heads
    vllm_config.compilation_config.static_forward_context[_LAYER_NAME] = native_layer

    canonical_spec = native_layer.spec
    kv_cache_config = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[build_kv_cache_tensor(canonical_spec, 4, [_LAYER_NAME])],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=[_LAYER_NAME],
                kv_cache_spec=canonical_spec,
            )
        ],
    )
    resolve_diffusion_kv_cache_layout(vllm_config)
    with set_current_vllm_config(vllm_config):
        attn_groups, _, kernel_block_sizes = init_attn_backend(
            kv_cache_config,
            vllm_config,
            device,
        )
        init_kv_cache(
            vllm_config.compilation_config.static_forward_context,
            kv_cache_config,
            device,
            kernel_block_sizes,
            vllm_config,
        )
        block_tables = BlockTables(
            block_sizes=[_BLOCK_SIZE],
            max_num_reqs=1,
            max_num_batched_tokens=_GLOBAL_TOKENS,
            max_num_blocks_per_group=[native_block_table.get_block_table_width(1, _BLOCK_SIZE)],
            device=device,
            kernel_block_sizes=kernel_block_sizes,
        )

    # Use a non-zero physical block to exercise logical-to-physical slots.
    block_tables.append_block_ids(0, ([3],), overwrite=True)
    block_tables.apply_staged_writes()
    adapter = DiffusionPagedAttentionAdapter(
        vllm_config=vllm_config,
        device=device,
        kv_cache_config=kv_cache_config,
        block_tables=block_tables,
        attn_groups=attn_groups,
        layers={_LAYER_NAME: native_layer},
        resolve_row=lambda *_args: DiffusionPagedAttentionRowBinding(
            row_index=0,
            max_seq_len=_GLOBAL_TOKENS,
        ),
    )
    return adapter, diffusion_layer, native_layer


def _build_omni_attention(diffusion_layer, strategy) -> Attention:
    layer = Attention.__new__(Attention)
    nn.Module.__init__(layer)
    layer._explicit_cross_contract = False
    layer._scheduler_paged_kv = True
    layer.prefix = _LAYER_NAME
    layer.paged_kv_cache_role = "primary"
    layer.attn_backend = SimpleNamespace(
        supports_paged_kv=True,
        get_name=lambda: "FLASH_ATTN",
    )
    layer.attention = FlashAttentionImpl(
        num_heads=_NUM_HEADS,
        head_size=_HEAD_SIZE,
        softmax_scale=diffusion_layer.softmax_scale,
        num_kv_heads=_NUM_HEADS,
    )
    layer.use_ring = False
    layer._no_parallel_strategy = object()
    layer._get_active_parallel_strategy = lambda: strategy
    layer._with_kv_cache_dtype = lambda metadata: metadata
    return layer


def _worker(
    local_rank: int,
    world_size: int,
    init_method: str,
    ulysses_degree: int,
    allgather_degree: int,
) -> None:
    device = torch.device(f"{current_omni_platform.device_type}:{local_rank}")
    current_omni_platform.set_device(device)
    init_distributed_environment(
        world_size=world_size,
        rank=local_rank,
        local_rank=local_rank,
        backend="nccl",
        distributed_init_method=init_method,
    )
    initialize_model_parallel(
        data_parallel_size=1,
        cfg_parallel_size=1,
        sequence_parallel_size=world_size,
        ulysses_degree=ulysses_degree,
        ring_degree=1,
        allgather_degree=allgather_degree,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
    )

    try:
        sp_group = get_sp_group()
        adapter, diffusion_layer, native_layer = _build_real_adapter(
            device,
            ulysses_degree=ulysses_degree,
        )
        if ulysses_degree > 1:
            strategy = UlyssesAllGatherKVParallelAttention(
                sp_group,
                scatter_idx=2,
                gather_idx=1,
                use_sync=False,
            )
        else:
            strategy = AllGatherKVParallelAttention(sp_group)
        attention = _build_omni_attention(diffusion_layer, strategy)
        batch = adapter.prepare_batch(
            [
                DiffusionPagedAttentionRow(
                    request_id="req-0",
                    sequence_id=0,
                    query_len=_GLOBAL_TOKENS,
                    seq_len=_GLOBAL_TOKENS,
                )
            ]
        )

        torch.manual_seed(20261025)
        qkv = torch.randn(
            3,
            1,
            _GLOBAL_TOKENS,
            _NUM_HEADS,
            _HEAD_SIZE,
            dtype=torch.float16,
            device=device,
        )
        global_query, global_key, global_value = qkv.unbind(0)
        local_tokens = _GLOBAL_TOKENS // world_size
        local_slice = slice(local_rank * local_tokens, (local_rank + 1) * local_tokens)
        query = global_query[:, local_slice].contiguous()
        key = global_key[:, local_slice].contiguous()
        value = global_value[:, local_slice].contiguous()

        global_expected = (
            F.scaled_dot_product_attention(
                global_query.transpose(1, 2),
                global_key.transpose(1, 2),
                global_value.transpose(1, 2),
                is_causal=False,
                scale=diffusion_layer.softmax_scale,
            )
            .transpose(1, 2)
            .contiguous()
        )
        expected = global_expected[:, local_slice]

        # None exercises layer.py's metadata construction for AllGather query ranges.
        with set_forward_context(), adapter.activate(batch):
            actual = attention._forward_impl(query, key, value, attn_metadata=None).clone()
        current_omni_platform.synchronize()

        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
        assert tuple(actual.shape) == (1, local_tokens, _NUM_HEADS, _HEAD_SIZE)
        assert batch.slot_mappings.tolist() == [list(range(3 * _BLOCK_SIZE, 3 * _BLOCK_SIZE + _GLOBAL_TOKENS))]

        local_heads = _NUM_HEADS // ulysses_degree
        head_start = sp_group.ulysses_rank * local_heads
        head_slice = slice(head_start, head_start + local_heads)
        expected_cache_key = global_key[:, :, head_slice].squeeze(0)
        expected_cache_value = global_value[:, :, head_slice].squeeze(0)
        key_cache, value_cache = native_layer.kv_cache.transpose(1, 2).split(_HEAD_SIZE, dim=-1)
        torch.testing.assert_close(key_cache[3, :_GLOBAL_TOKENS], expected_cache_key)
        torch.testing.assert_close(value_cache[3, :_GLOBAL_TOKENS], expected_cache_value)
        dist.barrier(device_ids=[local_rank])
    finally:
        destroy_distributed_env()


def _run_distributed_case(*, ulysses_degree: int, allgather_degree: int) -> None:
    world_size = ulysses_degree * allgather_degree
    if current_omni_platform.get_device_count() < world_size:
        pytest.skip(f"Requires {world_size} GPUs")

    torch.multiprocessing.spawn(
        _worker,
        args=(
            world_size,
            get_file_store_init_method(),
            ulysses_degree,
            allgather_degree,
        ),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.core_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": ["L4", "B200"]}, num_cards=2)
def test_paged_allgather_executes_native_kernel() -> None:
    _run_distributed_case(ulysses_degree=1, allgather_degree=2)


@pytest.mark.core_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": ["L4", "B200"]}, num_cards=4)
def test_paged_strict_ulysses_allgather_executes_native_kernel() -> None:
    _run_distributed_case(ulysses_degree=2, allgather_degree=2)
