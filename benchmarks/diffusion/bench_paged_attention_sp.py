# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Benchmark dense vs Scheduler-paged AllGather-KV SP strategies.

This is an end-to-end execution-path comparison, not a claim that paged
attention should be faster. Both cases execute the same configured collectives
and the same logical local-Q/global-KV attention. The ratio measures the cost
of native metadata construction, global KV cache writes, and paged reads.

The default sweep uses global sequence lengths requested for the RFC evidence:
256, 1024, 4096, 16384, and 32768 tokens. Kernel compilation is excluded by
warmup, dense/paged order alternates to reduce thermal bias, and every sample
reports the worst rank because collectives progress at straggler speed.

Run from the repository root on a machine with four GPUs:

    python benchmarks/diffusion/bench_paged_attention_sp.py

For plain four-rank AllGather-KV:

    python benchmarks/diffusion/bench_paged_attention_sp.py \
        --ulysses-degree 1 --allgather-degree 4

For a quick smoke run:

    python benchmarks/diffusion/bench_paged_attention_sp.py \
        --seqs 256,1024 --warmup 1 --iters 3
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from functools import partial
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch import nn
from vllm.config import AttentionConfig, CacheConfig, VllmConfig, set_current_vllm_config
from vllm.utils.network_utils import get_file_store_init_method
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.worker import block_table as native_block_table
from vllm.v1.worker.gpu.attn_utils import init_attn_backend, init_kv_cache
from vllm.v1.worker.gpu.block_table import BlockTables

from vllm_omni.diffusion.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
)
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.attention.parallel.allgather_kv import (
    AllGatherKVParallelAttention,
)
from vllm_omni.diffusion.attention.parallel.ulysses_allgather import (
    UlyssesAllGatherKVParallelAttention,
)
from vllm_omni.diffusion.diffusion_kv.layout import (
    build_kv_cache_tensor,
    resolve_diffusion_kv_cache_layout,
)
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
_ULYSSES_DEGREE = 2
_ALLGATHER_DEGREE = 2
_NUM_HEADS = 8
_HEAD_SIZE = 64
_BLOCK_SIZE = 16
_DTYPE = torch.float16


class _SmokeAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_heads = _NUM_HEADS
        self.num_kv_heads = _NUM_HEADS
        self.head_size = _HEAD_SIZE
        self.softmax_scale = _HEAD_SIZE**-0.5


def _build_real_adapter(device: torch.device, max_seq_len: int, *, ulysses_degree: int):
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
        dtype=_DTYPE,
        max_model_len=max_seq_len,
        original_max_model_len=max_seq_len,
        hf_config=SimpleNamespace(model_type="diffusion_paged_sp_benchmark"),
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
        dtype=_DTYPE,
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
    vllm_config.compilation_config.static_forward_context[_LAYER_NAME] = native_layer

    num_blocks = (max_seq_len + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    canonical_spec = native_layer.spec
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[build_kv_cache_tensor(canonical_spec, num_blocks, [_LAYER_NAME])],
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
            max_num_batched_tokens=max_seq_len,
            max_num_blocks_per_group=[native_block_table.get_block_table_width(max_seq_len, _BLOCK_SIZE)],
            device=device,
            kernel_block_sizes=kernel_block_sizes,
        )

    block_tables.append_block_ids(0, (tuple(range(num_blocks)),), overwrite=True)
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
            max_seq_len=max_seq_len,
        ),
    )
    return adapter, diffusion_layer


def _build_attention(diffusion_layer, strategy, *, paged: bool) -> Attention:
    layer = Attention.__new__(Attention)
    nn.Module.__init__(layer)
    layer._explicit_cross_contract = False
    layer._scheduler_paged_kv = paged
    layer.prefix = _LAYER_NAME
    layer.paged_kv_cache_role = "primary" if paged else None
    layer.attn_backend = FlashAttentionBackend
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
    layer._has_custom_attention = False
    layer.allow_fp32_fallback = False
    layer.backend_explicit = True
    layer.sdpa_fallback = None
    layer.attn_spec = None
    return layer


def _percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def _max_rank_ms(fn, device: torch.device) -> tuple[float, torch.Tensor]:
    dist.barrier(device_ids=[device.index])
    current_omni_platform.synchronize()
    start = time.perf_counter()
    output = fn()
    current_omni_platform.synchronize()
    elapsed = torch.tensor([(time.perf_counter() - start) * 1000.0], dtype=torch.float64, device=device)
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    return float(elapsed.item()), output


def _worker(
    local_rank: int,
    world_size: int,
    init_method: str,
    seqs: list[int],
    warmup: int,
    iters: int,
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
        adapter, diffusion_layer = _build_real_adapter(
            device,
            max(seqs),
            ulysses_degree=ulysses_degree,
        )
        if ulysses_degree > 1:
            strategy = UlyssesAllGatherKVParallelAttention(
                get_sp_group(),
                scatter_idx=2,
                gather_idx=1,
                use_sync=False,
            )
        else:
            strategy = AllGatherKVParallelAttention(get_sp_group())
        paged_attention = _build_attention(diffusion_layer, strategy, paged=True)
        dense_attention = _build_attention(diffusion_layer, strategy, paged=False)

        results: list[dict[str, int | float]] = []
        with set_forward_context():
            for seq_len in seqs:
                if seq_len % world_size:
                    raise ValueError(f"seq_len={seq_len} must be divisible by world_size={world_size}")
                batch = adapter.prepare_batch(
                    [
                        DiffusionPagedAttentionRow(
                            request_id="benchmark",
                            sequence_id=0,
                            query_len=seq_len,
                            seq_len=seq_len,
                        )
                    ]
                )
                torch.manual_seed(20261025 + seq_len)
                qkv = torch.randn(
                    3,
                    1,
                    seq_len,
                    _NUM_HEADS,
                    _HEAD_SIZE,
                    dtype=_DTYPE,
                    device=device,
                )
                global_query, global_key, global_value = qkv.unbind(0)
                local_tokens = seq_len // world_size
                local_slice = slice(local_rank * local_tokens, (local_rank + 1) * local_tokens)
                query = global_query[:, local_slice].contiguous()
                key = global_key[:, local_slice].contiguous()
                value = global_value[:, local_slice].contiguous()

                run_dense = partial(
                    dense_attention._forward_impl,
                    query,
                    key,
                    value,
                    attn_metadata=None,
                )
                run_paged = partial(
                    paged_attention._forward_impl,
                    query,
                    key,
                    value,
                    attn_metadata=None,
                )

                with adapter.activate(batch):
                    # Compile both paths and verify that the benchmark compares
                    # equivalent results before collecting any timing samples.
                    dense_reference = run_dense()
                    paged_reference = run_paged()
                    current_omni_platform.synchronize()
                    torch.testing.assert_close(
                        paged_reference,
                        dense_reference,
                        rtol=2e-2,
                        atol=2e-2,
                    )

                    for warmup_index in range(warmup):
                        if warmup_index % 2:
                            run_paged()
                            run_dense()
                        else:
                            run_dense()
                            run_paged()
                    current_omni_platform.synchronize()

                    dense_samples: list[float] = []
                    paged_samples: list[float] = []
                    for iteration in range(iters):
                        # Alternating order reduces monotonic clock/temperature
                        # bias without changing collective order across ranks.
                        if iteration % 2:
                            paged_ms, _ = _max_rank_ms(run_paged, device)
                            dense_ms, _ = _max_rank_ms(run_dense, device)
                        else:
                            dense_ms, _ = _max_rank_ms(run_dense, device)
                            paged_ms, _ = _max_rank_ms(run_paged, device)
                        dense_samples.append(dense_ms)
                        paged_samples.append(paged_ms)

                dense_p50 = statistics.median(dense_samples)
                paged_p50 = statistics.median(paged_samples)
                element_size = torch.empty((), dtype=_DTYPE).element_size()
                kernel_query_tokens = seq_len // allgather_degree
                local_heads = _NUM_HEADS // ulysses_degree
                local_kv_payload_bytes = 2 * kernel_query_tokens * local_heads * _HEAD_SIZE * element_size
                global_kv_write_bytes = 2 * seq_len * local_heads * _HEAD_SIZE * element_size
                result = {
                    "ulysses_degree": ulysses_degree,
                    "allgather_degree": allgather_degree,
                    "seq_len": seq_len,
                    "input_tokens_per_rank": local_tokens,
                    "kernel_query_tokens": kernel_query_tokens,
                    "kernel_kv_tokens": seq_len,
                    "local_kv_payload_bytes": local_kv_payload_bytes,
                    "global_kv_write_bytes": global_kv_write_bytes,
                    "dense_p50_ms": dense_p50,
                    "dense_p95_ms": _percentile(dense_samples, 0.95),
                    "paged_p50_ms": paged_p50,
                    "paged_p95_ms": _percentile(paged_samples, 0.95),
                    "paged_over_dense": paged_p50 / dense_p50,
                    "paged_overhead_percent": (paged_p50 / dense_p50 - 1.0) * 100.0,
                }
                results.append(result)
                if local_rank == 0:
                    print(f"PAGED_SP_BENCH_JSON={json.dumps(result, sort_keys=True)}", flush=True)

                del qkv, global_query, global_key, global_value, query, key, value, run_dense, run_paged
                current_omni_platform.synchronize()

        if local_rank == 0:
            strategy_name = (
                f"Ulysses({ulysses_degree}) x AllGather-KV({allgather_degree})"
                if ulysses_degree > 1
                else f"AllGather-KV({allgather_degree})"
            )
            print(f"\nDense vs paged {strategy_name} (worst-rank latency)")
            print("S       local-KV   global-write   dense p50   paged p50   paged/dense")
            for row in results:
                print(
                    f"{row['seq_len']:<7d} "
                    f"{row['local_kv_payload_bytes'] / (1024**2):>7.2f} MiB "
                    f"{row['global_kv_write_bytes'] / (1024**2):>10.2f} MiB "
                    f"{row['dense_p50_ms']:>9.3f} ms "
                    f"{row['paged_p50_ms']:>9.3f} ms "
                    f"{row['paged_over_dense']:>11.3f}x",
                    flush=True,
                )
    finally:
        destroy_distributed_env()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqs", default="256,1024,4096,16384,32768")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--ulysses-degree", type=int, default=_ULYSSES_DEGREE)
    parser.add_argument("--allgather-degree", type=int, default=_ALLGATHER_DEGREE)
    args = parser.parse_args()

    seqs = [int(value) for value in args.seqs.split(",")]
    if not seqs or any(value <= 0 for value in seqs):
        raise SystemExit("--seqs must contain positive integers")
    if args.warmup < 0 or args.iters <= 0:
        raise SystemExit("--warmup must be non-negative and --iters must be positive")
    if args.ulysses_degree <= 0 or args.allgather_degree <= 0:
        raise SystemExit("SP degrees must be positive")
    if _NUM_HEADS % args.ulysses_degree:
        raise SystemExit(f"num_heads={_NUM_HEADS} must be divisible by ulysses_degree={args.ulysses_degree}")
    world_size = args.ulysses_degree * args.allgather_degree
    if any(value % world_size for value in seqs):
        raise SystemExit(f"all sequence lengths must be divisible by world_size={world_size}")
    if current_omni_platform.get_device_count() < world_size:
        raise SystemExit(f"Need {world_size} GPUs")

    torch.multiprocessing.spawn(
        _worker,
        args=(
            world_size,
            get_file_store_init_method(),
            seqs,
            args.warmup,
            args.iters,
            args.ulysses_degree,
            args.allgather_degree,
        ),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
