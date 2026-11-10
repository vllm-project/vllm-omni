# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Benchmark dense vs Scheduler-paged Ring attention latency and memory.

Both paths use the same Q/K/V inputs, Ring topology, and Ring kernel. The paged
path additionally writes physical KV pages and materializes a contiguous Ring
shard. Timing reports worst-rank p50/p95. CUDA memory reports worst-rank
persistent paged state, forward peak deltas, and an estimated extra peak over
an equivalent dense-only process.

The default sweep uses global sequence lengths requested for the RFC evidence:
256, 1024, 4096, 16384, and 32768 tokens. Kernel compilation is excluded by
warmup, dense/paged order alternates to reduce thermal bias, and every sample
reports the worst rank because collectives progress at straggler speed. Latency
and peak memory are sampled in separate passes so CUDA peak-stat bookkeeping
cannot perturb the timed pass.

For four-rank pure Ring:

    python benchmarks/diffusion/bench_paged_ring_attention.py \
        --ulysses-degree 1 --ring-degree 4

For two-way Ulysses x two-way Ring:

    python benchmarks/diffusion/bench_paged_ring_attention.py \
        --ulysses-degree 2 --ring-degree 2

For a quick smoke run:

    python benchmarks/diffusion/bench_paged_ring_attention.py \
        --seqs 256,1024 --warmup 1 --iters 3
"""

from __future__ import annotations

import argparse
import gc
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
from vllm_omni.diffusion.attention.parallel.ring import RingParallelAttention
from vllm_omni.diffusion.attention.parallel.ulysses import UlyssesParallelAttention
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
_ULYSSES_DEGREE = 1
_RING_DEGREE = 4
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
    return adapter, diffusion_layer, native_layer


def _build_attention(
    diffusion_layer,
    strategy,
    *,
    paged: bool,
    ring_runner: RingParallelAttention | None = None,
) -> Attention:
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
    layer.use_ring = ring_runner is not None
    layer.ring_runner = ring_runner
    layer.softmax_scale = diffusion_layer.softmax_scale
    layer.causal = False
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


def _max_rank_values(values: tuple[float, ...], device: torch.device) -> list[float]:
    metrics = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(metrics, op=dist.ReduceOp.MAX)
    return metrics.tolist()


def _time_forward(fn, device: torch.device) -> float:
    """Return synchronized worst-rank latency without memory-stat operations."""
    dist.barrier(device_ids=[device.index])
    current_omni_platform.synchronize()

    start = time.perf_counter()
    output = fn()
    current_omni_platform.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    del output

    (elapsed_ms,) = _max_rank_values(
        (elapsed_ms,),
        device,
    )
    return elapsed_ms


def _memory_forward(fn, device: torch.device) -> tuple[int, int]:
    """Return worst-rank absolute CUDA allocated and reserved peaks."""
    dist.barrier(device_ids=[device.index])
    current_omni_platform.synchronize()
    torch.cuda.reset_peak_memory_stats(device)

    output = fn()
    current_omni_platform.synchronize()
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    del output

    peak_allocated, peak_reserved = _max_rank_values(
        (
            float(peak_allocated),
            float(peak_reserved),
        ),
        device,
    )
    return int(peak_allocated), int(peak_reserved)


def _worker(
    local_rank: int,
    world_size: int,
    init_method: str,
    seqs: list[int],
    warmup: int,
    iters: int,
    ulysses_degree: int,
    ring_degree: int,
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
        ring_degree=ring_degree,
        allgather_degree=1,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
    )

    try:
        sp_group = get_sp_group()
        ring_runner = RingParallelAttention(sp_group)
        if ulysses_degree > 1:
            strategy = UlyssesParallelAttention(
                sp_group,
                scatter_idx=2,
                gather_idx=1,
                use_sync=False,
            )
        else:
            strategy = ring_runner

        results: list[dict[str, int | float]] = []
        with set_forward_context():
            for seq_len in seqs:
                if seq_len % world_size:
                    raise ValueError(f"seq_len={seq_len} must be divisible by world_size={world_size}")

                gc.collect()
                torch.cuda.empty_cache()
                current_omni_platform.synchronize()
                state_allocated_before = torch.cuda.memory_allocated(device)
                state_reserved_before = torch.cuda.memory_reserved(device)

                adapter, diffusion_layer, native_layer = _build_real_adapter(
                    device,
                    seq_len,
                    ulysses_degree=ulysses_degree,
                )
                paged_attention = _build_attention(
                    diffusion_layer,
                    strategy,
                    paged=True,
                    ring_runner=ring_runner,
                )
                dense_attention = _build_attention(
                    diffusion_layer,
                    strategy,
                    paged=False,
                    ring_runner=ring_runner,
                )
                current_omni_platform.synchronize()
                state_allocated_after = torch.cuda.memory_allocated(device)
                state_reserved_after = torch.cuda.memory_reserved(device)
                paged_state_allocated, paged_state_reserved = _max_rank_values(
                    (
                        float(max(0, state_allocated_after - state_allocated_before)),
                        float(max(0, state_reserved_after - state_reserved_before)),
                    ),
                    device,
                )
                cache_bytes = native_layer.kv_cache.numel() * native_layer.kv_cache.element_size()
                (cache_bytes,) = _max_rank_values((float(cache_bytes),), device)

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
                    del dense_reference, paged_reference

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
                    dense_allocated_peaks: list[int] = []
                    paged_allocated_peaks: list[int] = []
                    dense_reserved_peaks: list[int] = []
                    paged_reserved_peaks: list[int] = []
                    for iteration in range(iters):
                        # Alternating order reduces monotonic clock/temperature
                        # bias without changing collective order across ranks.
                        if iteration % 2:
                            paged_ms = _time_forward(run_paged, device)
                            dense_ms = _time_forward(run_dense, device)
                        else:
                            dense_ms = _time_forward(run_dense, device)
                            paged_ms = _time_forward(run_paged, device)
                        dense_samples.append(dense_ms)
                        paged_samples.append(paged_ms)

                    # Peak-stat reset/query calls are intentionally outside the
                    # timing loop. They can perturb subsequent GPU work enough
                    # to create a false bimodal latency distribution.
                    for iteration in range(iters):
                        if iteration % 2:
                            paged_allocated, paged_reserved = _memory_forward(run_paged, device)
                            dense_allocated, dense_reserved = _memory_forward(run_dense, device)
                        else:
                            dense_allocated, dense_reserved = _memory_forward(run_dense, device)
                            paged_allocated, paged_reserved = _memory_forward(run_paged, device)
                        dense_allocated_peaks.append(dense_allocated)
                        paged_allocated_peaks.append(paged_allocated)
                        dense_reserved_peaks.append(dense_reserved)
                        paged_reserved_peaks.append(paged_reserved)

                dense_p50 = statistics.median(dense_samples)
                paged_p50 = statistics.median(paged_samples)
                element_size = torch.empty((), dtype=_DTYPE).element_size()
                kernel_query_tokens = seq_len // ring_degree
                local_heads = _NUM_HEADS // ulysses_degree
                staging_buffer_bytes = 2 * kernel_query_tokens * local_heads * _HEAD_SIZE * element_size
                dense_median_peak_allocated = max(
                    0,
                    int(statistics.median(dense_allocated_peaks)) - int(paged_state_allocated),
                )
                paged_median_peak_allocated = int(statistics.median(paged_allocated_peaks))
                dense_median_peak_reserved = max(
                    0,
                    int(statistics.median(dense_reserved_peaks)) - int(paged_state_reserved),
                )
                paged_median_peak_reserved = int(statistics.median(paged_reserved_peaks))
                result = {
                    "ulysses_degree": ulysses_degree,
                    "ring_degree": ring_degree,
                    "seq_len": seq_len,
                    "input_tokens_per_rank": local_tokens,
                    "kernel_query_tokens": kernel_query_tokens,
                    "staging_buffer_bytes": staging_buffer_bytes,
                    "kv_cache_bytes": int(cache_bytes),
                    "paged_state_allocated_bytes": int(paged_state_allocated),
                    "paged_state_reserved_bytes": int(paged_state_reserved),
                    "dense_median_peak_allocated_bytes": dense_median_peak_allocated,
                    "paged_median_peak_allocated_bytes": paged_median_peak_allocated,
                    "dense_median_peak_reserved_bytes": dense_median_peak_reserved,
                    "paged_median_peak_reserved_bytes": paged_median_peak_reserved,
                    "dense_p50_ms": dense_p50,
                    "dense_p95_ms": _percentile(dense_samples, 0.95),
                    "paged_p50_ms": paged_p50,
                    "paged_p95_ms": _percentile(paged_samples, 0.95),
                    "paged_over_dense": paged_p50 / dense_p50,
                    "paged_overhead_percent": (paged_p50 / dense_p50 - 1.0) * 100.0,
                }
                results.append(result)
                if local_rank == 0:
                    print(f"PAGED_RING_BENCH_JSON={json.dumps(result, sort_keys=True)}", flush=True)

                del (
                    adapter,
                    batch,
                    dense_attention,
                    diffusion_layer,
                    global_key,
                    global_query,
                    global_value,
                    key,
                    native_layer,
                    paged_attention,
                    qkv,
                    query,
                    run_dense,
                    run_paged,
                    value,
                )
                gc.collect()
                torch.cuda.empty_cache()
                current_omni_platform.synchronize()

        if local_rank == 0:
            strategy_name = (
                f"Ulysses({ulysses_degree}) x Ring({ring_degree})"
                if ulysses_degree > 1
                else f"Ring({ring_degree})"
            )
            print(f"\nDense vs paged {strategy_name} (worst-rank latency and CUDA memory)")
            print(
                "S       dense p50   paged p50   ratio   dense peak   paged peak   extra peak"
            )
            for row in results:
                extra_peak_mib = (
                    row["paged_median_peak_allocated_bytes"] - row["dense_median_peak_allocated_bytes"]
                ) / (1024**2)
                print(
                    f"{row['seq_len']:<7d} "
                    f"{row['dense_p50_ms']:>9.3f} ms "
                    f"{row['paged_p50_ms']:>9.3f} ms "
                    f"{row['paged_over_dense']:>6.3f}x "
                    f"{row['dense_median_peak_allocated_bytes'] / (1024**2):>10.2f}M "
                    f"{row['paged_median_peak_allocated_bytes'] / (1024**2):>10.2f}M "
                    f"{extra_peak_mib:>10.2f}M",
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
    parser.add_argument("--ring-degree", type=int, default=_RING_DEGREE)
    args = parser.parse_args()

    seqs = [int(value) for value in args.seqs.split(",")]
    if not seqs or any(value <= 0 for value in seqs):
        raise SystemExit("--seqs must contain positive integers")
    if args.warmup < 0 or args.iters <= 0:
        raise SystemExit("--warmup must be non-negative and --iters must be positive")
    if args.ulysses_degree <= 0:
        raise SystemExit("--ulysses-degree must be positive")
    if args.ring_degree <= 1:
        raise SystemExit("--ring-degree must be greater than one")
    if _NUM_HEADS % args.ulysses_degree:
        raise SystemExit(f"num_heads={_NUM_HEADS} must be divisible by ulysses_degree={args.ulysses_degree}")
    world_size = args.ulysses_degree * args.ring_degree
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
            args.ring_degree,
        ),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
