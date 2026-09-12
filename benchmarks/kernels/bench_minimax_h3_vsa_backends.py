# SPDX-License-Identifier: Apache-2.0
"""Compare FastVideo Triton and FlashInfer native blk64 VSA kernels.

This is a kernel-only benchmark. Sparse-index construction, BSHD/BHSD layout
conversion, and backend JIT compilation all happen before timing. The default
shape models one rank of MiniMax-H3 under Ulysses-8 for the 15-second 768p
workload; the synthetic sparse pattern keeps prefix queries dense and gives
each video query every prefix block plus 64 video blocks.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import math
import statistics
from collections.abc import Callable
from dataclasses import dataclass

import torch

BLOCK_SIZE = 64
HEAD_DIM = 128


@dataclass(frozen=True)
class Timing:
    p50_ms: float
    p90_ms: float
    minimum_ms: float
    maximum_ms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, default=109632)
    parser.add_argument("--num-heads", type=int, default=7)
    parser.add_argument("--prefix-blocks", type=int, default=8)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.seq_len <= 0 or args.seq_len % BLOCK_SIZE:
        raise ValueError("seq-len must be a positive multiple of 64")
    num_blocks = args.seq_len // BLOCK_SIZE
    if args.num_heads <= 0:
        raise ValueError("num-heads must be positive")
    if not 0 <= args.prefix_blocks < num_blocks:
        raise ValueError("prefix-blocks must be in [0, seq-len / 64)")
    if args.topk <= 0:
        raise ValueError("topk must be positive")
    if args.warmup <= 0 or args.repeat < 2:
        raise ValueError("warmup must be positive and repeat must be at least 2")


def build_h3_sparse_metadata(
    num_blocks: int,
    num_heads: int,
    prefix_blocks: int,
    topk: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Build deterministic prefix-dense plus video-top-k blk64 metadata."""
    video_blocks = num_blocks - prefix_blocks
    keep_video = min(topk, video_blocks)
    rows: list[list[int]] = []
    for query_block in range(num_blocks):
        if query_block < prefix_blocks:
            rows.append(list(range(num_blocks)))
            continue
        start = (query_block - prefix_blocks) % video_blocks
        selected = [prefix_blocks + (start + offset) % video_blocks for offset in range(keep_video)]
        rows.append([*range(prefix_blocks), *selected])

    max_nnz = max(len(row) for row in rows)
    q2k_idx = torch.full(
        (1, num_heads, num_blocks, max_nnz),
        -1,
        dtype=torch.int32,
        device=device,
    )
    q2k_num = torch.empty((1, num_heads, num_blocks), dtype=torch.int32, device=device)
    for query_block, row in enumerate(rows):
        q2k_idx[:, :, query_block, : len(row)] = torch.tensor(row, dtype=torch.int32, device=device)
        q2k_num[:, :, query_block] = len(row)

    block_sizes = torch.full((num_blocks,), BLOCK_SIZE, dtype=torch.int32, device=device)
    active_blocks_per_head = sum(len(row) for row in rows)
    return q2k_idx, q2k_num, block_sizes, active_blocks_per_head


def flashinfer_kernel() -> tuple[Callable[..., tuple[torch.Tensor, object]], str]:
    capability = torch.cuda.get_device_capability()
    if capability in ((10, 0), (10, 3)):
        from flashinfer.cute_dsl.sparse.bsa_attn_sm100_blk64 import (
            bsa_attn_sm100_blk64_fwd,
        )

        return bsa_attn_sm100_blk64_fwd, "sm100_blk64_cuda"
    if capability in ((12, 0), (12, 1)):
        from flashinfer.cute_dsl.sparse.bsa_attn_sm120 import (
            bsa_attn_sm120_blk64_fwd,
        )

        return bsa_attn_sm120_blk64_fwd, "sm120_blk64_cute_dsl"
    raise RuntimeError(f"FlashInfer blk64 VSA requires SM100/SM103 or SM120/SM121; current capability is {capability}")


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    return ordered[math.ceil(fraction * len(ordered)) - 1]


def summarize(samples: list[float]) -> Timing:
    return Timing(
        p50_ms=statistics.median(samples),
        p90_ms=percentile(samples, 0.9),
        minimum_ms=min(samples),
        maximum_ms=max(samples),
    )


def time_one(call: Callable[[], object]) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    call()
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


def package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "source-tree"


def main() -> None:
    args = parse_args()
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    device = torch.device("cuda")
    capability = torch.cuda.get_device_capability(device)
    num_blocks = args.seq_len // BLOCK_SIZE
    q2k_idx, q2k_num, block_sizes, active_blocks_per_head = build_h3_sparse_metadata(
        num_blocks,
        args.num_heads,
        args.prefix_blocks,
        args.topk,
        device,
    )

    torch.manual_seed(args.seed)
    shape = (1, args.seq_len, args.num_heads, HEAD_DIM)
    query = torch.randn(shape, dtype=torch.bfloat16, device=device)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    flashinfer_out = torch.empty_like(query)
    query_bhsd = query.transpose(1, 2).contiguous()
    key_bhsd = key.transpose(1, 2).contiguous()
    value_bhsd = value.transpose(1, 2).contiguous()

    from fastvideo_kernel.triton_kernels.block_sparse_attn_triton import (
        triton_block_sparse_attn_forward,
    )

    fi_kernel, fi_name = flashinfer_kernel()

    def run_triton() -> torch.Tensor:
        output, _ = triton_block_sparse_attn_forward(
            query_bhsd,
            key_bhsd,
            value_bhsd,
            q2k_idx,
            q2k_num,
            block_sizes,
        )
        return output

    def run_flashinfer() -> torch.Tensor:
        output, _ = fi_kernel(
            query,
            key,
            value,
            q2k_idx,
            q2k_idx.shape[-1],
            block_sizes=block_sizes,
            q2k_block_nums=q2k_num,
            softmax_scale=HEAD_DIM**-0.5,
            return_lse=False,
            out=flashinfer_out,
        )
        return output

    # Compile/autotune both providers and populate their allocator caches before
    # placing any CUDA events. Alternate the timed order to reduce clock drift.
    for _ in range(args.warmup):
        run_triton()
        run_flashinfer()
    torch.accelerator.synchronize()

    triton_samples: list[float] = []
    flashinfer_samples: list[float] = []
    for repetition in range(args.repeat):
        if repetition % 2:
            flashinfer_samples.append(time_one(run_flashinfer))
            triton_samples.append(time_one(run_triton))
        else:
            triton_samples.append(time_one(run_triton))
            flashinfer_samples.append(time_one(run_flashinfer))

    triton_output = run_triton().transpose(1, 2).contiguous()
    fi_output = run_flashinfer()
    torch.accelerator.synchronize()
    difference = fi_output.float() - triton_output.float()
    max_abs = float(difference.abs().max().item())
    relative_l2 = float((difference.norm() / triton_output.float().norm().clamp_min(1e-20)).item())
    if not torch.isfinite(fi_output).all():
        raise RuntimeError("FlashInfer output contains non-finite values")
    if max_abs > 0.04 or relative_l2 > 0.01:
        raise RuntimeError(f"backend mismatch: max_abs={max_abs:.8f}, relative_l2={relative_l2:.8f}")

    triton_timing = summarize(triton_samples)
    flashinfer_timing = summarize(flashinfer_samples)
    density = active_blocks_per_head / (num_blocks * num_blocks)
    active_blocks = active_blocks_per_head * args.num_heads
    attention_flops = 4 * active_blocks * BLOCK_SIZE**2 * HEAD_DIM

    print("MINIMAX_H3_VSA_BACKEND_BENCHMARK: PASS")
    print(
        f"gpu={torch.cuda.get_device_name(device)} capability={capability} "
        f"torch={torch.__version__} cuda={torch.version.cuda}"
    )
    print(
        f"fastvideo_kernel={package_version('fastvideo-kernel')} "
        f"source={inspect.getsourcefile(triton_block_sparse_attn_forward)}"
    )
    print(
        f"flashinfer={package_version('flashinfer-python')} backend={fi_name} source={inspect.getsourcefile(fi_kernel)}"
    )
    print(
        f"batch=1 seq_len={args.seq_len} heads={args.num_heads} "
        f"head_dim={HEAD_DIM} logical_block={BLOCK_SIZE} blocks={num_blocks} "
        f"prefix_blocks={args.prefix_blocks} topk={args.topk} "
        f"max_nnz={q2k_idx.shape[-1]} density={density:.6f}"
    )
    print(
        "fastvideo_triton "
        f"p50_ms={triton_timing.p50_ms:.6f} "
        f"p90_ms={triton_timing.p90_ms:.6f} "
        f"min_ms={triton_timing.minimum_ms:.6f} "
        f"max_ms={triton_timing.maximum_ms:.6f} "
        f"effective_tflops={attention_flops / triton_timing.p50_ms / 1e9:.3f}"
    )
    print(
        f"flashinfer_{fi_name} "
        f"p50_ms={flashinfer_timing.p50_ms:.6f} "
        f"p90_ms={flashinfer_timing.p90_ms:.6f} "
        f"min_ms={flashinfer_timing.minimum_ms:.6f} "
        f"max_ms={flashinfer_timing.maximum_ms:.6f} "
        f"effective_tflops={attention_flops / flashinfer_timing.p50_ms / 1e9:.3f}"
    )
    print(
        f"speedup={triton_timing.p50_ms / flashinfer_timing.p50_ms:.4f}x "
        f"max_abs={max_abs:.8f} relative_l2={relative_l2:.8f}"
    )


if __name__ == "__main__":
    main()
