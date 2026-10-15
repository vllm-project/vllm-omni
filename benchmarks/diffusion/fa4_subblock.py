# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Benchmark FA4 SubBlock attention calls on synthetic or captured BSHD inputs."""

import argparse
import json
from pathlib import Path

import torch

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionBackend, FlashAttentionImpl
from vllm_omni.diffusion.attention.block_sparse import BlockSparseAttention
from vllm_omni.diffusion.data import BlockSparseAttentionSpec


def latency(fn, iterations):
    for _ in range(5):
        fn()
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", help="Torch file with q/k/v and optional meta.scale/meta.prefix_len")
    parser.add_argument("--q-length", type=int, default=1024)
    parser.add_argument("--kv-length", type=int, default=1089)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--block-size", type=int, nargs=2, default=[64, 64], metavar=("BQ", "BKV"))
    parser.add_argument("--sparsity", type=float, default=0.75, help="Target fraction of unprotected blocks to drop")
    parser.add_argument("--output", type=Path, help="Write the measurement as JSON")
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    torch.manual_seed(17)
    scale, prefix = 128**-0.5, 0
    if args.capture:
        capture = torch.load(args.capture, map_location="cpu", weights_only=True)
        q, k, v = (capture[name].cuda() for name in ("q", "k", "v"))
        scale = capture.get("meta", {}).get("scale", q.shape[-1] ** -0.5)
        prefix = capture.get("meta", {}).get("prefix_len", 0)
    else:
        q = torch.randn(1, args.q_length, args.heads, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, args.kv_length, args.kv_heads, 128, device=q.device, dtype=q.dtype)
        v = torch.randn_like(k)
    spec = BlockSparseAttentionSpec(
        name="block_sparse",
        config={
            "block_size": args.block_size,
            "selection": {"name": "block_topk", "config": {"target_sparsity": args.sparsity}},
            "backend": {"require": "FLASH_ATTN", "implementation": "auto"},
        },
    )
    sparse = BlockSparseAttention(
        q.shape[2],
        k.shape[2],
        q.shape[-1],
        scale,
        False,
        None,
        spec,
        adapter=FlashAttentionBackend.get_block_sparse_adapter()(),
    )
    dense = FlashAttentionImpl(q.shape[2], q.shape[-1], scale, num_kv_heads=k.shape[2])
    metadata = AttentionMetadata(extra={"protected_kv_prefix": prefix})
    sparse.prepare_request(q, k, v, metadata)
    compiled_dense = torch.compile(dense.forward_cuda, fullgraph=True)
    compiled = torch.compile(sparse.forward_cuda, fullgraph=True)
    torch.testing.assert_close(compiled(q, k, v, metadata), sparse.forward_cuda(q, k, v, metadata), atol=0, rtol=0)
    torch.testing.assert_close(compiled_dense(q, k, v), dense.forward_cuda(q, k, v), atol=0, rtol=0)
    blocks = (k.shape[1] + args.block_size[1] - 1) // args.block_size[1]
    prefix_blocks = (prefix + args.block_size[1] - 1) // args.block_size[1]
    # Report the actual selected pattern, rather than duplicating budget arithmetic.
    selection = sparse.selector.select(q, k, scale, prefix)
    retained_blocks = torch.unique(selection.counts).item()
    assert retained_blocks >= prefix_blocks
    torch.testing.assert_close(
        selection.indices[..., :prefix_blocks],
        torch.arange(prefix_blocks, device=q.device, dtype=torch.int32).expand(
            *selection.indices.shape[:-1], prefix_blocks
        ),
    )
    unprotected_blocks = blocks - prefix_blocks
    from vllm_omni.diffusion.attention.backends.utils import fa

    report = json.dumps(
        {
            "benchmark_scope": "attention_call",
            "model_role_activation_verified": False,
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "q_shape": list(q.shape),
            "kv_shape": list(k.shape),
            "protected_prefix": prefix,
            "fa4_version": sparse.adapter.dependency_version,
            "dense_kernel": getattr(fa.flash_attn_func, "__module__", "varlen fallback"),
            "config": spec.config,
            "protected_blocks": prefix_blocks,
            "target_sparsity_scope": "unprotected_blocks",
            "unprotected_blocks": unprotected_blocks,
            "retained_unprotected_blocks": retained_blocks - prefix_blocks,
            "retained_blocks": retained_blocks,
            "overall_block_sparsity": 1 - retained_blocks / blocks,
            "unprotected_block_sparsity": (
                1 - (retained_blocks - prefix_blocks) / unprotected_blocks if unprotected_blocks else None
            ),
            "iterations": args.iterations,
            "warmup_iterations": 5,
            "eager_compiled_outputs_equal": True,
            "total_blocks": blocks,
            "dense_eager_ms": latency(lambda: dense.forward_cuda(q, k, v), args.iterations),
            "dense_compiled_ms": latency(lambda: compiled_dense(q, k, v), args.iterations),
            "sparse_eager_ms": latency(lambda: sparse.forward_cuda(q, k, v, metadata), args.iterations),
            "sparse_compiled_ms": latency(lambda: compiled(q, k, v, metadata), args.iterations),
        },
        indent=2,
    )
    print(report)
    if args.output:
        args.output.write_text(report + "\n")


if __name__ == "__main__":
    main()
