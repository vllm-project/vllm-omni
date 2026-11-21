# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compare Wan VSA packing or full attention with identical inputs.

Run from the repository root with an idle CUDA GPU:
    python -m benchmarks.diffusion.benchmark_video_tiles --mode pack
    python -m benchmarks.diffusion.benchmark_video_tiles --mode attention
"""

import argparse
import json
import math
import statistics
from unittest.mock import patch

import torch
from vllm.triton_utils import triton

from vllm_omni.diffusion.attention.backends import fastvideo_vsa
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.ops.video_tiles import get_tile_metadata, pack_video_tiles


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["pack", "attention"], default="pack")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--shape", type=int, nargs=3, default=(31, 22, 40))
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=24)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=6)
    args = parser.parse_args()
    torch.manual_seed(0)
    seq_shape = tuple(args.shape)
    tile_size = (4, 8, 8)
    tensor_shape = (args.batch, math.prod(seq_shape), args.heads, args.head_dim)
    inputs = tuple(torch.randn(tensor_shape, device="cuda", dtype=getattr(torch, args.dtype)) for _ in range(4))
    partition, sizes, non_pad, _ = get_tile_metadata(seq_shape, tile_size, 256, inputs[0].device)
    output_shape = (args.batch, sizes.numel() * 256, args.heads, args.head_dim)

    def reference_pack(x, dit_seq_shape, block_size):
        output = torch.zeros(output_shape, dtype=x.dtype, device=x.device)
        output[:, non_pad] = x[:, partition]
        return output

    impl = fastvideo_vsa.FastVideoVSAImpl(
        num_heads=args.heads,
        head_size=args.head_dim,
        softmax_scale=args.head_dim**-0.5,
        backend_kwargs={"topk": args.topk, "fallback_on_error": False},
    )
    metadata = AttentionMetadata(
        extra={"vsa_dit_seq_shape": seq_shape, "gate_compress": inputs[3], "preserve_vsa_all_blocks": True}
    )
    if args.mode == "attention":
        reason = impl._fallback_reason(*inputs[:3], metadata)
        if reason is not None:
            raise ValueError(f"VSA benchmark would fall back to SDPA: {reason}")

    def run():
        if args.mode == "attention":
            return (impl.forward_cuda(*inputs[:3], metadata),)
        return tuple(fastvideo_vsa.pack_video_tiles(x, seq_shape, tile_size) for x in inputs)

    with patch.object(fastvideo_vsa, "pack_video_tiles", reference_pack):
        expected = run()
    with patch.object(fastvideo_vsa, "pack_video_tiles", pack_video_tiles):
        actual = run()
    for actual_tensor, expected_tensor in zip(actual, expected):
        assert torch.isfinite(actual_tensor).all() and torch.isfinite(expected_tensor).all()
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)
    del actual_tensor, expected_tensor
    del actual, expected
    samples: dict[str, list[float]] = {"baseline": [], "fused": []}
    peaks: dict[str, int] = {}
    for round_index in range(args.rounds):
        order = [("baseline", reference_pack), ("fused", pack_video_tiles)]
        if round_index % 2:
            order.reverse()
        for name, packing in order:
            with patch.object(fastvideo_vsa, "pack_video_tiles", packing):
                samples[name].append(triton.testing.do_bench(run, warmup=100, rep=300))
                torch.accelerator.synchronize()
                torch.accelerator.reset_peak_memory_stats()
                live_bytes = torch.accelerator.memory_allocated()
                run()
                torch.accelerator.synchronize()
                peaks[name] = max(peaks.get(name, 0), torch.accelerator.max_memory_allocated() - live_bytes)
    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "triton": triton.__version__,
                "cuda": torch.version.cuda,
                "config": vars(args),
                "samples_ms": samples,
                "median_ms": {name: statistics.median(values) for name, values in samples.items()},
                "range_ms": {name: [min(values), max(values)] for name, values in samples.items()},
                "peak_extra_bytes": peaks,
                "outputs_exact": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
