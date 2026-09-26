# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Compare dense/skip SM120 attention including vLLM-Omni's FP8 conversion.

Run from the repository root with an SM120 GPU and FlashInfer PR #4859:
    python benchmarks/kernels/benchmark_flashinfer_skip_softmax.py --output results.json

Synthetic inputs demonstrate overhead and possible speedup, not model quality.
"""

import argparse
import importlib.metadata
import json
import math
import statistics
from pathlib import Path

import flashinfer
import torch
import torch.nn.functional as F
from flashinfer.testing import bench_gpu_time

from vllm_omni.diffusion.attention.backends.flashinfer_attn import FlashInferAttentionImpl
from vllm_omni.diffusion.data import AttentionSpec


def error(actual, reference):
    delta = actual.float() - reference.float()
    return {
        "max_abs": delta.abs().max().item(),
        "relative_l2": (delta.norm() / reference.float().norm().clamp_min(1e-12)).item(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=int, nargs="+", default=[4096, 16384])
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--head-dim", type=int, choices=[64, 128, 256], default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.0, 1e-5, 1e-4, 1e-3, 1e-2])
    parser.add_argument("--patterns", choices=["random", "controlled"], nargs="+", default=["random", "controlled"])
    parser.add_argument("--skip-fraction", type=float, default=0.9)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        parser.error("requires an SM120 GPU")
    if min(*args.lengths, args.heads, args.batch_size, args.warmup, args.repeats) <= 0:
        parser.error("lengths, heads, batch size, warmup and repeats must be positive")
    if not 0 <= args.skip_fraction < 1:
        parser.error("skip-fraction must be in [0, 1)")
    if any(not math.isfinite(t) or t < 0 for t in args.thresholds):
        parser.error("thresholds must be finite and non-negative")

    def make_impl(threshold):
        spec = AttentionSpec(
            backend="FLASHINFER_ATTN",
            quant={"flashinfer_backend": "cute-dsl-prims", "dtype_qk": "fp8_e4m3", "dtype_vo": "fp8_e4m3"},
            skip_softmax={"threshold": threshold} if threshold is not None else None,
        )
        return FlashInferAttentionImpl(
            args.heads, args.head_dim, args.head_dim**-0.5, backend_kwargs=spec.backend_kwargs()
        )

    def timing(fn):
        # Graph timing includes the casts and kernel but amortizes Python launch
        # overhead. Warm up before capture to exclude JIT compilation and planning.
        fn()
        samples = bench_gpu_time(
            fn,
            dry_run_iters=args.warmup,
            repeat_iters=args.repeats,
            enable_cupti=False,
            use_cuda_graph=True,
            cold_l2_cache=False,
        )
        return statistics.median(samples)

    results = {
        "environment": {
            "gpu": torch.cuda.get_device_name(),
            "compute_capability": torch.cuda.get_device_capability(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "flashinfer": flashinfer.__version__,
            "flashinfer_path": flashinfer.__file__,
            "cutlass_dsl": importlib.metadata.version("nvidia-cutlass-dsl"),
            "timing": "CUDA events with CUDA Graph replay, warm L2, includes Q/K/V casts",
        },
        "arguments": {**vars(args), "output": str(args.output)},
        "cases": [],
    }
    gen = torch.Generator(device="cuda").manual_seed(42)
    with torch.inference_mode():
        for length in args.lengths:
            shape = (args.batch_size, length, args.heads, args.head_dim)
            for pattern in args.patterns:
                q, k, v = [torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=gen) for _ in range(3)]
                controlled_fraction = None
                if pattern == "controlled":
                    # The rightmost high-score tile is visited first. Low-score
                    # whole tiles can be skipped when threshold > exp(-2*sqrt(D)).
                    dropped = int(length // 128 * args.skip_fraction) * 128
                    q.fill_(1)
                    k.fill_(1)
                    k[:, :dropped] = -1
                    controlled_fraction = dropped / length
                dense_impl = make_impl(None)
                dense = dense_impl.forward_cuda(q, k, v)
                dense_ms = timing(lambda: dense_impl.forward_cuda(q, k, v))
                # Only a subset of query rows is needed for a bounded-memory
                # FP32 reference; every key is retained and no causal mask is used.
                q_ref = q[:, :128].float().transpose(1, 2)
                reference = F.scaled_dot_product_attention(
                    q_ref, k.float().transpose(1, 2), v.float().transpose(1, 2)
                ).transpose(1, 2)
                case = {
                    "pattern": pattern,
                    "shape": shape,
                    "controlled_low_score_fraction": controlled_fraction,
                    "dense_ms": dense_ms,
                    "dense_vs_fp32_first_128_queries": error(dense[:, :128], reference),
                    "thresholds": [],
                }
                for threshold in args.thresholds:
                    impl = make_impl(threshold)
                    out = impl.forward_cuda(q, k, v)
                    if not torch.isfinite(out).all():
                        raise RuntimeError(f"Nonfinite output at {pattern=}, {length=}, {threshold=}")
                    if threshold == 0:
                        torch.testing.assert_close(out, dense, rtol=0, atol=0)
                    ms = timing(lambda: impl.forward_cuda(q, k, v))
                    row = {
                        "threshold": threshold,
                        "latency_ms": ms,
                        "speedup": dense_ms / ms,
                        "vs_dense_fp8": error(out, dense),
                        "vs_fp32_first_128_queries": error(out[:, :128], reference),
                    }
                    case["thresholds"].append(row)
                    print(json.dumps({"pattern": pattern, "length": length, "dense_ms": dense_ms, **row}), flush=True)
                results["cases"].append(case)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
