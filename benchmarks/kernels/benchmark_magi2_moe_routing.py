# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Benchmark MAGI-2 multi-head MoE route construction.

Measures the two route-construction stages of
``vllm_omni.diffusion.models.magi2.mh_moe`` in isolation:

* ``compute_topk_probs_and_indices`` -- sigmoid, selection bias, top-k,
  probability gather and L1 normalization;
* ``global_sort_routes`` -- stable flattened-expert sort, counts and CSR
  offsets.

Both are reported against their unfused reference implementations, which are
also used as automatic fallbacks for unsupported inputs.  This is a synthetic
kernel microbenchmark: it builds random router logits and does not load a
checkpoint.  ``--with-experts`` additionally times the fused expert kernel so
the routing share of a whole MoE layer can be read off directly, at the cost of
about 6 GiB of expert weights.

Example:
    python benchmarks/kernels/benchmark_magi2_moe_routing.py \
        --tokens 1024,7674,29184,58368 --warmup 20 --iterations 50
"""

import argparse
import math
import statistics
from collections.abc import Callable

import torch
from vllm.triton_utils import triton

from vllm_omni.diffusion.models.magi2.mh_moe import (
    _reference_global_sort_routes,
    _reference_topk_probs_and_indices,
    compute_topk_probs_and_indices,
    global_sort_routes,
    triton_mh_moe_forward,
)

MiB = 1024**2

# sand-ai/MAGI-2-preview: 12 hidden-state heads, each routing to its own bank of
# 256 experts, top-6.  540p is 56x32 latent patches over 16 latent frames, so a
# full-trajectory denoise step sees roughly 29k video tokens per forward.
DEFAULT_HEADS = 12
DEFAULT_EXPERTS = 256
DEFAULT_TOP_K = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokens", default="1024,4096,7674,16384,29184,58368")
    parser.add_argument("--heads", type=int, default=DEFAULT_HEADS)
    parser.add_argument("--experts", type=int, default=DEFAULT_EXPERTS)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--head-dim", type=int, default=256, help="Only used by --with-experts.")
    parser.add_argument("--expert-dim", type=int, default=1280, help="Only used by --with-experts.")
    parser.add_argument("--no-bias", action="store_true", help="Route without the auxiliary-free selection bias.")
    parser.add_argument("--no-route-norm", action="store_true")
    parser.add_argument("--with-experts", action="store_true", help="Also time the fused expert kernel.")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def percentile(samples: list[float], probability: float) -> float:
    ordered = sorted(samples)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def measure_latency_us(operation: Callable[[], object], warmup: int, iterations: int) -> list[float]:
    for _ in range(warmup):
        operation()
    torch.accelerator.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        operation()
        end.record()
    torch.accelerator.synchronize()
    return [start.elapsed_time(end) * 1000.0 for start, end in zip(starts, ends, strict=True)]


def measure_extra_peak_mib(operation: Callable[[], object]) -> float:
    torch.accelerator.synchronize()
    torch.accelerator.reset_peak_memory_stats()
    allocated = torch.accelerator.memory_allocated()
    operation()
    torch.accelerator.synchronize()
    return (torch.accelerator.max_memory_allocated() - allocated) / MiB


def print_header(args: argparse.Namespace) -> None:
    print(f"device={torch.cuda.get_device_name()} torch={torch.__version__}")
    print(
        f"heads={args.heads} experts={args.experts} top_k={args.top_k} "
        f"bias={not args.no_bias} route_norm={not args.no_route_norm} "
        f"warmup={args.warmup} iterations={args.iterations} seed={args.seed}"
    )
    print(
        "tokens   ref_topk  new_topk  ref_sort  new_sort   ref_tot   new_tot  speedup"
        "  ref_MiB  new_MiB   ties  max_dprob"
    )


def benchmark_shape(tokens: int, args: argparse.Namespace) -> None:
    device = torch.device("cuda")
    heads, experts, top_k = args.heads, args.experts, args.top_k
    logits = torch.randn(heads, tokens, experts, device=device, dtype=torch.float32)
    bias = None if args.no_bias else torch.randn(heads, experts, device=device, dtype=torch.float32) * 0.01
    route_norm = not args.no_route_norm

    def reference_route() -> tuple[torch.Tensor, torch.Tensor]:
        return _reference_topk_probs_and_indices(logits, top_k, expert_bias=bias, route_norm=route_norm)

    def fused_route() -> tuple[torch.Tensor, torch.Tensor]:
        return compute_topk_probs_and_indices(logits, top_k, expert_bias=bias, route_norm=route_norm)

    reference_probs, reference_indices = reference_route()
    fused_probs, fused_indices = fused_route()
    # Rows whose biased selection scores tie are the only rows allowed to
    # disagree: torch.topk does not define the order of an exact tie.
    agreeing = (reference_indices == fused_indices).all(dim=-1)
    tied_rows = int((~agreeing).sum().item())
    max_dprob = (reference_probs - fused_probs)[agreeing].abs().max().item() if bool(agreeing.any()) else 0.0

    reference_layout = _reference_global_sort_routes(fused_probs, fused_indices, experts)
    fused_layout = global_sort_routes(fused_probs, fused_indices, experts)
    for expected, actual, name in zip(reference_layout, fused_layout, ("gather_ids", "probs", "offsets"), strict=True):
        if not torch.equal(expected, actual):
            raise AssertionError(f"global_sort_routes diverged from the reference layout in {name}")

    reference_peak = measure_extra_peak_mib(lambda: _reference_global_sort_routes(*reference_route(), experts))
    fused_peak = measure_extra_peak_mib(lambda: global_sort_routes(*fused_route(), experts))

    samples = {
        "ref_topk": measure_latency_us(reference_route, args.warmup, args.iterations),
        "new_topk": measure_latency_us(fused_route, args.warmup, args.iterations),
        "ref_sort": measure_latency_us(
            lambda: _reference_global_sort_routes(fused_probs, fused_indices, experts), args.warmup, args.iterations
        ),
        "new_sort": measure_latency_us(
            lambda: global_sort_routes(fused_probs, fused_indices, experts), args.warmup, args.iterations
        ),
    }
    median = {name: statistics.median(values) for name, values in samples.items()}
    reference_total = median["ref_topk"] + median["ref_sort"]
    fused_total = median["new_topk"] + median["new_sort"]
    print(
        f"{tokens:<7} {median['ref_topk']:>9.1f} {median['new_topk']:>9.1f} "
        f"{median['ref_sort']:>9.1f} {median['new_sort']:>9.1f} "
        f"{reference_total:>9.1f} {fused_total:>9.1f} {reference_total / fused_total:>7.2f}x "
        f"{reference_peak:>8.1f} {fused_peak:>8.1f} {tied_rows:>6} {max_dprob:>10.2e}"
    )
    for name in ("ref_topk", "new_topk", "ref_sort", "new_sort"):
        values = samples[name]
        p10, p90 = percentile(values, 0.1), percentile(values, 0.9)
        print(f"        {name}: p10={p10:.1f} p50={median[name]:.1f} p90={p90:.1f} us")

    if args.with_experts:
        benchmark_expert_share(tokens, args, logits, bias, fused_total, reference_total)


def benchmark_expert_share(
    tokens: int,
    args: argparse.Namespace,
    logits: torch.Tensor,
    bias: torch.Tensor | None,
    fused_total: float,
    reference_total: float,
) -> None:
    device = torch.device("cuda")
    heads, experts = args.heads, args.experts
    flat_experts = heads * experts
    hidden = torch.randn(tokens, heads, args.head_dim, device=device, dtype=torch.bfloat16)
    w_gate = torch.randn(flat_experts, args.head_dim, args.expert_dim, device=device, dtype=torch.bfloat16) * 0.02
    w_up = torch.randn(flat_experts, args.head_dim, args.expert_dim, device=device, dtype=torch.bfloat16) * 0.02
    w_down = torch.randn(flat_experts, args.expert_dim, args.head_dim, device=device, dtype=torch.bfloat16) * 0.02
    probs, indices = compute_topk_probs_and_indices(logits, args.top_k, expert_bias=bias)
    gather_ids, sorted_probs, offsets = global_sort_routes(probs, indices, experts)

    def run_experts() -> torch.Tensor:
        return triton_mh_moe_forward(hidden, gather_ids, sorted_probs, offsets, w_gate, w_up, w_down)

    try:
        expert_us = statistics.median(measure_latency_us(run_experts, args.warmup, max(args.iterations // 5, 5)))
    except triton.runtime.errors.OutOfResources as error:
        # The expert kernel picks its tiling from the device capability; a tile
        # that does not fit is a property of that kernel, not of route
        # construction, so keep the routing numbers and report the shortfall.
        print(f"        experts: skipped, {error.__class__.__name__}: {str(error).splitlines()[0]}")
        torch.accelerator.empty_cache()
        return
    print(
        f"        experts: {expert_us:.1f} us -> routing share "
        f"{reference_total / (reference_total + expert_us) * 100:.1f}% (reference) "
        f"-> {fused_total / (fused_total + expert_us) * 100:.1f}% (fused); "
        f"MoE-layer speedup {(reference_total + expert_us) / (fused_total + expert_us):.2f}x"
    )
    torch.accelerator.empty_cache()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA GPU")
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")
    torch.manual_seed(args.seed)
    print_header(args)
    with torch.inference_mode():
        for value in args.tokens.split(","):
            tokens = int(value)
            if tokens <= 0:
                raise ValueError(f"token count must be positive, got {tokens}")
            benchmark_shape(tokens, args)


if __name__ == "__main__":
    main()
