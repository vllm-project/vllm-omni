# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compare MAGI-2 routed MoE with equal preprocessing boundaries.

python -m benchmarks.kernels.benchmark_magi2_bf16_moe --tokens 2 4096 10000

Default (`--mode dispatch`): the same MAGI-2 layer is timed with
`MAGI2_USE_BF16_MOE_KERNEL=0/1`, including routing and all preprocessing.
With `--mode routed`, routing probabilities/IDs are inputs to BOTH timed paths. Sorting,
padding, packing, allocations and reduction remain inside their respective
calls. The reference uses deterministic routed storage in diagnostic mode.
Use --mode dispatch to measure the
actual layer with MAGI2_USE_BF16_MOE_KERNEL=0/1, including the router. Unsupported
dispatch fails explicitly; it is never silently replaced by another baseline.

Stage timings are standalone diagnostics, not additive or env speedup claims.
These synthetic local shapes are not full-model performance measurements.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import statistics
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[2, 4096])
    parser.add_argument("--heads", type=int, default=3)
    parser.add_argument("--experts", type=int, default=8, help="Experts per local head (synthetic default).")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=419)
    parser.add_argument("--mode", choices=["routed", "dispatch"], default="dispatch")
    parser.add_argument("--stages", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if min(*args.tokens, args.heads, args.experts, args.top_k, args.iters) < 1 or args.warmup < 0:
        parser.error("Shapes/iters must be positive; warmup must be nonnegative")
    if args.top_k > args.experts:
        parser.error("top-k must not exceed experts per head")
    return args


@contextmanager
def env_switch(enabled):
    # Restore both variables even if a backend compilation or correctness check fails.
    with patch.dict(
        os.environ,
        MAGI2_USE_BF16_MOE_KERNEL=str(int(enabled)),
        # Triton-MUSA 3.6 and CUDA Triton 3.7 both support the BF16 atomic
        # reference path. Keep env-off/on as a true dispatch A/B.
        MAGI2_DETERMINISTIC="0",
    ):
        yield


def stats(values):
    if not values or any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError("Missing/non-positive/non-finite timing sample")
    ordered = sorted(values)
    return {
        "mean_us": statistics.mean(values),
        "p50_us": statistics.median(values),
        "p90_us": ordered[math.ceil(0.9 * len(ordered)) - 1],
        "samples_us": values,
    }


def measure(torch, calls, args):
    """Alternating AB/BA; cold-cache flush is outside each timed region."""
    cuda = getattr(torch, "cuda")
    names = list(calls)
    sequence = names + names[::-1]
    wall_samples = []
    labels = []
    counter = 0

    def invoke():
        nonlocal counter
        name = sequence[counter % len(sequence)]
        counter += 1
        cuda.synchronize()
        start = time.perf_counter_ns()
        output = calls[name]()
        cuda.synchronize()
        elapsed = (time.perf_counter_ns() - start) / 1000
        wall_samples.append(elapsed)
        labels.append(name)
        return output

    count = len(names) * args.iters
    cache = torch.empty(512 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    gpu_ms = []
    for index in range(len(names) * args.warmup + count):
        cache.zero_()
        cuda.synchronize()
        begin, end = cuda.Event(enable_timing=True), cuda.Event(enable_timing=True)
        begin.record()
        invoke()
        end.record()
        end.synchronize()
        if index >= len(names) * args.warmup:
            gpu_ms.append(begin.elapsed_time(end))
    labels, wall_samples = labels[-count:], wall_samples[-count:]
    result = {}
    for name in names:
        wall = [x for label, x in zip(labels, wall_samples) if label == name]
        event = [float(x) * 1000 for label, x in zip(labels, gpu_ms) if label == name]
        if len(wall) != args.iters or len(event) != args.iters:
            raise RuntimeError("Timing harness did not return the requested sample counts")
        result[name] = {"synchronized_wall": stats(wall), "gpu_event": stats(event)}
    return result


def make_inputs(torch, tokens, args):
    # Fan-in-scaled synthetic weights. Unlike the old script, W_up and W_gate
    # have the same distribution and top-k IDs are unique within each token.
    torch.manual_seed(args.seed)
    x = torch.randn(tokens, args.heads, 256, device="cuda").to(torch.bfloat16)
    shape = (args.heads * args.experts, 256, 1280)
    gate, up = [(torch.randn(shape, device="cuda") / 16).to(torch.bfloat16) for _ in range(2)]
    down = (torch.randn(shape[0], 1280, 256, device="cuda") / math.sqrt(1280)).to(torch.bfloat16)
    logits = torch.randn(args.heads, tokens, args.experts, device="cuda")
    selected = logits.topk(args.top_k, dim=-1)
    probabilities = selected.values.softmax(-1)
    return x, probabilities, selected.indices, gate, up, down


def check_and_capture(torch, moe, kernels, data, args):
    x, probabilities, indices, gate, up, down = data
    gather, weights, offsets = moe.global_sort_routes(probabilities, indices, args.experts)
    sorted_routes, launches = [], []
    scatter, invoke = kernels._deterministic_scatter, moe.invoke_fused_moe_bf16

    def capture_scatter(routes, *params):
        sorted_routes.append(routes.detach().clone())
        return scatter(routes, *params)

    def capture_launch(*params, **kwargs):
        launches.append((params, kwargs))
        return invoke(*params, **kwargs)

    with patch.object(kernels, "_deterministic_scatter", capture_scatter):
        moe.triton_mh_moe_forward(x, gather, weights, offsets, gate, up, down, deterministic=True)
    with patch.object(moe, "invoke_fused_moe_bf16", capture_launch):
        actual = moe._bf16_fused_moe_forward(*data)
    if len(sorted_routes) != 1 or len(launches) != 2:
        raise RuntimeError("Unexpected kernel dispatch: benchmark contract needs updating")
    flat_experts = (indices + torch.arange(args.heads, device=x.device)[:, None, None] * args.experts).flatten()
    order = flat_experts.argsort(stable=True)
    new_routes = launches[1][0][2].reshape(-1, 256).index_select(0, order)
    scatter_ids = gather.long() * args.heads + flat_experts[order] // args.experts
    expected = torch.zeros(x.shape[0] * args.heads, 256, device=x.device, dtype=torch.float32)
    expected.index_add_(0, scatter_ids, sorted_routes[0].float())
    expected = expected.to(x.dtype).view_as(x)
    errors = []
    # Same expert/FP32 reduction contract as test_bf16_moe_wiring.py, rather
    # than relaxing tolerances on the inherited BF16 scatter reduction.
    for result, reference in ((new_routes, sorted_routes[0]), (actual, expected)):
        if not (torch.isfinite(result).all() and torch.isfinite(reference).all()):
            raise RuntimeError("Non-finite MoE output")
        torch.testing.assert_close(result, reference, rtol=2e-2, atol=0.5)
        relative_l2 = (result.float() - reference.float()).norm() / reference.float().norm().clamp_min(1e-6)
        if relative_l2.item() >= 1e-3:
            raise AssertionError(f"MoE relative L2 {relative_l2.item()} >= 1e-3")
        errors.append(
            {"max_abs": (result.float() - reference.float()).abs().max().item(), "relative_l2": relative_l2.item()}
        )
    return launches, errors


def benchmark(torch, moe, kernels, tokens, args):
    data = make_inputs(torch, tokens, args)
    x, probabilities, indices, gate, up, down = data
    launches, errors = check_and_capture(torch, moe, kernels, data, args)

    def reference():
        gather, weights, offsets = moe.global_sort_routes(probabilities, indices, args.experts)
        return moe.triton_mh_moe_forward(x, gather, weights, offsets, gate, up, down, deterministic=True)

    calls = {
        "reference_deterministic_routed": reference,
        "candidate_routed": lambda: moe._bf16_fused_moe_forward(*data),
    }
    if args.mode == "dispatch":
        config = moe.Magi2MultiHeadMoEConfig(
            args.heads * 256, args.heads, args.experts, args.top_k, 1280, torch.bfloat16
        )
        layer = moe.Magi2MultiHeadMoE(config).to(x.device)
        for parameter, value in ((layer.W_gate, gate), (layer.W_up, up), (layer.W_down, down)):
            parameter.copy_(value)
        layer.gate.normal_(std=1 / 16)

        def dispatch(enabled):
            with env_switch(enabled):
                return layer(x.reshape(tokens, -1))

        calls = {"env_off_dispatch": lambda: dispatch(False), "env_on_dispatch": lambda: dispatch(True)}
        # Includes reduction-policy differences; use unchanged tolerances.
        torch.testing.assert_close(dispatch(True), dispatch(False), rtol=2e-2, atol=0.5)
    timings = measure(torch, calls, args)
    names = list(calls)
    result = {"status": "pass", "tokens": tokens, "mode": args.mode, "correctness": errors, "timings": timings}
    result["speedup_p50_wall"] = (
        timings[names[0]]["synchronized_wall"]["p50_us"] / timings[names[1]]["synchronized_wall"]["p50_us"]
    )
    if args.stages:
        route_ids = (indices.int() + torch.arange(args.heads, device=x.device)[:, None, None] * args.experts).reshape(
            -1, args.top_k
        )

        def gems():
            for params, kwargs in launches:
                moe.invoke_fused_moe_bf16(*params, **kwargs)

        result["standalone_stages_not_additive"] = measure(
            torch,
            {
                "reference_sort": lambda: moe.global_sort_routes(probabilities, indices, args.experts),
                "candidate_align": lambda: moe._align_bf16_routes(route_ids, args.heads * args.experts, 128),
                "candidate_pack_w13": lambda: torch.stack((gate.transpose(1, 2), up.transpose(1, 2)), dim=2).reshape(
                    args.heads * args.experts, 2560, 256
                ),
                "candidate_two_gemms_prepared": gems,
            },
            args,
        )
    return result


def main():
    args = parse_args()
    import torch

    if hasattr(torch.version, "musa") and torch.version.musa is not None:
        importlib.import_module("torchada")  # before importing vLLM / capturing CUDA APIs
    cuda = getattr(torch, "cuda")
    has_gpu = cuda.is_available() or (
        hasattr(torch.version, "musa") and torch.version.musa is not None and torch.musa.is_available()
    )
    if not has_gpu:
        print(json.dumps({"status": "skip", "reason": "No visible CUDA/MUSA device"}))
        return
    import vllm_omni.diffusion.models.magi2.fused_moe_kernels as kernels
    import vllm_omni.diffusion.models.magi2.mh_moe as moe

    root = Path(__file__).resolve().parents[2]
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        sha = os.environ.get("MAGI2_BENCH_SOURCE_SHA", "unknown")
    report = {
        "device": cuda.get_device_name(),
        "sm_count": cuda.get_device_properties(0).multi_processor_count,
        "source_sha": sha,
        "module_path": moe.__file__,
        "torch": torch.__version__,
        "args": {**vars(args), "output": str(args.output)},
        "cache": "512 MiB flush outside each region",
        "capture": "eager; JIT warmed; no graph/compile",
        "results": [],
    }
    with torch.inference_mode():
        for tokens in args.tokens:
            result = benchmark(torch, moe, kernels, tokens, args)
            report["results"].append(result)
            print(json.dumps(result), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
