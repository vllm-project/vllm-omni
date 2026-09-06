# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _time_cuda(fn, warmups: int, repeats: int) -> list[float]:
    for _ in range(warmups):
        fn()
    torch.accelerator.synchronize()
    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        torch.accelerator.synchronize()
        timings.append((time.perf_counter() - started) * 1_000)
    return timings


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "p10_ms": _percentile(values, 10),
        "p90_ms": _percentile(values, 90),
    }


def _parameter_bytes(module: torch.nn.Module) -> int:
    return sum(parameter.numel() * parameter.element_size() for parameter in module.parameters())


def main() -> None:
    args = _parse_args()
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29632")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")

    from vllm.config import DeviceConfig, VllmConfig
    from vllm.v1.worker.workspace import init_workspace_manager

    import vllm_omni.diffusion.models.hidream_image.hidream_image_transformer as hidream
    from vllm_omni.diffusion.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm_omni.diffusion.forward_context import set_forward_context

    class _BenchmarkPlatform:
        def __init__(self, *, cuda: bool):
            self.cuda = cuda

        def is_cuda(self) -> bool:
            return self.cuda

    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    vllm_config = VllmConfig(device_config=DeviceConfig(device="cuda"))

    with set_forward_context(vllm_config=vllm_config):
        init_distributed_environment(world_size=1, rank=0, local_rank=0)
        initialize_model_parallel()
        init_workspace_manager(device)

        torch.manual_seed(0)
        hidream.current_omni_platform = _BenchmarkPlatform(cuda=False)
        native = (
            hidream.MOEFeedForwardSwiGLU(
                dim=2560,
                hidden_dim=10240,
                num_routed_experts=4,
                num_activated_experts=2,
                _force_inference_output=True,
                prefix="benchmark.native",
            )
            .to(device=device, dtype=dtype)
            .eval()
        )

        hidream.current_omni_platform = _BenchmarkPlatform(cuda=True)
        packed = (
            hidream.MOEFeedForwardSwiGLU(
                dim=2560,
                hidden_dim=10240,
                num_routed_experts=4,
                num_activated_experts=2,
                _force_inference_output=True,
                prefix="benchmark.packed",
            )
            .to(device=device, dtype=dtype)
            .eval()
        )

        packed.gate.load_state_dict(native.gate.state_dict())
        packed.shared_experts.load_state_dict(native.shared_experts.state_dict())
        routed_experts = packed.experts.routed_experts
        with torch.no_grad():
            routed_experts.w13_weight.copy_(
                torch.stack([torch.cat((expert.w1.weight, expert.w3.weight)) for expert in native.experts])
            )
            routed_experts.w2_weight.copy_(torch.stack([expert.w2.weight for expert in native.experts]))
        routed_experts.quant_method.process_weights_after_loading(routed_experts)

        torch.manual_seed(1)
        hidden_states = torch.randn(1, args.tokens, 2560, device=device, dtype=dtype)
        with torch.inference_mode():
            expected = native(hidden_states)
            actual = packed(hidden_states)
            difference = (actual.float() - expected.float()).abs()
            tolerance = 1e-2 + 1e-2 * expected.float().abs()
            difference_sample = difference.flatten()[::16]
            correctness = {
                "atol": 1e-2,
                "rtol": 1e-2,
                "passes_combined_tolerance": bool(torch.all(difference <= tolerance).item()),
                "max_abs": difference.max().item(),
                "mean_abs": difference.mean().item(),
                "p99_abs_sampled": torch.quantile(difference_sample, 0.99).item(),
                "p99_sample_elements": difference_sample.numel(),
                "violating_elements": int((difference > tolerance).sum().item()),
                "total_elements": difference.numel(),
            }
            native_times = _time_cuda(lambda: native(hidden_states), args.warmups, args.repeats)
            packed_times = _time_cuda(lambda: packed(hidden_states), args.warmups, args.repeats)

    native_summary = _summary(native_times)
    packed_summary = _summary(packed_times)
    result = {
        "shape": {"tokens": args.tokens, "hidden_size": 2560, "experts": 4, "top_k": 2, "intermediate": 6912},
        "dtype": str(dtype),
        "gpu": torch.cuda.get_device_name(0),
        "warmups": args.warmups,
        "repeats": args.repeats,
        "correctness": correctness,
        "storage": {
            "native_routed_bytes": _parameter_bytes(native.experts),
            "packed_routed_bytes": _parameter_bytes(routed_experts),
            "native_total_bytes": _parameter_bytes(native),
            "packed_total_bytes": _parameter_bytes(packed),
        },
        "native": {"raw_ms": native_times, **native_summary},
        "packed": {"raw_ms": packed_times, **packed_summary},
        "speedup_median_percent": 100
        * (native_summary["median_ms"] - packed_summary["median_ms"])
        / native_summary["median_ms"],
        "backend": type(routed_experts.quant_method.moe_kernel).__name__,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
