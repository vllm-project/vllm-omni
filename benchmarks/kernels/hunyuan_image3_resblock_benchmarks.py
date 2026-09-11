# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

_SEED = 20260911
_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
_TOLERANCES = {
    torch.float32: (2e-5, 2e-5),
    torch.float16: (5e-3, 5e-3),
    torch.bfloat16: (2e-2, 2e-2),
}


@dataclass(frozen=True)
class BlockCase:
    batch: int
    in_channels: int
    out_channels: int
    height: int
    width: int
    use_conv: bool = False
    up: bool = False
    down: bool = False


CASES = {
    "identity": BlockCase(1, 128, 128, 16, 16),
    "batched_identity": BlockCase(2, 128, 128, 32, 32),
    "channel_change": BlockCase(1, 128, 256, 16, 16),
    "convolutional_skip": BlockCase(1, 128, 256, 16, 16, use_conv=True),
    "production_down_branch": BlockCase(1, 128, 128, 32, 32, down=True),
    "production_up_branch": BlockCase(1, 128, 128, 16, 16, up=True),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark HunyuanImage3 native/NVIDIA ResBlocks with identical weights and inputs."
    )
    parser.add_argument("--dtype", choices=tuple(_DTYPES), default="bfloat16")
    parser.add_argument("--case", choices=tuple(CASES), default="batched_identity")
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if args.warmups <= 0:
        parser.error("--warmups must be positive")
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    if args.samples < 2:
        parser.error("--samples must be at least 2 to compute quartiles")
    return args


def _load_runtime_types() -> tuple[type[torch.nn.Module], type[torch.nn.Module], object]:
    from vllm_omni.diffusion.models.hunyuan_image3.layers.native.transformer_blocks import (
        ResBlock as NativeResBlock,
    )
    from vllm_omni.diffusion.models.hunyuan_image3.layers.nvidia.transformer_blocks import (
        ResBlock as NvidiaResBlock,
    )
    from vllm_omni.platforms import current_omni_platform

    return NativeResBlock, NvidiaResBlock, current_omni_platform


def _constructor_kwargs(case: BlockCase, dtype: torch.dtype) -> dict[str, object]:
    return {
        "in_channels": case.in_channels,
        "emb_channels": 512,
        "out_channels": case.out_channels,
        "dropout": 0.0,
        "use_conv": case.use_conv,
        "dims": 2,
        "up": case.up,
        "down": case.down,
        "device": torch.device("cuda"),
        "dtype": dtype,
    }


def _randomize_parameters(module: torch.nn.Module, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(_SEED)
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            values = torch.randn(
                parameter.shape,
                generator=generator,
                device=parameter.device,
                dtype=torch.float32,
            )
            if parameter.ndim == 1 and name.endswith("weight"):
                values = 1.0 + values * 0.05
            else:
                values = values * 0.02
            parameter.copy_(values.to(dtype=dtype))
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def _make_blocks(
    case: BlockCase,
    dtype: torch.dtype,
) -> tuple[torch.nn.Module, torch.nn.Module, object]:
    NativeResBlock, NvidiaResBlock, current_omni_platform = _load_runtime_types()
    if not current_omni_platform.is_cuda():
        raise RuntimeError("HunyuanImage3 NVIDIA ResBlock benchmark requires CUDA")

    kwargs = _constructor_kwargs(case, dtype)
    native = NativeResBlock(**kwargs).eval()
    nvidia = NvidiaResBlock(**kwargs).eval()
    state = _randomize_parameters(native, dtype)
    nvidia.load_state_dict(state, strict=True)
    return native, nvidia, current_omni_platform


def _make_inputs(case: BlockCase, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(_SEED + 1)
    x = torch.randn(
        case.batch,
        case.in_channels,
        case.height,
        case.width,
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    ).to(dtype)
    emb = torch.randn(
        case.batch,
        512,
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    ).to(dtype)
    return x, emb


def _time_block(
    block: torch.nn.Module,
    x: torch.Tensor,
    emb: torch.Tensor,
    *,
    warmups: int,
    iterations: int,
) -> float:
    with torch.inference_mode():
        for _ in range(warmups):
            block(x, emb)
        torch.accelerator.synchronize()

        start = torch.Event(enable_timing=True)
        end = torch.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            block(x, emb)
        end.record()
        end.synchronize()
    return start.elapsed_time(end) / iterations


def _timing_summary(samples: list[float]) -> dict[str, object]:
    q1, _, q3 = statistics.quantiles(samples, n=4)
    return {
        "samples": samples,
        "median": statistics.median(samples),
        "q1": q1,
        "q3": q3,
    }


def _git_head() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _run(args: argparse.Namespace) -> dict[str, object]:
    dtype = _DTYPES[args.dtype]
    case = CASES[args.case]
    native, nvidia, current_omni_platform = _make_blocks(case, dtype)
    x, emb = _make_inputs(case, dtype)

    with torch.inference_mode():
        native_output = native(x, emb)
        nvidia_output = nvidia(x, emb)
    rtol, atol = _TOLERANCES[dtype]
    torch.testing.assert_close(nvidia_output, native_output, rtol=rtol, atol=atol)
    abs_diff = (nvidia_output.float() - native_output.float()).abs()

    timings = {"native": [], "nvidia": []}
    for sample_index in range(args.samples):
        order = ("native", "nvidia") if sample_index % 2 == 0 else ("nvidia", "native")
        blocks = {"native": native, "nvidia": nvidia}
        for name in order:
            torch.accelerator.synchronize()
            timings[name].append(
                _time_block(
                    blocks[name],
                    x,
                    emb,
                    warmups=args.warmups,
                    iterations=args.iterations,
                )
            )

    native_summary = _timing_summary(timings["native"])
    nvidia_summary = _timing_summary(timings["nvidia"])
    native_median = float(native_summary["median"])
    nvidia_median = float(nvidia_summary["median"])
    relative_change = (nvidia_median / native_median - 1.0) * 100.0

    return {
        "scope": "HunyuanImage3 ResBlock block-level only",
        "commit": _git_head(),
        "gpu": current_omni_platform.get_device_name(),
        "torch_version": torch.__version__,
        "accelerator_version": current_omni_platform.get_device_version(),
        "dtype": args.dtype,
        "case": args.case,
        "case_config": asdict(case),
        "shape": {
            "x": list(x.shape),
            "emb": list(emb.shape),
        },
        "warmups": args.warmups,
        "iterations": args.iterations,
        "samples": args.samples,
        "rtol": rtol,
        "atol": atol,
        "max_abs_diff": abs_diff.max().item(),
        "mean_abs_diff": abs_diff.mean().item(),
        "native_ms": native_summary,
        "nvidia_ms": nvidia_summary,
        "relative_change_percent": relative_change,
    }


def main() -> None:
    args = _parse_args()
    result = _run(args)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(f"{rendered}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
