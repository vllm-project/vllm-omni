# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the Qwen-Image-Edit select01 modulation hot path.

This script intentionally starts with the current PyTorch implementation. It is
used as the before/after baseline for SGLang PR #20395/#21318 style fusion:

* layernorm + scale/shift + binary select01 gate
* residual + gated update + layernorm + scale/shift + binary select01 gate

Run on a CUDA machine for actionable numbers, for example:

    python benchmarks/kernels/bench_qwen_select01_modulation.py \
        --device cuda --dtype bf16 --seq-lens 4096 8192 16384 \
        --output-json /tmp/qwen_select01_baseline.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F

for candidate in Path(__file__).resolve().parents:
    if (candidate / "vllm_omni").is_dir():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break


_FUSED_FNS: tuple[Callable[..., object], Callable[..., object]] | None = None


def _dtype(name: str) -> torch.dtype:
    return {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }[name]


def _select01_modulation(
    mod_params: torch.Tensor,
    index: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shift, scale, gate = mod_params.chunk(3, dim=-1)

    if index is None:
        return scale.unsqueeze(1), shift.unsqueeze(1), gate.unsqueeze(1)

    actual_batch = shift.size(0) // 2
    shift_0, shift_1 = shift[:actual_batch], shift[actual_batch:]
    scale_0, scale_1 = scale[:actual_batch], scale[actual_batch:]
    gate_0, gate_1 = gate[:actual_batch], gate[actual_batch:]

    idx = index.unsqueeze(-1) == 0
    shift = torch.where(idx, shift_0.unsqueeze(1), shift_1.unsqueeze(1))
    scale = torch.where(idx, scale_0.unsqueeze(1), scale_1.unsqueeze(1))
    gate = torch.where(idx, gate_0.unsqueeze(1), gate_1.unsqueeze(1))
    return scale, shift, gate


def _native_layernorm_select01(
    x: torch.Tensor,
    mod_params: torch.Tensor,
    index: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale, shift, gate = _select01_modulation(mod_params, index)
    out = F.layer_norm(x.float(), (x.shape[-1],), eps=eps).to(x.dtype)
    return out * (1 + scale) + shift, gate


def _native_residual_layernorm_select01(
    x: torch.Tensor,
    residual: torch.Tensor,
    residual_gate: torch.Tensor,
    mod_params: torch.Tensor,
    index: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scale, shift, gate = _select01_modulation(mod_params, index)
    residual_out = residual + residual_gate * x
    out = F.layer_norm(residual_out.float(), (residual_out.shape[-1],), eps=eps).to(residual_out.dtype)
    return out * (1 + scale) + shift, residual_out, gate


def _load_fused_fns() -> tuple[Callable[..., object], Callable[..., object]]:
    global _FUSED_FNS
    if _FUSED_FNS is None:
        module_paths = [
            *(
                candidate / "vllm_omni/diffusion/layers/qwen_select01_modulation.py"
                for candidate in Path(__file__).resolve().parents
            ),
            Path(__file__).with_name("qwen_select01_modulation.py"),
        ]
        module_path = next((path for path in module_paths if path.exists()), None)
        if module_path is None:
            from vllm_omni.diffusion.layers.qwen_select01_modulation import (
                fused_layernorm_select01,
                fused_residual_layernorm_select01,
            )
        else:
            spec = importlib.util.spec_from_file_location("qwen_select01_modulation", module_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load fused module from {module_path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            fused_layernorm_select01 = module.fused_layernorm_select01
            fused_residual_layernorm_select01 = module.fused_residual_layernorm_select01

        _FUSED_FNS = (fused_layernorm_select01, fused_residual_layernorm_select01)
    return _FUSED_FNS


def _time_cuda(fn: Callable[[], object], warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.accelerator.synchronize()

    times: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.accelerator.synchronize()
        times.append(float(start.elapsed_time(end)))
    return times


def _reset_peak_memory_stats(device: torch.device) -> None:
    reset_peak_memory_stats = getattr(torch.accelerator, "reset_peak_memory_stats", None)
    if reset_peak_memory_stats is None:
        reset_peak_memory_stats = getattr(getattr(torch, device.type), "reset_peak_memory_stats")
    reset_peak_memory_stats(device)


def _max_memory_allocated(device: torch.device) -> int:
    max_memory_allocated = getattr(torch.accelerator, "max_memory_allocated", None)
    if max_memory_allocated is None:
        max_memory_allocated = getattr(getattr(torch, device.type), "max_memory_allocated")
    return int(max_memory_allocated(device))


def _time_cpu(fn: Callable[[], object], warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        fn()

    times: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def _summarize(times: list[float]) -> dict[str, float]:
    ordered = sorted(times)
    return {
        "median_ms": statistics.median(ordered),
        "min_ms": ordered[0],
        "p20_ms": ordered[max(0, int(len(ordered) * 0.20) - 1)],
        "p80_ms": ordered[min(len(ordered) - 1, int(len(ordered) * 0.80))],
    }


def _make_inputs(
    batch: int,
    seq_len: int,
    hidden_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {
        "x": torch.randn(batch, seq_len, hidden_size, device=device, dtype=dtype),
        "residual": torch.randn(batch, seq_len, hidden_size, device=device, dtype=dtype),
        "residual_gate": torch.randn(batch, 1, hidden_size, device=device, dtype=dtype),
        "mod_params": torch.randn(batch * 2, hidden_size * 3, device=device, dtype=dtype),
        "index": torch.randint(0, 2, (batch, seq_len), device=device, dtype=torch.int64),
    }


def _bench_one(
    name: str,
    fn: Callable[[], object],
    device: torch.device,
    warmup: int,
    iters: int,
) -> dict[str, object]:
    if device.type == "cuda":
        _reset_peak_memory_stats(device)
        times = _time_cuda(fn, warmup, iters)
        peak_memory_mb = _max_memory_allocated(device) / 1024**2
    else:
        times = _time_cpu(fn, warmup, iters)
        peak_memory_mb = None

    return {
        "name": name,
        **_summarize(times),
        "peak_memory_mb": peak_memory_mb,
    }


def _run(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available on this machine.")

    dtype = _dtype(args.dtype)
    rows: list[dict[str, object]] = []

    for seq_len in args.seq_lens:
        tensors = _make_inputs(args.batch_size, seq_len, args.hidden_size, dtype, device)
        impls = ["native"] if args.impl == "native" else ["fused"] if args.impl == "fused" else ["native", "fused"]

        def first_norm() -> object:
            return _native_layernorm_select01(
                tensors["x"],
                tensors["mod_params"],
                tensors["index"],
                args.eps,
            )

        def residual_norm() -> object:
            return _native_residual_layernorm_select01(
                tensors["x"],
                tensors["residual"],
                tensors["residual_gate"],
                tensors["mod_params"],
                tensors["index"],
                args.eps,
            )

        callables: list[tuple[str, Callable[[], object]]] = []
        if "native" in impls:
            callables.extend(
                [
                    ("native_layernorm_select01", first_norm),
                    ("native_residual_layernorm_select01", residual_norm),
                ]
            )
        if "fused" in impls:
            fused_layernorm_select01, fused_residual_layernorm_select01 = _load_fused_fns()

            def fused_first_norm() -> object:
                return fused_layernorm_select01(
                    tensors["x"],
                    tensors["mod_params"],
                    tensors["index"],
                    args.eps,
                )

            def fused_residual_norm() -> object:
                return fused_residual_layernorm_select01(
                    tensors["x"],
                    tensors["residual"],
                    tensors["residual_gate"].expand_as(tensors["residual"]),
                    tensors["mod_params"],
                    tensors["index"],
                    args.eps,
                )

            callables.extend(
                [
                    ("fused_layernorm_select01", fused_first_norm),
                    ("fused_residual_layernorm_select01", fused_residual_norm),
                ]
            )

        if args.check and "fused" in impls:
            fused_layernorm_select01, fused_residual_layernorm_select01 = _load_fused_fns()
            ref_norm, ref_gate = first_norm()
            actual_norm, actual_gate = fused_layernorm_select01(
                tensors["x"],
                tensors["mod_params"],
                tensors["index"],
                args.eps,
            )
            torch.testing.assert_close(actual_norm, ref_norm, atol=args.check_atol, rtol=args.check_rtol)
            torch.testing.assert_close(actual_gate, ref_gate, atol=0, rtol=0)
            ref_norm, ref_residual, ref_gate = residual_norm()
            actual_norm, actual_residual, actual_gate = fused_residual_layernorm_select01(
                tensors["x"],
                tensors["residual"],
                tensors["residual_gate"].expand_as(tensors["residual"]),
                tensors["mod_params"],
                tensors["index"],
                args.eps,
            )
            torch.testing.assert_close(actual_norm, ref_norm, atol=args.check_atol, rtol=args.check_rtol)
            torch.testing.assert_close(actual_residual, ref_residual, atol=args.check_atol, rtol=args.check_rtol)
            torch.testing.assert_close(actual_gate, ref_gate, atol=0, rtol=0)

        for name, fn in callables:
            row = _bench_one(name, fn, device, args.warmup, args.iters)
            row.update(
                {
                    "batch_size": args.batch_size,
                    "seq_len": seq_len,
                    "hidden_size": args.hidden_size,
                    "dtype": args.dtype,
                    "device": str(device),
                }
            )
            rows.append(row)
            print(
                f"{name:<38} B={args.batch_size:<2} L={seq_len:<6} C={args.hidden_size:<5} "
                f"median={row['median_ms']:.4f} ms p20={row['p20_ms']:.4f} p80={row['p80_ms']:.4f}"
            )

    result = {
        "benchmark": "qwen_select01_modulation",
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "rows": rows,
    }

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", default=default_device)
    parser.add_argument("--dtype", default="bf16", choices=["fp32", "float32", "fp16", "float16", "bf16", "bfloat16"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[4096, 8192, 16384])
    parser.add_argument("--hidden-size", type=int, default=3072)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--impl", choices=["native", "fused", "all"], default="native")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--check-atol", type=float, default=5e-2)
    parser.add_argument("--check-rtol", type=float, default=5e-2)
    parser.add_argument("--output-json")
    return parser.parse_args()


if __name__ == "__main__":
    _run(_parse_args())
