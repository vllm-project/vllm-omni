# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark diffusion RMSNorm/LayerNorm provider choices.

This ports the benchmark discipline from sgl-project/sglang#20632 into
vLLM-Omni without changing any production runtime path.  The script compares
the native vLLM-Omni math, PyTorch eager/compile baselines, upstream vLLM CUDA
custom ops when installed, and optional third-party providers such as
FlashInfer when available.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import platform
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

EPS = 1e-6

TORCH_LN = "torch.nn.LayerNorm"
VLLM_OMNI_RMS = "vllm_omni.diffusion.layers.RMSNorm"
VLLM_OMNI_LN = "vllm_omni.diffusion.layers.LayerNorm"
VLLM_C_RMS = "vllm._custom_ops.rms_norm"
VLLM_C_FUSED = "vllm._custom_ops.fused_add_rms_norm"

VALID_OPS = ("rmsnorm", "fused_add_rmsnorm", "layernorm")

PROVIDER_NAMES: dict[str, tuple[str, ...]] = {
    "rmsnorm": (
        "vllm_omni_native",
        "torch_builtin",
        "torch_compile_native",
        "torch_compile_builtin",
        "vllm_cuda",
        "flashinfer",
    ),
    "fused_add_rmsnorm": (
        "vllm_omni_native",
        "torch_compile_native",
        "vllm_cuda",
        "flashinfer",
    ),
    "layernorm": (
        "vllm_omni_native",
        "torch_builtin",
        "torch_compile_native",
        "flashinfer",
    ),
}


def synchronize_accelerator() -> None:
    if hasattr(torch, "accelerator") and torch.accelerator.is_available():
        torch.accelerator.synchronize()


@dataclass(frozen=True)
class ShapeCase:
    shape_id: str
    op: str
    input_shape: tuple[int, ...]
    source_model: str
    source_gpu_config: str
    source_impl: str

    @property
    def rows(self) -> int:
        value = 1
        for dim in self.input_shape[:-1]:
            value *= dim
        return value

    @property
    def hidden_size(self) -> int:
        return self.input_shape[-1]


@dataclass
class Provider:
    name: str
    fn: Callable[[], Any]
    reset: Callable[[], None] | None = None


@dataclass
class BenchRow:
    op: str
    provider: str
    dtype: str
    shape_id: str
    source_model: str
    source_gpu_config: str
    source_input_shape: str
    source_impl: str
    rows: int
    hidden_size: int
    status: str
    median_us: float | str
    p10_us: float | str
    p90_us: float | str
    min_us: float | str
    max_us: float | str
    max_abs_diff: float | str
    max_rel_diff: float | str
    error: str


ACTUAL_DIFFUSION_GROUPS: tuple[tuple[str, str, tuple[tuple[str, str, tuple[int, ...], str], ...]], ...] = (
    (
        "qwen",
        "1 GPU",
        (
            ("qwen_ln_4096x3072", "layernorm", (1, 4096, 3072), VLLM_OMNI_LN),
            ("qwen_ln_26x3072", "layernorm", (1, 26, 3072), VLLM_OMNI_LN),
            ("qwen_ln_6x3072", "layernorm", (1, 6, 3072), VLLM_OMNI_LN),
            ("qwen_rms_26x3584", "rmsnorm", (1, 26, 3584), VLLM_OMNI_RMS),
            ("qwen_rms_6x3584", "rmsnorm", (1, 6, 3584), VLLM_OMNI_RMS),
        ),
    ),
    (
        "qwen-edit",
        "1 GPU",
        (
            ("qwen_edit_ln_200x3072", "layernorm", (1, 200, 3072), VLLM_OMNI_LN),
            ("qwen_edit_ln_203x3072", "layernorm", (1, 203, 3072), VLLM_OMNI_LN),
            ("qwen_edit_ln_8308x3072", "layernorm", (1, 8308, 3072), TORCH_LN),
            ("qwen_edit_rms_200x3584", "rmsnorm", (1, 200, 3584), VLLM_OMNI_RMS),
            ("qwen_edit_rms_203x3584", "rmsnorm", (1, 203, 3584), VLLM_OMNI_RMS),
        ),
    ),
    (
        "flux",
        "1 GPU",
        (
            ("flux_ln_77x768", "layernorm", (1, 77, 768), TORCH_LN),
            ("flux_ln_512x3072", "layernorm", (1, 512, 3072), TORCH_LN),
            ("flux_ln_4096x3072", "layernorm", (1, 4096, 3072), TORCH_LN),
            ("flux_ln_4608x3072", "layernorm", (1, 4608, 3072), TORCH_LN),
            ("flux_rms_512x4096", "rmsnorm", (1, 512, 4096), VLLM_OMNI_RMS),
        ),
    ),
    (
        "flux2",
        "1 GPU",
        (
            ("flux2_ln_512x6144", "layernorm", (1, 512, 6144), TORCH_LN),
            ("flux2_ln_4096x6144", "layernorm", (1, 4096, 6144), TORCH_LN),
            ("flux2_ln_4608x6144", "layernorm", (1, 4608, 6144), TORCH_LN),
            ("flux2_rms_4608x48x128", "rmsnorm", (1, 4608, 48, 128), VLLM_OMNI_RMS),
        ),
    ),
    (
        "zimage",
        "1 GPU",
        (
            ("zimage_ln_4128x3840", "layernorm", (1, 4128, 3840), TORCH_LN),
            ("zimage_rms_32x3840", "rmsnorm", (1, 32, 3840), VLLM_OMNI_RMS),
            ("zimage_rms_4096x3840", "rmsnorm", (1, 4096, 3840), VLLM_OMNI_RMS),
            ("zimage_rms_4128x3840", "rmsnorm", (1, 4128, 3840), VLLM_OMNI_RMS),
            ("zimage_rms_32x2560", "rmsnorm", (32, 2560), VLLM_OMNI_RMS),
        ),
    ),
    (
        "wan-ti2v",
        "1 GPU",
        (
            ("wan_ti2v_ln_17850x3072", "layernorm", (1, 17850, 3072), VLLM_OMNI_LN),
            ("wan_ti2v_rms_17850x3072", "rmsnorm", (1, 17850, 3072), VLLM_OMNI_RMS),
            ("wan_ti2v_rms_512x3072", "rmsnorm", (1, 512, 3072), VLLM_OMNI_RMS),
            ("wan_ti2v_rms_512x4096", "rmsnorm", (1, 512, 4096), VLLM_OMNI_RMS),
        ),
    ),
    (
        "hunyuanvideo",
        "1 GPU",
        (
            ("hunyuan_ln_46x768", "layernorm", (1, 46, 768), TORCH_LN),
            ("hunyuan_ln_45x3072", "layernorm", (1, 45, 3072), VLLM_OMNI_LN),
            ("hunyuan_ln_27030x3072", "layernorm", (1, 27030, 3072), VLLM_OMNI_LN),
            ("hunyuan_ln_27075x3072", "layernorm", (1, 27075, 3072), VLLM_OMNI_LN),
            ("hunyuan_rms_140x4096", "rmsnorm", (1, 140, 4096), VLLM_OMNI_RMS),
            ("hunyuan_rms_45x24x128", "rmsnorm", (1, 45, 24, 128), VLLM_OMNI_RMS),
            ("hunyuan_rms_27030x24x128", "rmsnorm", (1, 27030, 24, 128), VLLM_OMNI_RMS),
            ("hunyuan_rms_27075x24x128", "rmsnorm", (1, 27075, 24, 128), VLLM_OMNI_RMS),
            ("hunyuan_fused_add_140x4096", "fused_add_rmsnorm", (140, 4096), VLLM_C_FUSED),
        ),
    ),
    (
        "mova-720p",
        "4 GPU, ulysses=4, ring=1",
        (
            ("mova_ln_101x1536", "layernorm", (1, 101, 1536), TORCH_LN),
            ("mova_ln_403x1536", "layernorm", (1, 403, 1536), TORCH_LN),
            ("mova_ln_44100x5120", "layernorm", (1, 44100, 5120), TORCH_LN),
            ("mova_ln_176400x5120", "layernorm", (1, 176400, 5120), VLLM_OMNI_LN),
            ("mova_rms_101x1536", "rmsnorm", (1, 101, 1536), VLLM_OMNI_RMS),
            ("mova_rms_101x5120", "rmsnorm", (1, 101, 5120), VLLM_OMNI_RMS),
            ("mova_rms_44100x1536", "rmsnorm", (1, 44100, 1536), VLLM_OMNI_RMS),
            ("mova_rms_44100x5120", "rmsnorm", (1, 44100, 5120), VLLM_OMNI_RMS),
            ("mova_rms_512x1536", "rmsnorm", (1, 512, 1536), VLLM_OMNI_RMS),
            ("mova_rms_512x4096", "rmsnorm", (1, 512, 4096), VLLM_OMNI_RMS),
            ("mova_rms_512x5120", "rmsnorm", (1, 512, 5120), VLLM_OMNI_RMS),
        ),
    ),
)


def actual_diffusion_shapes() -> list[ShapeCase]:
    return [
        ShapeCase(shape_id, op, input_shape, model, gpu_config, source_impl)
        for model, gpu_config, cases in ACTUAL_DIFFUSION_GROUPS
        for shape_id, op, input_shape, source_impl in cases
    ]


def parse_csv(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return mapping[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype {name!r}") from exc


def dtype_name(dtype: torch.dtype) -> str:
    mapping = {
        torch.bfloat16: "bf16",
        torch.float16: "fp16",
        torch.float32: "fp32",
    }
    return mapping.get(dtype, str(dtype).removeprefix("torch."))


def parse_int_csv(text: str) -> list[int]:
    return [int(item) for item in parse_csv(text)]


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def import_optional(module_name: str):
    return importlib.import_module(module_name)


def vllm_omni_native_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    input_dtype = x.dtype
    x_f32 = x.to(torch.float32)
    variance = x_f32.pow(2).mean(-1, keepdim=True)
    out = x_f32 * torch.rsqrt(variance + eps)
    out = weight.to(torch.float32) * out
    return out.to(input_dtype)


def vllm_omni_native_fused_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    summed_f32 = x.float() + residual.float()
    summed = summed_f32.to(x.dtype)
    variance = summed_f32.pow(2).mean(-1, keepdim=True)
    normed = summed_f32 * torch.rsqrt(variance + eps)
    out = (weight.to(torch.float32) * normed).to(x.dtype)
    return out, summed


def vllm_omni_native_layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    input_dtype = x.dtype
    return F.layer_norm(
        x.float(),
        (x.shape[-1],),
        weight.float(),
        bias.float(),
        eps,
    ).to(input_dtype)


def torch_builtin_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    if not hasattr(F, "rms_norm"):
        raise RuntimeError("torch.nn.functional.rms_norm is not available")
    return F.rms_norm(x, (x.shape[-1],), weight, eps=eps)


def compile_provider(fn: Callable[[], Any]) -> Callable[[], Any]:
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is not available")
    return torch.compile(fn, mode="reduce-overhead", fullgraph=False)


def make_case_tensors(
    case: ShapeCase,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> dict[str, torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    def randn(shape: tuple[int, ...], tensor_dtype: torch.dtype = dtype) -> torch.Tensor:
        return torch.randn(shape, generator=gen, dtype=tensor_dtype).to(device=device)

    hidden_size = case.hidden_size
    tensors = {
        "x": randn(case.input_shape),
        "weight": randn((hidden_size,)),
    }
    if case.op == "layernorm":
        tensors["bias"] = randn((hidden_size,))
    if case.op == "fused_add_rmsnorm":
        tensors["residual"] = randn(case.input_shape)
    return tensors


def build_rmsnorm_providers(
    tensors: dict[str, torch.Tensor],
    eps: float,
    provider_filter: set[str] | None,
) -> list[Provider]:
    x = tensors["x"]
    weight = tensors["weight"]
    providers: list[Provider] = []

    if wants_provider(provider_filter, "vllm_omni_native"):
        providers.append(Provider("vllm_omni_native", lambda: vllm_omni_native_rms_norm(x, weight, eps)))
    if wants_provider(provider_filter, "torch_builtin"):
        providers.append(Provider("torch_builtin", lambda: torch_builtin_rms_norm(x, weight, eps)))

    def native_fn():
        return vllm_omni_native_rms_norm(x, weight, eps)

    def builtin_fn():
        return torch_builtin_rms_norm(x, weight, eps)

    if wants_provider(provider_filter, "torch_compile_native"):
        providers.append(Provider("torch_compile_native", compile_provider(native_fn)))
    if wants_provider(provider_filter, "torch_compile_builtin"):
        providers.append(Provider("torch_compile_builtin", compile_provider(builtin_fn)))

    if wants_provider(provider_filter, "vllm_cuda"):
        try:
            from vllm import _custom_ops as ops

            x_2d = x.reshape(-1, x.shape[-1]).contiguous()
            out_2d = torch.empty_like(x_2d)

            def vllm_cuda():
                ops.rms_norm(out_2d, x_2d, weight, eps)
                return out_2d.reshape_as(x)

            providers.append(Provider("vllm_cuda", vllm_cuda))
        except Exception as exc:
            providers.append(unsupported_provider("vllm_cuda", exc))

    if wants_provider(provider_filter, "flashinfer"):
        try:
            flashinfer_norm = import_optional("flashinfer.norm")
            out = torch.empty_like(x)

            def flashinfer():
                return flashinfer_norm.rmsnorm(x, weight, eps=eps, out=out)

            providers.append(Provider("flashinfer", flashinfer))
        except Exception as exc:
            providers.append(unsupported_provider("flashinfer", exc))

    return filter_providers("rmsnorm", providers, provider_filter)


def build_fused_add_rmsnorm_providers(
    tensors: dict[str, torch.Tensor],
    eps: float,
    provider_filter: set[str] | None,
) -> list[Provider]:
    x = tensors["x"]
    residual = tensors["residual"]
    weight = tensors["weight"]
    providers: list[Provider] = []

    if wants_provider(provider_filter, "vllm_omni_native"):
        providers.append(
            Provider(
                "vllm_omni_native",
                lambda: vllm_omni_native_fused_add_rms_norm(x, residual, weight, eps),
            )
        )

    def native_fn():
        return vllm_omni_native_fused_add_rms_norm(x, residual, weight, eps)

    if wants_provider(provider_filter, "torch_compile_native"):
        providers.append(Provider("torch_compile_native", compile_provider(native_fn)))

    if wants_provider(provider_filter, "vllm_cuda"):
        try:
            from vllm import _custom_ops as ops

            base_x = x.clone()
            base_residual = residual.clone()
            work_x = x.clone()
            work_residual = residual.clone()

            def reset():
                work_x.copy_(base_x)
                work_residual.copy_(base_residual)

            def vllm_cuda():
                work_x_2d = work_x.reshape(-1, work_x.shape[-1])
                work_residual_2d = work_residual.reshape(-1, work_residual.shape[-1])
                ops.fused_add_rms_norm(work_x_2d, work_residual_2d, weight, eps)
                return work_x, work_residual

            providers.append(Provider("vllm_cuda", vllm_cuda, reset))
        except Exception as exc:
            providers.append(unsupported_provider("vllm_cuda", exc))

    if wants_provider(provider_filter, "flashinfer"):
        try:
            flashinfer_norm = import_optional("flashinfer.norm")
            base_x = x.clone()
            base_residual = residual.clone()
            work_x = x.clone()
            work_residual = residual.clone()

            def reset():
                work_x.copy_(base_x)
                work_residual.copy_(base_residual)

            def flashinfer():
                result = flashinfer_norm.fused_add_rmsnorm(
                    work_x,
                    work_residual,
                    weight,
                    eps=eps,
                )
                if isinstance(result, tuple):
                    return result
                return work_x, work_residual

            providers.append(Provider("flashinfer", flashinfer, reset))
        except Exception as exc:
            providers.append(unsupported_provider("flashinfer", exc))

    return filter_providers("fused_add_rmsnorm", providers, provider_filter)


def build_layernorm_providers(
    tensors: dict[str, torch.Tensor],
    eps: float,
    provider_filter: set[str] | None,
) -> list[Provider]:
    x = tensors["x"]
    weight = tensors["weight"]
    bias = tensors["bias"]
    providers: list[Provider] = []

    if wants_provider(provider_filter, "vllm_omni_native"):
        providers.append(Provider("vllm_omni_native", lambda: vllm_omni_native_layer_norm(x, weight, bias, eps)))
    if wants_provider(provider_filter, "torch_builtin"):
        providers.append(Provider("torch_builtin", lambda: F.layer_norm(x, (x.shape[-1],), weight, bias, eps)))

    def native_fn():
        return vllm_omni_native_layer_norm(x, weight, bias, eps)

    if wants_provider(provider_filter, "torch_compile_native"):
        providers.append(Provider("torch_compile_native", compile_provider(native_fn)))

    if wants_provider(provider_filter, "flashinfer"):
        try:
            flashinfer_norm = import_optional("flashinfer.norm")

            def flashinfer():
                return flashinfer_norm.layernorm(x, weight.float(), bias.float(), eps)

            providers.append(Provider("flashinfer", flashinfer))
        except Exception as exc:
            providers.append(unsupported_provider("flashinfer", exc))

    return filter_providers("layernorm", providers, provider_filter)


def unsupported_provider(name: str, exc: Exception) -> Provider:
    def raise_unsupported():
        raise RuntimeError(f"provider unavailable: {type(exc).__name__}: {exc}")

    return Provider(name, raise_unsupported)


def wants_provider(provider_filter: set[str] | None, provider_name: str) -> bool:
    return provider_filter is None or provider_name in provider_filter


def filter_providers(
    op_name: str,
    providers: list[Provider],
    provider_filter: set[str] | None,
) -> list[Provider]:
    if provider_filter is None:
        return providers
    valid = set(PROVIDER_NAMES[op_name])
    unknown = provider_filter - valid
    if unknown:
        raise ValueError(f"Unknown provider(s) for {op_name}: {sorted(unknown)}")
    return [provider for provider in providers if provider.name in provider_filter]


def build_providers(
    case: ShapeCase,
    tensors: dict[str, torch.Tensor],
    eps: float,
    provider_filter: set[str] | None,
) -> list[Provider]:
    if case.op == "rmsnorm":
        return build_rmsnorm_providers(tensors, eps, provider_filter)
    if case.op == "fused_add_rmsnorm":
        return build_fused_add_rmsnorm_providers(tensors, eps, provider_filter)
    if case.op == "layernorm":
        return build_layernorm_providers(tensors, eps, provider_filter)
    raise ValueError(f"Unsupported op {case.op!r}")


def reference_output(case: ShapeCase, tensors: dict[str, torch.Tensor], eps: float) -> Any:
    if case.op == "rmsnorm":
        return vllm_omni_native_rms_norm(tensors["x"], tensors["weight"], eps)
    if case.op == "fused_add_rmsnorm":
        return vllm_omni_native_fused_add_rms_norm(
            tensors["x"],
            tensors["residual"],
            tensors["weight"],
            eps,
        )
    if case.op == "layernorm":
        return vllm_omni_native_layer_norm(
            tensors["x"],
            tensors["weight"],
            tensors["bias"],
            eps,
        )
    raise ValueError(f"Unsupported op {case.op!r}")


def flatten_output(output: Any) -> tuple[torch.Tensor, ...]:
    if isinstance(output, torch.Tensor):
        return (output,)
    if isinstance(output, (tuple, list)) and all(isinstance(item, torch.Tensor) for item in output):
        return tuple(output)
    raise TypeError(f"Provider returned unsupported output type {type(output).__name__}")


def compare_output(actual: Any, expected: Any) -> tuple[float, float]:
    actual_tensors = flatten_output(actual)
    expected_tensors = flatten_output(expected)
    if len(actual_tensors) != len(expected_tensors):
        raise ValueError(f"Output arity mismatch: actual={len(actual_tensors)} expected={len(expected_tensors)}")

    max_abs = 0.0
    max_rel = 0.0
    for actual_tensor, expected_tensor in zip(actual_tensors, expected_tensors):
        if actual_tensor.shape != expected_tensor.shape:
            raise ValueError(
                f"Output shape mismatch: actual={tuple(actual_tensor.shape)} expected={tuple(expected_tensor.shape)}"
            )
        diff = (actual_tensor.float() - expected_tensor.float()).abs()
        rel = diff / (expected_tensor.float().abs() + 1e-6)
        max_abs = max(max_abs, float(diff.max().item()))
        max_rel = max(max_rel, float(rel.max().item()))
    return max_abs, max_rel


def tolerance_for_dtype(dtype: torch.dtype, atol: float | None, rtol: float | None) -> tuple[float, float]:
    if atol is not None and rtol is not None:
        return atol, rtol
    if dtype == torch.float32:
        return atol if atol is not None else 1e-4, rtol if rtol is not None else 1e-4
    return atol if atol is not None else 5e-2, rtol if rtol is not None else 5e-2


def run_correctness_check(
    provider: Provider,
    expected: Any,
    dtype: torch.dtype,
    atol: float | None,
    rtol: float | None,
) -> tuple[str, float | str, float | str, str]:
    try:
        if provider.reset is not None:
            provider.reset()
        actual = provider.fn()
        synchronize_accelerator()
        max_abs, max_rel = compare_output(actual, expected)
        atol_value, rtol_value = tolerance_for_dtype(dtype, atol, rtol)
        if max_abs > atol_value and max_rel > rtol_value:
            return (
                "failed_correctness",
                max_abs,
                max_rel,
                f"max_abs={max_abs:.6g} > {atol_value:.6g} and max_rel={max_rel:.6g} > {rtol_value:.6g}",
            )
        return "ok", max_abs, max_rel, ""
    except Exception as exc:
        return "unsupported", "", "", f"{type(exc).__name__}: {exc}"


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    index = q * (len(ordered) - 1)
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    fraction = index - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def benchmark_provider(
    provider: Provider,
    warmup: int,
    iters: int,
    device: torch.device,
) -> tuple[float, float, float, float, float]:
    for _ in range(warmup):
        if provider.reset is not None:
            provider.reset()
        provider.fn()
    if device.type == "cuda":
        synchronize_accelerator()

    times_us: list[float] = []
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(iters):
            if provider.reset is not None:
                provider.reset()
            start.record()
            provider.fn()
            end.record()
            end.synchronize()
            times_us.append(start.elapsed_time(end) * 1000.0)
    else:
        for _ in range(iters):
            if provider.reset is not None:
                provider.reset()
            start_time = time.perf_counter()
            provider.fn()
            times_us.append((time.perf_counter() - start_time) * 1_000_000.0)

    return (
        statistics.median(times_us),
        percentile(times_us, 0.10),
        percentile(times_us, 0.90),
        min(times_us),
        max(times_us),
    )


def row_metadata(case: ShapeCase, provider: Provider, dtype: torch.dtype) -> dict[str, Any]:
    return {
        "op": case.op,
        "provider": provider.name,
        "dtype": dtype_name(dtype),
        "shape_id": case.shape_id,
        "source_model": case.source_model,
        "source_gpu_config": case.source_gpu_config,
        "source_input_shape": str(list(case.input_shape)),
        "source_impl": case.source_impl,
        "rows": case.rows,
        "hidden_size": case.hidden_size,
    }


def run_provider_row(
    case: ShapeCase,
    provider: Provider,
    expected: Any,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
    device: torch.device,
    skip_correctness: bool,
    benchmark_failed_correctness: bool,
    atol: float | None,
    rtol: float | None,
) -> BenchRow:
    metadata = row_metadata(case, provider, dtype)
    status = "ok"
    max_abs: float | str = ""
    max_rel: float | str = ""
    error = ""

    if not skip_correctness:
        status, max_abs, max_rel, error = run_correctness_check(provider, expected, dtype, atol, rtol)

    if status == "ok" or (status == "failed_correctness" and benchmark_failed_correctness):
        try:
            median_us, p10_us, p90_us, min_us, max_us = benchmark_provider(
                provider,
                warmup,
                iters,
                device,
            )
        except Exception as exc:
            status = "unsupported"
            median_us = p10_us = p90_us = min_us = max_us = ""
            error = f"{type(exc).__name__}: {exc}"
    else:
        median_us = p10_us = p90_us = min_us = max_us = ""

    return BenchRow(
        **metadata,
        status=status,
        median_us=median_us,
        p10_us=p10_us,
        p90_us=p90_us,
        min_us=min_us,
        max_us=max_us,
        max_abs_diff=max_abs,
        max_rel_diff=max_rel,
        error=error,
    )


def make_grid_cases(
    ops: list[str],
    hidden_sizes: list[int],
    batch_sizes: list[int],
) -> list[ShapeCase]:
    return [
        ShapeCase(
            shape_id=f"grid_{op}_{batch_size}x{hidden_size}",
            op=op,
            input_shape=(batch_size, hidden_size),
            source_model="grid",
            source_gpu_config="1 GPU",
            source_impl="synthetic grid",
        )
        for op in ops
        for batch_size in batch_sizes
        for hidden_size in hidden_sizes
    ]


def make_smoke_cases(ops: list[str]) -> list[ShapeCase]:
    smoke_shapes = {
        "rmsnorm": (4, 128),
        "fused_add_rmsnorm": (4, 128),
        "layernorm": (4, 128),
    }
    return [
        ShapeCase(
            shape_id=f"smoke_{op}_4x128",
            op=op,
            input_shape=smoke_shapes[op],
            source_model="smoke",
            source_gpu_config="1 GPU",
            source_impl="synthetic smoke",
        )
        for op in ops
    ]


def filter_cases(
    cases: list[ShapeCase],
    ops: set[str],
    models: set[str] | None,
    shape_ids: set[str] | None,
    include_multi_gpu_source_shapes: bool,
    limit_cases: int | None,
) -> list[ShapeCase]:
    filtered = [case for case in cases if case.op in ops]
    if models is not None:
        filtered = [case for case in filtered if case.source_model in models]
    if shape_ids is not None:
        filtered = [case for case in filtered if case.shape_id in shape_ids]
    if not include_multi_gpu_source_shapes:
        filtered = [case for case in filtered if case.source_gpu_config == "1 GPU"]
    if limit_cases is not None:
        filtered = filtered[:limit_cases]
    return filtered


def run_suite(args: argparse.Namespace) -> list[BenchRow]:
    ops = parse_csv(args.ops)
    validate_ops(ops)
    dtype_values = [dtype_from_name(name) for name in parse_csv(args.dtypes)]
    device = torch.device(args.device)
    provider_filter = None if args.providers == "all" else set(parse_csv(args.providers))

    if args.shape_preset == "diffusion-actual":
        cases = actual_diffusion_shapes()
    elif args.shape_preset == "smoke":
        cases = make_smoke_cases(ops)
    else:
        cases = make_grid_cases(ops, parse_int_csv(args.hidden_sizes), parse_int_csv(args.batch_sizes))

    cases = filter_cases(
        cases,
        set(ops),
        set(parse_csv(args.models)) if args.models else None,
        set(parse_csv(args.shape_ids)) if args.shape_ids else None,
        args.include_multi_gpu_source_shapes,
        args.limit_cases,
    )
    if not cases:
        raise RuntimeError("No benchmark cases selected")

    rows: list[BenchRow] = []
    for case_index, case in enumerate(cases):
        for dtype_index, dtype in enumerate(dtype_values):
            seed = args.seed + case_index * 1009 + dtype_index
            tensors = make_case_tensors(case, dtype, device, seed)
            expected = None
            if not args.skip_correctness:
                expected = reference_output(case, tensors, args.eps)
            if device.type == "cuda":
                synchronize_accelerator()
            providers = build_providers(case, tensors, args.eps, provider_filter)
            for provider in providers:
                row = run_provider_row(
                    case=case,
                    provider=provider,
                    expected=expected,
                    dtype=dtype,
                    warmup=args.warmup,
                    iters=args.iters,
                    device=device,
                    skip_correctness=args.skip_correctness,
                    benchmark_failed_correctness=args.benchmark_failed_correctness,
                    atol=args.atol,
                    rtol=args.rtol,
                )
                rows.append(row)
                print_row_progress(row)
    return rows


def validate_ops(ops: list[str]) -> None:
    invalid = sorted(set(ops) - set(VALID_OPS))
    if invalid:
        raise ValueError(f"Unsupported op(s): {invalid}. Valid ops: {list(VALID_OPS)}")


def print_row_progress(row: BenchRow) -> None:
    if row.status == "ok":
        print(f"{row.op}/{row.dtype}/{row.shape_id}/{row.provider}: {float(row.median_us):.3f} us")
    else:
        print(f"{row.op}/{row.dtype}/{row.shape_id}/{row.provider}: {row.status} {row.error}")


def write_csv(rows: list[BenchRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def json_default(obj: Any) -> Any:
    if isinstance(obj, torch.dtype):
        return dtype_name(obj)
    return str(obj)


def environment_metadata() -> dict[str, Any]:
    cuda_available = False
    cuda_probe_error = ""
    try:
        cuda_available = torch.cuda.is_available()
    except Exception as exc:
        cuda_probe_error = f"{type(exc).__name__}: {exc}"
    metadata: dict[str, Any] = {
        "python": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": cuda_available,
        "argv": sys.argv,
        "cwd": os.getcwd(),
    }
    if cuda_probe_error:
        metadata["cuda_probe_error"] = cuda_probe_error
    if cuda_available:
        try:
            device_index = torch.accelerator.current_device_index()
            props = torch.cuda.get_device_properties(device_index)
            metadata.update(
                {
                    "cuda_device_index": device_index,
                    "cuda_device_name": torch.cuda.get_device_name(device_index),
                    "cuda_device_capability": torch.cuda.get_device_capability(device_index),
                    "cuda_multi_processor_count": props.multi_processor_count,
                    "cuda_total_memory": props.total_memory,
                }
            )
        except Exception as exc:
            metadata["cuda_metadata_error"] = f"{type(exc).__name__}: {exc}"
    for module_name in ("vllm", "flashinfer"):
        try:
            module = import_optional(module_name)
            metadata[f"{module_name}_version"] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            metadata[f"{module_name}_version"] = f"unavailable: {type(exc).__name__}: {exc}"
    return metadata


def write_json(rows: list[BenchRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": environment_metadata(),
        "rows": [asdict(row) for row in rows],
    }
    output_path.write_text(
        json.dumps(payload, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )


def write_markdown(rows: list[BenchRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "# Diffusion Norm Provider Benchmark",
        "",
        "## Winners",
        "",
        "| op | dtype | shape | model | winner | median_us | spread_us |",
        "|---|---|---|---|---|---:|---:|",
    ]

    groups: dict[tuple[str, str, str], list[BenchRow]] = {}
    for row in rows:
        groups.setdefault((row.op, row.dtype, row.shape_id), []).append(row)

    for key in sorted(groups):
        ok_rows = [row for row in groups[key] if row.status == "ok" and row.median_us != ""]
        if not ok_rows:
            sample = groups[key][0]
            lines.append(f"| {sample.op} | {sample.dtype} | {sample.shape_id} | {sample.source_model} | none |  |  |")
            continue
        winner = min(ok_rows, key=lambda row: float(row.median_us))
        spread = float(winner.p90_us) - float(winner.p10_us)
        lines.append(
            f"| {winner.op} | {winner.dtype} | {winner.shape_id} | "
            f"{winner.source_model} | {winner.provider} | "
            f"{float(winner.median_us):.3f} | {spread:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Rows",
            "",
            "| op | dtype | shape | provider | status | median_us | max_abs_diff | error |",
            "|---|---|---|---|---|---:|---:|---|",
        ]
    )
    for row in rows:
        median = "" if row.median_us == "" else f"{float(row.median_us):.3f}"
        max_abs = "" if row.max_abs_diff == "" else f"{float(row.max_abs_diff):.6g}"
        error = str(row.error).replace("|", "\\|")
        lines.append(
            f"| {row.op} | {row.dtype} | {row.shape_id} | {row.provider} | "
            f"{row.status} | {median} | {max_abs} | {error} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def list_shapes(include_multi_gpu_source_shapes: bool) -> None:
    rows = actual_diffusion_shapes()
    if not include_multi_gpu_source_shapes:
        rows = [row for row in rows if row.source_gpu_config == "1 GPU"]
    for row in rows:
        print(f"{row.shape_id}\t{row.op}\t{list(row.input_shape)}\t{row.source_model}\t{row.source_gpu_config}")


def list_providers() -> None:
    for op_name in VALID_OPS:
        print(f"{op_name}: {', '.join(PROVIDER_NAMES[op_name])}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark RMSNorm/LayerNorm providers on diffusion shapes.")
    parser.add_argument(
        "--shape-preset",
        choices=("grid", "diffusion-actual", "smoke"),
        default="grid",
        help="Synthetic grid, SGLang #20632 diffusion shapes, or tiny smoke shapes.",
    )
    parser.add_argument(
        "--hidden-sizes",
        default="128,512,1024,3072,4096",
        help="Comma-separated hidden sizes for --shape-preset grid.",
    )
    parser.add_argument(
        "--batch-sizes",
        default="1,16,512",
        help="Comma-separated row counts for --shape-preset grid.",
    )
    parser.add_argument(
        "--ops",
        default="rmsnorm,fused_add_rmsnorm,layernorm",
        help="Comma-separated ops: rmsnorm, fused_add_rmsnorm, layernorm.",
    )
    parser.add_argument(
        "--dtypes",
        default="bf16,fp16",
        help="Comma-separated dtypes: bf16, fp16, fp32.",
    )
    parser.add_argument(
        "--providers",
        default="all",
        help="Comma-separated provider subset, or 'all'.",
    )
    parser.add_argument("--models", default="", help="Comma-separated source model filter.")
    parser.add_argument("--shape-ids", default="", help="Comma-separated shape_id filter.")
    parser.add_argument(
        "--include-multi-gpu-source-shapes",
        action="store_true",
        help="Include source shapes captured from multi-GPU SGLang runs.",
    )
    parser.add_argument("--limit-cases", type=int, default=None, help="Limit selected cases.")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device, usually cuda or cpu.",
    )
    parser.add_argument("--eps", type=float, default=EPS)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=None)
    parser.add_argument("--rtol", type=float, default=None)
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument(
        "--benchmark-failed-correctness",
        action="store_true",
        help="Still time providers that fail correctness tolerance.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(project_root() / "benchmarks" / "results" / "norm_impls"),
    )
    parser.add_argument("--list-shapes", action="store_true")
    parser.add_argument("--list-providers", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_shapes:
        list_shapes(args.include_multi_gpu_source_shapes)
        return
    if args.list_providers:
        list_providers()
        return
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")

    rows = run_suite(args)
    output_dir = Path(args.output_dir)
    csv_path = output_dir / "diffusion_norm_impls.csv"
    json_path = output_dir / "diffusion_norm_impls.json"
    md_path = output_dir / "diffusion_norm_impls_summary.md"
    write_csv(rows, csv_path)
    write_json(rows, json_path)
    write_markdown(rows, md_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
