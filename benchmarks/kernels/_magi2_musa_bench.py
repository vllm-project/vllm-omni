# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Cold-cache torch.cuda timing and correctness gates for MAGI-2 microbenchmarks."""

import argparse
import importlib
import importlib.metadata
import json
import logging
import math
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

LOGGER = logging.getLogger(__name__)


def emit(**fields: Any) -> None:
    LOGGER.info(json.dumps(fields, sort_keys=True))


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 4, 16, 64, 129, 4096])
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--num-tests", type=int, default=30, help="Measured samples per implementation.")
    parser.add_argument("--warmup", type=int, default=10, help="Interleaved warmup calls before sampling.")
    parser.add_argument("--seed", type=int, default=42)


def load_musa(args: argparse.Namespace, parser: argparse.ArgumentParser) -> Any:
    if min(*args.tokens, args.num_tests) <= 0 or args.warmup < 0:
        parser.error("tokens and num-tests must be positive; warmup must be nonnegative")
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    torch = importlib.import_module("torch")
    cuda = getattr(torch, "cuda")
    # torchada is needed only on MUSA; after it is loaded, use torch.cuda for
    # both CUDA and MUSA as documented by torchada.
    if hasattr(torch.version, "musa") and torch.version.musa is not None:
        try:
            importlib.import_module("torchada")
        except ModuleNotFoundError as error:
            if error.name != "torchada":
                raise
    has_gpu = cuda.is_available() or (
        hasattr(torch.version, "musa") and torch.version.musa is not None and torch.musa.is_available()
    )
    if not has_gpu:
        emit(status="skip", reason="CUDA/MUSA device is unavailable", torch=torch.__version__)
        return None
    torch.manual_seed(args.seed)
    cuda.manual_seed_all(args.seed)
    return torch


def report_environment(torch: Any, args: argparse.Namespace, module: ModuleType, flag: str) -> None:
    root = Path(__file__).resolve().parents[2]
    packages = {}
    for name in ("torch", "torch_musa", "torchada", "triton", "vllm", "vllm-omni"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    cuda = getattr(torch, "cuda")
    properties = cuda.get_device_properties(cuda.current_device())
    try:
        source_sha = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        source_dirty = bool(subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True))
    except (subprocess.CalledProcessError, FileNotFoundError):
        source_sha = os.environ.get("MAGI2_BENCH_SOURCE_SHA", "unknown")
        source_dirty = None
    emit(
        status="environment",
        command=sys.argv,
        source_root=str(root),
        source_sha=source_sha,
        source_dirty=source_dirty,
        kernel_source=module.__file__,
        device=cuda.get_device_name(cuda.current_device()),
        sm_count=getattr(properties, "multi_processor_count", None),
        packages=packages,
        env={name: os.environ.get(name) for name in (flag, "MUSA_VISIBLE_DEVICES", "TORCHDYNAMO_DISABLE")},
        config=vars(args),
        mode="eager operator diagnostic; no compile or graph capture",
        comparison=f"forward_native (equivalent to {flag}=0) versus CustomOp dispatch ({flag}=1)",
        timing="torch.cuda.Event; alternating native/fused calls; vanilla 512 MiB flush outside timed regions",
        timing_scope="whole callable, including device launch gaps; not isolated kernel duration",
        l2_flush=True,
        l2_flush_bytes=512 * 1024 * 1024,
    )


class _LaunchProbe:
    def __init__(self, kernel: Any) -> None:
        self.kernel = kernel
        self.count = 0

    def __getitem__(self, grid: Any) -> Any:
        self.count += 1
        return self.kernel[grid]


def check_pair(
    torch: Any,
    reference: Callable,
    fused: Callable,
    module: ModuleType,
    kernels: tuple[str, ...],
    *,
    rtol: float,
    atol: float,
) -> dict:
    cuda = getattr(torch, "cuda")
    expected = reference()
    probes = {name: _LaunchProbe(getattr(module, name)) for name in kernels}
    try:
        for name, probe in probes.items():
            setattr(module, name, probe)
        actual = fused()
        cuda.synchronize()
    finally:
        for name, probe in probes.items():
            setattr(module, name, probe.kernel)
    if any(probe.count != 1 for probe in probes.values()):
        raise RuntimeError(f"Expected one launch per fused kernel; got {[(k, p.count) for k, p in probes.items()]}")
    expected = expected if isinstance(expected, tuple) else (expected,)
    actual = actual if isinstance(actual, tuple) else (actual,)
    errors = []
    for result, baseline in zip(actual, expected, strict=True):
        for value in (baseline, result):
            if not torch.isfinite(value).all().item():
                raise AssertionError("Nonfinite benchmark output")
            if value.numel() > 1 and (value == value.flatten()[0]).all().item():
                raise AssertionError("Constant benchmark output")
        torch.testing.assert_close(result, baseline, rtol=rtol, atol=atol)
        difference = (result.float() - baseline.float()).abs()
        errors.append(
            {
                "max_abs": difference.max().item(),
                "max_rel": (difference / baseline.float().abs().clamp_min(1e-6)).max().item(),
            }
        )
    return {"parity": "pass", "errors": errors, "rtol": rtol, "atol": atol, "kernels": list(kernels)}


def _percentile(samples: list[float], quantile: float) -> float:
    ordered = sorted(samples)
    position = (len(ordered) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def measure_pair(torch: Any, reference: Callable, fused: Callable, args: argparse.Namespace) -> dict:
    cuda = getattr(torch, "cuda")
    functions = (reference, fused)
    order: list[int] = []

    def alternate() -> Any:
        mode = len(order) % 2
        order.append(mode)
        return functions[mode]()

    times_ms = []
    for _ in range(args.warmup):
        alternate()
    cuda.synchronize()
    order.clear()
    flush = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    for _ in range(2 * args.num_tests):
        flush.zero_()
        cuda.synchronize()
        start = cuda.Event(enable_timing=True)
        end = cuda.Event(enable_timing=True)
        start.record()
        alternate()
        end.record()
        end.synchronize()
        times_ms.append(start.elapsed_time(end))
    samples: tuple[list[float], list[float]] = ([], [])
    for mode, milliseconds in zip(order[-len(times_ms) :], times_ms, strict=True):
        if not math.isfinite(milliseconds) or milliseconds <= 0:
            raise RuntimeError(f"Invalid GPU event sample: {milliseconds}")
        samples[mode].append(milliseconds * 1000)
    stats = {}
    for label, values in zip(("native", "fused"), samples, strict=True):
        if len(values) != args.num_tests:
            raise RuntimeError("GPU timing harness returned an unexpected sample count")
        p50 = _percentile(values, 0.5)
        stats[label] = {"median_us": p50, "p50_us": p50, "p90_us": _percentile(values, 0.9), "samples": len(values)}
    stats["speedup_p50"] = stats["native"]["p50_us"] / stats["fused"]["p50_us"]
    return stats
