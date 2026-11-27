#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Calibrate device copy bandwidth, dense matmul throughput and launch latency.

These microbenchmarks measure particular shapes and dispatch paths, not hardware
upper limits or universal launch minima. Faster kernels or other shapes may
exceed them. Use them for calibrated estimates, never strict physical bounds.

Runs on CUDA GPUs and on Ascend NPUs (``torch_npu``). The reduced-precision
float32 mode is named ``tf32`` on CUDA and ``hf32`` on Ascend; both appear under
``matmul_tflops`` so ``--compute-bound NAME=TFLOP,<dtype>`` selects the mode that
actually runs on the device.

``launch_latency_us`` is the per-kernel cost of a tiny op issued eagerly from
PyTorch (host dispatch included) and replayed from a captured graph; it feeds
``ceiling_report.py --launch-bound``. ``graph`` is null when capture fails.
"""

import argparse
import json
import time

import torch


class Backend:
    """Device-specific entry points used by the peak measurements."""

    def __init__(self, name: str):
        self.name = name
        self.module = torch.cuda if name == "cuda" else torch.npu
        # Ascend names the reduced-precision float32 cube mode HF32.
        self.reduced_fp32 = "tf32" if name == "cuda" else "hf32"

    def synchronize(self) -> None:
        self.module.synchronize()

    def device_name(self) -> str:
        return self.module.get_device_name()

    def _reduced_fp32_flags(self) -> list:
        if self.name == "cuda":
            return [torch.backends.cuda.matmul]
        # torch_npu gates HF32 per operator family; matmul is the one measured
        # here, but conv shares the flag on some releases.
        return [flags for flags in (getattr(torch.npu, "matmul", None),) if hasattr(flags, "allow_hf32")]

    def supports_reduced_fp32(self) -> bool:
        return bool(self._reduced_fp32_flags())

    def set_reduced_fp32(self, allow: bool) -> list[bool]:
        attr = "allow_tf32" if self.name == "cuda" else "allow_hf32"
        previous = []
        for flags in self._reduced_fp32_flags():
            previous.append(getattr(flags, attr))
            setattr(flags, attr, allow)
        return previous

    def restore_reduced_fp32(self, previous: list[bool]) -> None:
        attr = "allow_tf32" if self.name == "cuda" else "allow_hf32"
        for flags, value in zip(self._reduced_fp32_flags(), previous):
            setattr(flags, attr, value)


def resolve_backend(requested: str) -> Backend:
    if requested in ("auto", "cuda") and torch.cuda.is_available():
        return Backend("cuda")
    if requested in ("auto", "npu"):
        try:
            import torch_npu  # noqa: F401
        except ImportError:
            pass
        else:
            if torch.npu.is_available():
                return Backend("npu")
    raise SystemExit(f"no available device for --device {requested}")


def _time_ms(backend: Backend, fn, iters: int) -> float:
    for _ in range(3):
        fn()
    backend.synchronize()
    start = backend.module.Event(enable_timing=True)
    end = backend.module.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    backend.synchronize()
    return start.elapsed_time(end) / iters


def copy_bandwidth_gbs(backend: Backend, num_bytes: int, iters: int) -> float:
    src = torch.empty(num_bytes, dtype=torch.uint8, device=backend.name)
    dst = torch.empty_like(src)
    ms = _time_ms(backend, lambda: dst.copy_(src), iters)
    # A device copy reads and writes every byte.
    return 2 * num_bytes / (ms * 1e-3) / 1e9


def matmul_tflops(backend: Backend, dtype: torch.dtype, n: int, iters: int, reduced_fp32: bool) -> float:
    previous = backend.set_reduced_fp32(reduced_fp32)
    try:
        a = torch.randn(n, n, device=backend.name, dtype=dtype)
        b = torch.randn(n, n, device=backend.name, dtype=dtype)
        ms = _time_ms(backend, lambda: a @ b, iters)
    finally:
        backend.restore_reduced_fp32(previous)
    return 2 * n**3 / (ms * 1e-3) / 1e12


def _wall_us(backend: Backend, fn, iters: int) -> float:
    fn()
    backend.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    backend.synchronize()
    return (time.perf_counter() - start) / iters * 1e6


def launch_latency_us(backend: Backend, kernels: int, iters: int) -> dict[str, float | None]:
    x = torch.zeros(1, device=backend.name)

    def issue() -> None:
        for _ in range(kernels):
            x.add_(1)

    result: dict[str, float | None] = {"eager": _wall_us(backend, issue, iters) / kernels, "graph": None}
    graph_cls = getattr(backend.module, "CUDAGraph" if backend.name == "cuda" else "NPUGraph", None)
    capture = getattr(backend.module, "graph", None)
    if graph_cls is None or capture is None:
        return result
    try:
        graph = graph_cls()
        with capture(graph):
            issue()
        result["graph"] = _wall_us(backend, graph.replay, iters) / kernels
    except RuntimeError:
        pass
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "cuda", "npu"], default="auto")
    parser.add_argument("--copy-gib", type=float, default=4.0)
    parser.add_argument("--matmul-n", type=int, default=8192)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--launch-kernels", type=int, default=200, help="tiny kernels per launch-latency sample")
    parser.add_argument("--json", help="write results to this path")
    args = parser.parse_args()

    backend = resolve_backend(args.device)

    tflops = {
        "bf16": matmul_tflops(backend, torch.bfloat16, args.matmul_n, args.iters, reduced_fp32=False),
        "fp16": matmul_tflops(backend, torch.float16, args.matmul_n, args.iters, reduced_fp32=False),
        "fp32": matmul_tflops(backend, torch.float32, args.matmul_n, args.iters, reduced_fp32=False),
    }
    if backend.supports_reduced_fp32():
        tflops[backend.reduced_fp32] = matmul_tflops(
            backend, torch.float32, args.matmul_n, args.iters, reduced_fp32=True
        )

    result = {
        "measurement_kind": "empirical_calibration",
        "measurement_parameters": {
            "matmul_n": args.matmul_n,
            "copy_gib": args.copy_gib,
            "iters": args.iters,
            "launch_kernels": args.launch_kernels,
        },
        "device_type": backend.name,
        "device": backend.device_name(),
        "copy_bandwidth_gbs": copy_bandwidth_gbs(backend, int(args.copy_gib * 2**30), args.iters),
        "matmul_tflops": tflops,
        "launch_latency_us": launch_latency_us(backend, args.launch_kernels, args.iters),
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.json:
        with open(args.json, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
