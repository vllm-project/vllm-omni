#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Measure device copy bandwidth and dense matmul throughput for ceiling bounds."""

import argparse
import json

import torch


def _time_ms(fn, iters: int) -> float:
    for _ in range(3):
        fn()
    torch.accelerator.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.accelerator.synchronize()
    return start.elapsed_time(end) / iters


def copy_bandwidth_gbs(num_bytes: int, iters: int) -> float:
    src = torch.empty(num_bytes, dtype=torch.uint8, device="cuda")
    dst = torch.empty_like(src)
    ms = _time_ms(lambda: dst.copy_(src), iters)
    # A device copy reads and writes every byte.
    return 2 * num_bytes / (ms * 1e-3) / 1e9


def matmul_tflops(dtype: torch.dtype, n: int, iters: int, allow_tf32: bool) -> float:
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    try:
        a = torch.randn(n, n, device="cuda", dtype=dtype)
        b = torch.randn(n, n, device="cuda", dtype=dtype)
        ms = _time_ms(lambda: a @ b, iters)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    return 2 * n**3 / (ms * 1e-3) / 1e12


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--copy-gib", type=float, default=4.0)
    parser.add_argument("--matmul-n", type=int, default=8192)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--json", help="write results to this path")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required")

    result = {
        "device": torch.cuda.get_device_name(),
        "copy_bandwidth_gbs": copy_bandwidth_gbs(int(args.copy_gib * 2**30), args.iters),
        "matmul_tflops": {
            "bf16": matmul_tflops(torch.bfloat16, args.matmul_n, args.iters, allow_tf32=False),
            "fp16": matmul_tflops(torch.float16, args.matmul_n, args.iters, allow_tf32=False),
            "tf32": matmul_tflops(torch.float32, args.matmul_n, args.iters, allow_tf32=True),
            "fp32": matmul_tflops(torch.float32, args.matmul_n, args.iters, allow_tf32=False),
        },
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.json:
        with open(args.json, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
