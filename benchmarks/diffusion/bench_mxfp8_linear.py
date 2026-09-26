# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compare CUDA SwiGLU + MXFP8 GEMM with its fused activation producer."""

import argparse
import json
import statistics

import torch

from vllm_omni.diffusion.layers.activation import SiluAndMul
from vllm_omni.diffusion.layers.mxfp8 import mxfp8_linear, mxfp8_quantize_swizzled, silu_mxfp8_linear


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=14336)
    parser.add_argument("--output", type=int, default=5376)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    torch.manual_seed(7415)
    device = torch.get_device_module()
    x = torch.randn(args.rows, args.hidden * 2, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(args.output, args.hidden, device="cuda", dtype=torch.bfloat16) * 0.01
    q, scale = mxfp8_quantize_swizzled(weight)
    activation = SiluAndMul()
    paths = {
        "unfused": lambda: mxfp8_linear(activation(x), q, scale),
        "fused": lambda: silu_mxfp8_linear(x, q, scale),
    }
    with torch.inference_mode():
        assert torch.equal(paths["unfused"](), paths["fused"]())
        samples = {}
        for name, fn in paths.items():
            for _ in range(5):
                fn()
            device.synchronize()
            measurements = []
            for _ in range(args.repeats):
                start, end = device.Event(enable_timing=True), device.Event(enable_timing=True)
                start.record()
                fn()
                end.record()
                end.synchronize()
                measurements.append(start.elapsed_time(end))
            samples[name] = {
                "median_ms": statistics.median(measurements),
                "min_ms": min(measurements),
                "max_ms": max(measurements),
                "samples_ms": measurements,
            }
    print(
        json.dumps(
            {
                "shape": vars(args),
                "torch": torch.__version__,
                "warmup": 5,
                "gpu": device.get_device_name(),
                "byte_equal": True,
                "results": samples,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
