# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Checkpoint-free A/B benchmark of the production NPU FIR resamplers.

Inputs are synthetic, not captured model workloads. Timing includes padding,
convolution, chunk slicing/concatenation, dtype conversions, and output cropping.
Each case is isolated in a subprocess because a native NPU convolution failure
can leave the device context unusable. No end-to-end model speedup is implied.
"""

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

ENV = "VLLM_OMNI_NPU_ANTIALIAS_MAX_CONV_LENGTH"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--channels", type=int, nargs="+", default=[32, 128])
    parser.add_argument("--lengths", type=int, nargs="+", default=[512, 6144, 8192, 12288, 49152])
    parser.add_argument("--max-conv-length", type=int, default=6144)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case", type=str, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.batch, *args.channels, *args.lengths, args.iterations, args.repeats, args.timeout) < 1:
        parser.error("batch, shapes, iterations, repeats and timeout must be positive")
    if args.warmup < 0 or args.max_conv_length < 12:
        parser.error("warmup must be nonnegative; max-conv-length must be >= 12")
    return args


def run_case(args):
    import torch
    import torch_npu
    from vllm_omni.model_executor.models.common import alias_free_activation as fir

    if not fir.current_omni_platform.is_npu():
        raise RuntimeError("This benchmark exercises the production NPU branch and requires an NPU platform")
    device = torch.device(args.device)
    if device.type != "npu":
        raise ValueError("--device must select an NPU")
    torch.npu.set_device(device)
    torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.conv.allow_hf32 = False
    case = json.loads(args.case)
    dtype = getattr(torch, args.dtype)
    generator = torch.Generator(device="cpu").manual_seed(42)
    x_cpu = torch.randn(args.batch, case["channels"], case["length"], generator=generator, device="cpu").to(dtype)
    x = x_cpu.to(device)
    os.environ[ENV] = str(case["cap"])
    with torch.device(device):
        module = fir.UpSample1d().to(dtype=dtype).eval()
    assert module._max_conv_length == case["cap"]
    weight_cpu = module.filter.cpu().expand(case["channels"], -1, -1)
    padded = torch.nn.functional.pad(x_cpu.float(), (module.pad, module.pad), mode="replicate").to(dtype)
    reference = module.ratio * torch.nn.functional.conv_transpose1d(
        padded, weight_cpu, stride=module.stride, groups=case["channels"]
    ).to(dtype)
    reference = reference[..., module.pad_left : -module.pad_right]
    tolerance = {torch.float32: (1e-4, 1e-5), torch.float16: (2e-3, 2e-3), torch.bfloat16: (2e-2, 2e-2)}
    times = []
    with torch.inference_mode():
        actual = module(x)
        actual_cpu = actual.cpu()
        rtol, atol = tolerance[dtype]
        torch.testing.assert_close(actual_cpu, reference, rtol=rtol, atol=atol)
        del actual
        for _ in range(args.warmup):
            module(x)
        for _ in range(args.repeats):
            torch.npu.synchronize(device)
            start = time.perf_counter()
            for _ in range(args.iterations):
                module(x)
            torch.npu.synchronize(device)
            times.append((time.perf_counter() - start) * 1e6 / args.iterations)
    error = (actual_cpu.float() - reference.float()).abs()
    result = {
        **case,
        "status": "ok",
        "batch": args.batch,
        "dtype": args.dtype,
        "median_us": statistics.median(times),
        "repeat_us": times,
        "max_abs_error": error.max().item(),
        "mean_abs_error": error.mean().item(),
        "output_shape": list(actual_cpu.shape),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "device": torch.npu.get_device_name(device),
            "source_module": fir.__name__,
            "jit_compile": False,
            "conv_allow_hf32": torch.npu.conv.allow_hf32,
            "allow_internal_format": getattr(torch.npu.config, "allow_internal_format", "unavailable"),
            "cann_version": os.environ.get("CANN_VERSION", "record installed CANN version separately"),
        },
    }
    print("RESULT " + json.dumps(result), flush=True)


def main():
    args = parse_args()
    if args.case:
        run_case(args)
        return
    results = []
    report = {
        "input_source": "synthetic",
        "config": {key: value for key, value in vars(args).items() if key not in {"output", "case"}},
        "results": results,
    }
    print("C L whole_us chunked_us speedup max_abs_error", flush=True)
    for channels in args.channels:
        for length in args.lengths:
            pair = {}
            caps = [0, args.max_conv_length]
            if len(results) % 4:
                caps.reverse()
            for cap in caps:
                case = {"channels": channels, "length": length, "cap": cap}
                command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--case", json.dumps(case)]
                try:
                    completed = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout)
                    records = [line[7:] for line in completed.stdout.splitlines() if line.startswith("RESULT ")]
                    if completed.returncode or not records:
                        result = {
                            **case,
                            "status": "failed",
                            "returncode": completed.returncode,
                            "error": "Child failed or returned no result; rerun this case locally for diagnostics.",
                        }
                    else:
                        result = json.loads(records[-1])
                except subprocess.TimeoutExpired:
                    result = {**case, "status": "failed", "error": f"timeout after {args.timeout}s"}
                results.append(result)
                pair[cap] = result
            whole, chunked = pair[0], pair[args.max_conv_length]
            if whole["status"] == chunked["status"] == "ok":
                print(
                    f"{channels} {length} {whole['median_us']:.1f} {chunked['median_us']:.1f} "
                    f"{whole['median_us'] / chunked['median_us']:.2f}x {chunked['max_abs_error']:.6g}",
                    flush=True,
                )
            else:
                print(f"{channels} {length} whole={whole['status']} chunked={chunked['status']}", flush=True)
            if args.output:
                args.output.write_text(json.dumps(report, indent=2) + "\n")
    if any(result["status"] != "ok" for result in results):
        print(
            "Some cases failed; inspect the JSON report. No speedup is reported for failed baselines.", file=sys.stderr
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
