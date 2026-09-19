# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Benchmark CosyVoice3 TensorRT cross-request CFG batching.

Uses one dual-profile TensorRT engine. Profile 0 executes the pre-batching
schedule (N serial CFG-batch-2 calls); profile 1 executes the same N requests as
one CFG-batch-2N call. Keeping both paths in one engine isolates batching from
engine/version differences and avoids duplicating TensorRT weights/contexts.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

import tensorrt as trt
import torch

from vllm_omni.model_executor.models.cosyvoice3.flow_estimator_trt import (
    TrtContextWrapper,
    build_flow_estimator_trt,
)
from vllm_omni.platforms import current_omni_platform


def _make_inputs(batch: int, length: int, seed: int, device: torch.device) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return {
        "x": 0.1 * torch.randn(batch, 80, length, device=device, dtype=torch.float16, generator=generator),
        "mask": torch.ones(batch, 1, length, device=device, dtype=torch.float16),
        "mu": 0.1 * torch.randn(batch, 80, length, device=device, dtype=torch.float16, generator=generator),
        "t": torch.rand(batch, device=device, dtype=torch.float16, generator=generator),
        "spks": 0.1 * torch.randn(batch, 80, device=device, dtype=torch.float16, generator=generator),
        "cond": 0.1 * torch.randn(batch, 80, length, device=device, dtype=torch.float16, generator=generator),
    }


def _select_rows(inputs: dict[str, torch.Tensor], rows: list[int]) -> dict[str, torch.Tensor]:
    index = torch.tensor(rows, device=inputs["x"].device, dtype=torch.long)
    return {name: tensor.index_select(0, index).contiguous() for name, tensor in inputs.items()}


def _run_estimator(wrapper: TrtContextWrapper, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    batch_size = int(inputs["x"].shape[0])
    [context, stream], engine = wrapper.acquire_estimator(batch_size)
    caller_stream = torch.cuda.current_stream(inputs["x"].device)
    try:
        stream.wait_stream(caller_stream)
        with torch.cuda.stream(stream):
            for name, tensor in inputs.items():
                if not context.set_input_shape(name, tuple(tensor.shape)):
                    raise RuntimeError(f"TensorRT rejected {name} shape {tuple(tensor.shape)}")
                context.set_tensor_address(name, tensor.data_ptr())
            output = torch.empty_like(inputs["x"])
            context.set_tensor_address("estimator_out", output.data_ptr())
            if not context.execute_async_v3(stream.cuda_stream):
                raise RuntimeError("TensorRT execute_async_v3 failed")
            for tensor in (*inputs.values(), output):
                tensor.record_stream(stream)
        caller_stream.wait_stream(stream)
        output.record_stream(caller_stream)
        return output
    finally:
        wrapper.release_estimator(context, stream)


def _time_cuda(fn) -> tuple[object, float]:
    torch.accelerator.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    end.synchronize()
    return result, float(start.elapsed_time(end))


def _serial_cfg2(
    wrapper: TrtContextWrapper,
    request_inputs: list[dict[str, torch.Tensor]],
) -> list[torch.Tensor]:
    return [_run_estimator(wrapper, inputs) for inputs in request_inputs]


def _reassemble_serial(outputs: list[torch.Tensor], requests: int) -> torch.Tensor:
    reference = torch.empty(
        2 * requests,
        *outputs[0].shape[1:],
        device=outputs[0].device,
        dtype=outputs[0].dtype,
    )
    for i, output in enumerate(outputs):
        reference[i].copy_(output[0])
        reference[i + requests].copy_(output[1])
    return reference


def _measure_interleaved_blocks(
    serial_fn,
    batched_fn,
    *,
    warmup: int,
    repeats: int,
    block_size: int,
) -> tuple[list[float], list[float], list[float]]:
    """Alternate short same-profile blocks to reduce long-run GPU drift.

    Profile switching happens once at each block boundary. The first call after
    a switch is intentionally untimed, so the reported samples represent the
    steady-state estimator calls made repeatedly inside an Euler solve.
    """
    for _ in range(warmup):
        serial_fn()
        batched_fn()
    torch.accelerator.synchronize()

    serial_ms: list[float] = []
    batched_ms: list[float] = []
    block_speedups: list[float] = []
    remaining = repeats
    block_index = 0
    while remaining > 0:
        count = min(block_size, remaining)
        order = (
            (("serial", serial_fn), ("batched", batched_fn))
            if block_index % 2 == 0
            else (("batched", batched_fn), ("serial", serial_fn))
        )
        block_samples: dict[str, list[float]] = {}
        for name, fn in order:
            fn()
            torch.accelerator.synchronize()
            samples = [_time_cuda(fn)[1] for _ in range(count)]
            block_samples[name] = samples
            if name == "serial":
                serial_ms.extend(samples)
            else:
                batched_ms.extend(samples)
        block_speedups.append(statistics.median(block_samples["serial"]) / statistics.median(block_samples["batched"]))
        remaining -= count
        block_index += 1

    return serial_ms, batched_ms, block_speedups


def _bootstrap_median_ci(
    values: list[float],
    *,
    samples: int = 5000,
    seed: int = 20260918,
) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(values)
    if n == 0:
        raise ValueError("cannot bootstrap an empty sample")
    medians = sorted(statistics.median(rng.choices(values, k=n)) for _ in range(samples))
    low_index = int(0.025 * (samples - 1))
    high_index = int(0.975 * (samples - 1))
    return medians[low_index], medians[high_index]


def run_case(
    *,
    wrapper: TrtContextWrapper,
    requests: int,
    length: int,
    warmup: int,
    repeats: int,
    block_size: int,
    device: torch.device,
) -> dict[str, float | int | bool | list[float]]:
    cfg_batch = 2 * requests
    inputs = _make_inputs(
        cfg_batch,
        length,
        seed=20260918 + requests * 1000 + length,
        device=device,
    )
    # Pre-split the serial baseline before timing. In production the pre-batching
    # baseline already owns separate per-request tensors, so charging index_select
    # only to serial execution would overstate the benefit of TensorRT batching.
    request_inputs = [_select_rows(inputs, [i, i + requests]) for i in range(requests)]

    def serial_call():
        return _serial_cfg2(wrapper, request_inputs)

    def batched_call():
        return _run_estimator(wrapper, inputs)

    # Parity is checked once outside the timed blocks. This switches profile at
    # most once and makes the difference attributable to TensorRT tactics/batch
    # shape rather than to different model inputs.
    serial_outputs = serial_call()
    batched_output = batched_call()
    torch.accelerator.synchronize()
    serial_reference = _reassemble_serial(serial_outputs, requests)
    difference = (batched_output - serial_reference).float()

    serial_ms, batched_ms, block_speedups = _measure_interleaved_blocks(
        serial_call,
        batched_call,
        warmup=warmup,
        repeats=repeats,
        block_size=block_size,
    )

    serial_median = statistics.median(serial_ms)
    batched_median = statistics.median(batched_ms)
    speedup_ci = _bootstrap_median_ci(
        block_speedups,
        seed=20260918 + requests * 1000 + length,
    )
    return {
        "requests": requests,
        "cfg_batch": cfg_batch,
        "length": length,
        "serial_cfg2_median_ms": serial_median,
        "batched_cfg2n_median_ms": batched_median,
        "throughput_speedup": serial_median / batched_median,
        "block_speedup_median": statistics.median(block_speedups),
        "block_speedup_ci95": [speedup_ci[0], speedup_ci[1]],
        "positive_blocks": sum(speedup > 1.0 for speedup in block_speedups),
        "blocks": len(block_speedups),
        "latency_reduction_percent": 100.0 * (serial_median - batched_median) / serial_median,
        "repeats": repeats,
        "block_size": block_size,
        "max_abs_error": float(difference.abs().max().item()),
        "mean_abs_error": float(difference.abs().mean().item()),
        "all_finite": bool(torch.isfinite(batched_output).all() and torch.isfinite(serial_reference).all()),
        "block_speedups": block_speedups,
        "serial_samples_ms": serial_ms,
        "batched_samples_ms": batched_ms,
    }


def run_single_request_regression(
    *,
    dynamic_wrapper: TrtContextWrapper,
    static_wrapper: TrtContextWrapper,
    length: int,
    warmup: int,
    repeats: int,
    device: torch.device,
) -> dict[str, float | int | bool | list[float]]:
    """Compare dual-profile CFG2 against the legacy fixed-batch engine."""
    inputs = _make_inputs(
        2,
        length,
        seed=20260920 + length,
        device=device,
    )

    def dynamic_call():
        return _run_estimator(dynamic_wrapper, inputs)

    def static_call():
        return _run_estimator(static_wrapper, inputs)

    for _ in range(warmup):
        static_call()
        dynamic_call()
    torch.accelerator.synchronize()

    static_ms: list[float] = []
    dynamic_ms: list[float] = []
    paired_delta_percent: list[float] = []
    for index in range(repeats):
        if index % 2 == 0:
            static_time = _time_cuda(static_call)[1]
            dynamic_time = _time_cuda(dynamic_call)[1]
        else:
            dynamic_time = _time_cuda(dynamic_call)[1]
            static_time = _time_cuda(static_call)[1]
        static_ms.append(static_time)
        dynamic_ms.append(dynamic_time)
        paired_delta_percent.append(100.0 * (dynamic_time - static_time) / static_time)

    static_output = static_call()
    dynamic_output = dynamic_call()
    torch.accelerator.synchronize()
    difference = (dynamic_output - static_output).float()
    ci = _bootstrap_median_ci(
        paired_delta_percent,
        seed=20260920 + length,
    )
    return {
        "requests": 1,
        "cfg_batch": 2,
        "length": length,
        "legacy_static_median_ms": statistics.median(static_ms),
        "dual_profile_cfg2_median_ms": statistics.median(dynamic_ms),
        "paired_dynamic_over_static_percent_median": statistics.median(paired_delta_percent),
        "paired_dynamic_over_static_percent_ci95": [ci[0], ci[1]],
        "dual_profile_faster_pairs": sum(dynamic < static for dynamic, static in zip(dynamic_ms, static_ms)),
        "repeats": repeats,
        "max_abs_error": float(difference.abs().max().item()),
        "mean_abs_error": float(difference.abs().mean().item()),
        "all_finite": bool(torch.isfinite(dynamic_output).all() and torch.isfinite(static_output).all()),
        "paired_delta_percent": paired_delta_percent,
        "legacy_static_samples_ms": static_ms,
        "dual_profile_cfg2_samples_ms": dynamic_ms,
    }


def _parse_int_list(value: str) -> list[int]:
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx_path", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lengths", type=_parse_int_list, default=[41, 191])
    parser.add_argument("--requests", type=_parse_int_list, default=[2, 4, 8])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=60)
    parser.add_argument("--block-size", type=int, default=10)
    parser.add_argument(
        "--compare-static-single",
        action="store_true",
        help="also compare dual-profile CFG2 against the legacy fixed-batch engine",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.requests) < 2:
        raise ValueError("--requests values must be >= 2 for cross-request batching")
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")
    if args.block_size < 1:
        raise ValueError("--block-size must be >= 1")

    device = torch.device(args.device)
    current_omni_platform.set_device(device)
    max_cfg_batch = 2 * max(args.requests)

    wrapper = build_flow_estimator_trt(
        str(args.onnx_path),
        device=device,
        max_cfg_batch=max_cfg_batch,
    )
    results = [
        run_case(
            wrapper=wrapper,
            requests=requests,
            length=length,
            warmup=args.warmup,
            repeats=args.repeats,
            block_size=args.block_size,
            device=device,
        )
        for length in args.lengths
        for requests in args.requests
    ]

    single_request_regression = []
    if args.compare_static_single:
        static_wrapper = build_flow_estimator_trt(
            str(args.onnx_path),
            device=device,
        )
        single_request_regression = [
            run_single_request_regression(
                dynamic_wrapper=wrapper,
                static_wrapper=static_wrapper,
                length=length,
                warmup=args.warmup,
                repeats=args.repeats,
                device=device,
            )
            for length in args.lengths
        ]

    payload = {
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "tensorrt": trt.__version__,
        "onnx_path": str(args.onnx_path),
        "max_cfg_batch": max_cfg_batch,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "block_size": args.block_size,
        "engine_device_memory_bytes": int(wrapper.trt_engine.device_memory_size_v2),
        "profile_device_memory_bytes": [
            int(wrapper.trt_engine.get_device_memory_size_for_profile_v2(index))
            for index in range(wrapper.trt_engine.num_optimization_profiles)
        ],
        "profile_x_shapes": [
            [list(shape) for shape in wrapper.trt_engine.get_tensor_profile_shape("x", index)]
            for index in range(wrapper.trt_engine.num_optimization_profiles)
        ],
        "single_request_regression": single_request_regression,
        "results": results,
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
