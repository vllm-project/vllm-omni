# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Profile MiniCPM-o 4.5 SigLIP encoder padding and layout overhead.

The benchmark runs the real 27-layer vision encoder architecture with synthetic
hidden states. Random weights are sufficient because the measured tensor shapes,
FlashAttention calls, padding helpers, and CUDA kernels match the production path.

Example:
    python benchmarks/minicpmo/profile_vision_encoder.py \
        --batch-size 2 \
        --seq-len 4900 \
        --valid-lengths 4900,2401 \
        --warmup 3 \
        --iterations 10
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
import logging
import math
import statistics
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function

from vllm_omni.model_executor.models.minicpmo_4_5 import minicpmo_4_5_omni_llm as minicpmo_model

logger = logging.getLogger(__name__)

_DEFAULT_HIDDEN_SIZE = 1152
_DEFAULT_INTERMEDIATE_SIZE = 4304
_DEFAULT_NUM_HEADS = 16
_DEFAULT_NUM_LAYERS = 27
_DEFAULT_IMAGE_SIZE = 980
_DEFAULT_PATCH_SIZE = 14
_DEFAULT_SEQ_LEN = (_DEFAULT_IMAGE_SIZE // _DEFAULT_PATCH_SIZE) ** 2
_SUPPORTED_CASES = ("dense", "all_valid_mask", "padded")

_PROFILE_RANGES = {
    "_get_unpad_data": "minicpmo.flash_attn.unpad_metadata",
    "index_first_axis": "minicpmo.flash_attn.index_first_axis",
    "unpad_input": "minicpmo.flash_attn.unpad_input",
    "pad_input": "minicpmo.flash_attn.pad_input",
    "flash_attn_func": "minicpmo.flash_attn.dense",
    "flash_attn_varlen_func": "minicpmo.flash_attn.varlen",
}

_PROFILE_CATEGORIES = {
    "unpad_metadata": ("minicpmo.flash_attn.unpad_metadata",),
    "index_first_axis": ("minicpmo.flash_attn.index_first_axis",),
    "unpad_input": ("minicpmo.flash_attn.unpad_input",),
    "pad_input": ("minicpmo.flash_attn.pad_input",),
    "dense_flash_attention": ("minicpmo.flash_attn.dense",),
    "varlen_flash_attention": ("minicpmo.flash_attn.varlen",),
    "contiguous": ("aten::contiguous",),
    "materialization": ("aten::clone", "aten::copy_", "aten::_to_copy"),
    "concat": ("aten::cat",),
    "layout_views": ("aten::view", "aten::reshape", "aten::transpose", "aten::permute"),
    "kernel_launch": ("cudalaunchkernel",),
    "device_copy": ("cudamemcpy", "memcpy"),
    "device_sync": ("cudadevicesynchronize",),
}


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    attention_mask: torch.Tensor | None
    valid_lengths: tuple[int, ...]
    description: str


@dataclass(frozen=True)
class LatencyStats:
    samples_ms: tuple[float, ...]
    mean_ms: float
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    stdev_ms: float


@dataclass
class InstrumentationState:
    enabled: bool = False
    record_cuda_events: bool = False
    record_ranges: bool = False
    layer_events: dict[int, list[tuple[Any, Any]]] = field(default_factory=dict)

    def clear_layer_events(self) -> None:
        self.layer_events.clear()


class ProfiledEncoderLayer(nn.Module):
    def __init__(self, layer: nn.Module, layer_index: int, state: InstrumentationState) -> None:
        super().__init__()
        self.layer = layer
        self.layer_index = layer_index
        self.state = state

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if not self.state.enabled:
            return self.layer(*args, **kwargs)

        start_event = None
        end_event = None
        if self.state.record_cuda_events:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

        range_context = (
            record_function(f"minicpmo.vision.layer.{self.layer_index:02d}")
            if self.state.record_ranges
            else nullcontext()
        )
        with range_context:
            output = self.layer(*args, **kwargs)

        if start_event is not None and end_event is not None:
            end_event.record()
            self.state.layer_events.setdefault(self.layer_index, []).append((start_event, end_event))
        return output


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a percentile for an empty sequence")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _latency_stats(samples_ms: Sequence[float]) -> LatencyStats:
    if not samples_ms:
        raise ValueError("At least one latency sample is required")
    samples = tuple(samples_ms)
    return LatencyStats(
        samples_ms=samples,
        mean_ms=statistics.fmean(samples),
        median_ms=statistics.median(samples),
        p95_ms=_percentile(samples, 0.95),
        min_ms=min(samples),
        max_ms=max(samples),
        stdev_ms=statistics.pstdev(samples),
    )


def _stats_to_dict(stats: LatencyStats) -> dict[str, Any]:
    return {
        "samples_ms": list(stats.samples_ms),
        "mean_ms": stats.mean_ms,
        "median_ms": stats.median_ms,
        "p95_ms": stats.p95_ms,
        "min_ms": stats.min_ms,
        "max_ms": stats.max_ms,
        "stdev_ms": stats.stdev_ms,
    }


def _parse_dtype(value: str) -> torch.dtype:
    dtypes = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    try:
        return dtypes[value]
    except KeyError as exc:
        choices = ", ".join(sorted(dtypes))
        raise argparse.ArgumentTypeError(f"Unsupported dtype {value!r}; choose from {choices}") from exc


def _parse_cases(value: str) -> tuple[str, ...]:
    cases = tuple(item.strip() for item in value.split(",") if item.strip())
    invalid = sorted(set(cases) - set(_SUPPORTED_CASES))
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Unsupported cases: {', '.join(invalid)}; choose from {', '.join(_SUPPORTED_CASES)}"
        )
    if not cases:
        raise argparse.ArgumentTypeError("At least one benchmark case is required")
    return cases


def _default_valid_lengths(batch_size: int, seq_len: int) -> tuple[int, ...]:
    if batch_size == 1:
        return (seq_len,)
    max_side = max(1, round(math.sqrt(seq_len)))
    min_side = max(1, round(max_side * 0.7))
    lengths = []
    for index in range(batch_size):
        ratio = index / (batch_size - 1)
        side = round(max_side - ratio * (max_side - min_side))
        lengths.append(min(seq_len, max(1, side**2)))
    lengths[0] = seq_len
    return tuple(lengths)


def _parse_valid_lengths(value: str | None, batch_size: int, seq_len: int) -> tuple[int, ...]:
    if value is None:
        return _default_valid_lengths(batch_size, seq_len)
    try:
        lengths = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("--valid-lengths must be a comma-separated list of integers") from exc
    if len(lengths) != batch_size:
        raise ValueError(f"Expected {batch_size} valid lengths, received {len(lengths)}")
    if any(length < 1 or length > seq_len for length in lengths):
        raise ValueError(f"Each valid length must be in [1, {seq_len}]")
    return lengths


def _require_cuda_flash_attention() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the MiniCPM-o vision encoder profiler")
    missing = [name for name in _PROFILE_RANGES if not callable(getattr(minicpmo_model, name, None))]
    if missing:
        raise RuntimeError(
            "Upstream flash-attn is unavailable or incomplete; missing symbols: " + ", ".join(sorted(missing))
        )
    try:
        import flash_attn
    except ImportError as exc:
        raise RuntimeError("Install upstream flash-attn before running this benchmark") from exc
    return getattr(flash_attn, "__version__", "unknown")


def _build_encoder(args: argparse.Namespace, dtype: torch.dtype, state: InstrumentationState) -> nn.Module:
    config = minicpmo_model.SiglipVisionConfig(
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.layers,
        num_attention_heads=args.num_heads,
        image_size=args.image_size,
        patch_size=args.patch_size,
        attention_dropout=0.0,
    )
    config._attn_implementation = "flash_attention_2"

    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device(args.device):
            encoder = minicpmo_model.SiglipEncoder(config)
    finally:
        torch.set_default_dtype(previous_dtype)

    encoder.eval()
    for index, layer in enumerate(encoder.layers):
        encoder.layers[index] = ProfiledEncoderLayer(layer, index, state)
    return encoder


def _build_cases(
    case_names: Sequence[str],
    batch_size: int,
    seq_len: int,
    valid_lengths: tuple[int, ...],
    device: str,
) -> list[BenchmarkCase]:
    all_valid_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)
    positions = torch.arange(seq_len, device=device).unsqueeze(0)
    lengths_tensor = torch.tensor(valid_lengths, device=device).unsqueeze(1)
    padded_mask = positions < lengths_tensor

    definitions = {
        "dense": BenchmarkCase(
            name="dense",
            attention_mask=None,
            valid_lengths=tuple(seq_len for _ in range(batch_size)),
            description="Dense FlashAttention with no padding mask.",
        ),
        "all_valid_mask": BenchmarkCase(
            name="all_valid_mask",
            attention_mask=all_valid_mask,
            valid_lengths=tuple(seq_len for _ in range(batch_size)),
            description="All tokens valid, but force unpad -> varlen FlashAttention -> pad.",
        ),
        "padded": BenchmarkCase(
            name="padded",
            attention_mask=padded_mask,
            valid_lengths=valid_lengths,
            description="Heterogeneous valid lengths using the production padding path.",
        ),
    }
    return [definitions[name] for name in case_names]


def _run_encoder(encoder: nn.Module, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    return encoder(
        inputs_embeds=hidden_states,
        attention_mask=attention_mask,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    )[0]


def _memory_allocated() -> int:
    memory_allocated = getattr(torch.accelerator, "memory_allocated", None)
    return int(memory_allocated()) if callable(memory_allocated) else 0


def _benchmark_latency(
    encoder: nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    warmup: int,
    iterations: int,
) -> tuple[LatencyStats, int, int]:
    with torch.inference_mode():
        for _ in range(warmup):
            output = _run_encoder(encoder, hidden_states, attention_mask)
        torch.accelerator.synchronize()
        del output

        baseline_memory_bytes = _memory_allocated()
        torch.accelerator.reset_peak_memory_stats()
        event_pairs = []
        for _ in range(iterations):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            output = _run_encoder(encoder, hidden_states, attention_mask)
            end_event.record()
            event_pairs.append((start_event, end_event))
            del output
        torch.accelerator.synchronize()

    samples_ms = [start.elapsed_time(end) for start, end in event_pairs]
    peak_memory_bytes = torch.accelerator.max_memory_allocated()
    return _latency_stats(samples_ms), baseline_memory_bytes, peak_memory_bytes


def _measure_layer_times(
    encoder: nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    state: InstrumentationState,
    iterations: int,
) -> list[dict[str, Any]]:
    state.clear_layer_events()
    state.enabled = True
    state.record_cuda_events = True
    state.record_ranges = False
    try:
        with torch.inference_mode():
            for _ in range(iterations):
                output = _run_encoder(encoder, hidden_states, attention_mask)
                del output
            torch.accelerator.synchronize()
    finally:
        state.enabled = False
        state.record_cuda_events = False

    rows = []
    for layer_index, event_pairs in sorted(state.layer_events.items()):
        stats = _latency_stats([start.elapsed_time(end) for start, end in event_pairs])
        rows.append({"layer": layer_index, **_stats_to_dict(stats)})
    return rows


def _profile_event_value(event: Any, names: Sequence[str]) -> float:
    for name in names:
        value = getattr(event, name, None)
        if isinstance(value, int | float):
            return float(value)
    return 0.0


def _profile_event_to_dict(event: Any) -> dict[str, Any]:
    return {
        "key": str(event.key),
        "count": int(event.count),
        "self_cpu_time_us": _profile_event_value(event, ("self_cpu_time_total",)),
        "cpu_time_us": _profile_event_value(event, ("cpu_time_total",)),
        "self_device_time_us": _profile_event_value(event, ("self_cuda_time_total", "self_device_time_total")),
        "device_time_us": _profile_event_value(event, ("cuda_time_total", "device_time_total")),
        "self_cpu_memory_bytes": _profile_event_value(event, ("self_cpu_memory_usage",)),
        "self_device_memory_bytes": _profile_event_value(
            event,
            ("self_cuda_memory_usage", "self_device_memory_usage"),
        ),
    }


def _aggregate_profile_categories(events: Sequence[Any]) -> dict[str, dict[str, float | int]]:
    categories: dict[str, dict[str, float | int]] = {}
    for category, patterns in _PROFILE_CATEGORIES.items():
        aggregate: dict[str, float | int] = {
            "count": 0,
            "self_cpu_time_us": 0.0,
            "cpu_time_us": 0.0,
            "self_device_time_us": 0.0,
            "device_time_us": 0.0,
            "self_cpu_memory_bytes": 0.0,
            "self_device_memory_bytes": 0.0,
        }
        for event in events:
            key = str(event.key).lower()
            if not any(pattern.lower() in key for pattern in patterns):
                continue
            row = _profile_event_to_dict(event)
            aggregate["count"] += row["count"]
            for name in (
                "self_cpu_time_us",
                "cpu_time_us",
                "self_device_time_us",
                "device_time_us",
                "self_cpu_memory_bytes",
                "self_device_memory_bytes",
            ):
                aggregate[name] += row[name]
        categories[category] = aggregate
    return categories


def _install_function_ranges(state: InstrumentationState) -> dict[str, Callable[..., Any]]:
    originals = {}
    for attribute, range_name in _PROFILE_RANGES.items():
        original = getattr(minicpmo_model, attribute)
        originals[attribute] = original

        @functools.wraps(original)
        def wrapped(
            *args: Any,
            __original: Callable[..., Any] = original,
            __range_name: str = range_name,
            **kwargs: Any,
        ) -> Any:
            if not state.enabled or not state.record_ranges:
                return __original(*args, **kwargs)
            with record_function(__range_name):
                return __original(*args, **kwargs)

        setattr(minicpmo_model, attribute, wrapped)
    return originals


def _restore_functions(originals: dict[str, Callable[..., Any]]) -> None:
    for attribute, original in originals.items():
        setattr(minicpmo_model, attribute, original)


def _profile_case(
    encoder: nn.Module,
    hidden_states: torch.Tensor,
    benchmark_case: BenchmarkCase,
    state: InstrumentationState,
    output_dir: Path,
    profile_iterations: int,
    top_k: int,
    profile_memory: bool,
    record_shapes: bool,
    with_stack: bool,
) -> dict[str, Any]:
    state.enabled = True
    state.record_cuda_events = False
    state.record_ranges = True
    try:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            profile_memory=profile_memory,
            record_shapes=record_shapes,
            with_stack=with_stack,
        ) as profiler:
            with torch.inference_mode():
                for _ in range(profile_iterations):
                    with record_function(f"minicpmo.vision.case.{benchmark_case.name}"):
                        output = _run_encoder(encoder, hidden_states, benchmark_case.attention_mask)
                    del output
                torch.accelerator.synchronize()
    finally:
        state.enabled = False
        state.record_ranges = False

    trace_path = output_dir / f"trace_{benchmark_case.name}.json"
    profiler.export_chrome_trace(str(trace_path))
    events = list(profiler.key_averages(group_by_input_shape=record_shapes))
    event_rows = [_profile_event_to_dict(event) for event in events]
    event_rows.sort(key=lambda row: row["self_device_time_us"], reverse=True)

    try:
        table = profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=top_k)
    except (AttributeError, KeyError, RuntimeError):
        table = profiler.key_averages().table(sort_by="self_device_time_total", row_limit=top_k)
    logger.info("Top profiler events for %s:\n%s", benchmark_case.name, table)
    return {
        "trace_path": str(trace_path),
        "categories": _aggregate_profile_categories(events),
        "top_events": event_rows[:top_k],
    }


def _write_layer_csv(path: Path, cases: dict[str, dict[str, Any]]) -> None:
    fields = ("case", "layer", "mean_ms", "median_ms", "p95_ms", "min_ms", "max_ms", "stdev_ms")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case_name, case_result in cases.items():
            for layer in case_result["layers"]:
                writer.writerow({name: case_name if name == "case" else layer[name] for name in fields})


def _build_comparisons(cases: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
    if "dense" not in cases:
        return {}
    dense_ms = cases["dense"]["latency"]["mean_ms"]
    comparisons = {}
    for case_name in ("all_valid_mask", "padded"):
        if case_name not in cases:
            continue
        case_ms = cases[case_name]["latency"]["mean_ms"]
        delta_ms = case_ms - dense_ms
        comparisons[f"{case_name}_vs_dense"] = {
            "delta_ms": delta_ms,
            "delta_percent": delta_ms / dense_ms * 100.0,
            "speedup": dense_ms / case_ms,
        }
    return comparisons


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", type=_parse_dtype, default=torch.bfloat16, metavar="{bfloat16,float16}")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=_DEFAULT_SEQ_LEN)
    parser.add_argument(
        "--valid-lengths",
        help="Comma-separated valid token counts for the padded case; defaults to square-like decreasing lengths.",
    )
    parser.add_argument("--cases", type=_parse_cases, default=_SUPPORTED_CASES)
    parser.add_argument("--layers", type=int, default=_DEFAULT_NUM_LAYERS)
    parser.add_argument("--hidden-size", type=int, default=_DEFAULT_HIDDEN_SIZE)
    parser.add_argument("--intermediate-size", type=int, default=_DEFAULT_INTERMEDIATE_SIZE)
    parser.add_argument("--num-heads", type=int, default=_DEFAULT_NUM_HEADS)
    parser.add_argument("--image-size", type=int, default=_DEFAULT_IMAGE_SIZE)
    parser.add_argument("--patch-size", type=int, default=_DEFAULT_PATCH_SIZE)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--layer-iterations", type=int, default=3)
    parser.add_argument("--profile-iterations", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--profile-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--record-shapes", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--with-stack", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results/minicpmo_vision_encoder"),
    )
    args = parser.parse_args()

    positive_fields = (
        "batch_size",
        "seq_len",
        "layers",
        "hidden_size",
        "intermediate_size",
        "num_heads",
        "warmup",
        "iterations",
        "layer_iterations",
        "profile_iterations",
        "top_k",
    )
    for field_name in positive_fields:
        if getattr(args, field_name) < 1:
            parser.error(f"--{field_name.replace('_', '-')} must be at least 1")
    if args.hidden_size % args.num_heads != 0:
        parser.error("--hidden-size must be divisible by --num-heads")
    try:
        args.valid_lengths = _parse_valid_lengths(args.valid_lengths, args.batch_size, args.seq_len)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    flash_attn_version = _require_cuda_flash_attention()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gpu_index = torch.device(args.device).index or 0
    torch.cuda.set_device(gpu_index)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    state = InstrumentationState()
    originals = _install_function_ranges(state)
    try:
        encoder = _build_encoder(args, args.dtype, state)
        hidden_states = torch.randn(
            args.batch_size,
            args.seq_len,
            args.hidden_size,
            device=args.device,
            dtype=args.dtype,
        )
        cases = _build_cases(
            args.cases,
            args.batch_size,
            args.seq_len,
            args.valid_lengths,
            args.device,
        )

        capability = torch.cuda.get_device_capability(gpu_index)
        parameter_count = sum(parameter.numel() for parameter in encoder.parameters())
        summary: dict[str, Any] = {
            "environment": {
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "flash_attn_version": flash_attn_version,
                "gpu": torch.cuda.get_device_name(gpu_index),
                "compute_capability": list(capability),
            },
            "config": {
                "device": args.device,
                "dtype": str(args.dtype),
                "batch_size": args.batch_size,
                "seq_len": args.seq_len,
                "valid_lengths": list(args.valid_lengths),
                "layers": args.layers,
                "hidden_size": args.hidden_size,
                "intermediate_size": args.intermediate_size,
                "num_heads": args.num_heads,
                "head_dim": args.hidden_size // args.num_heads,
                "image_size": args.image_size,
                "patch_size": args.patch_size,
                "parameter_count": parameter_count,
                "warmup": args.warmup,
                "iterations": args.iterations,
                "layer_iterations": args.layer_iterations,
                "profile_iterations": args.profile_iterations,
            },
            "cases": {},
        }

        logger.info(
            "Profiling %d layers, batch=%d, seq=%d, dtype=%s on %s",
            args.layers,
            args.batch_size,
            args.seq_len,
            args.dtype,
            summary["environment"]["gpu"],
        )
        for benchmark_case in cases:
            logger.info("Running case %s: %s", benchmark_case.name, benchmark_case.description)
            latency, baseline_memory_bytes, peak_memory_bytes = _benchmark_latency(
                encoder,
                hidden_states,
                benchmark_case.attention_mask,
                args.warmup,
                args.iterations,
            )
            layers = _measure_layer_times(
                encoder,
                hidden_states,
                benchmark_case.attention_mask,
                state,
                args.layer_iterations,
            )
            profiler_result = _profile_case(
                encoder,
                hidden_states,
                benchmark_case,
                state,
                args.output_dir,
                args.profile_iterations,
                args.top_k,
                args.profile_memory,
                args.record_shapes,
                args.with_stack,
            )
            summary["cases"][benchmark_case.name] = {
                "description": benchmark_case.description,
                "valid_lengths": list(benchmark_case.valid_lengths),
                "valid_tokens": sum(benchmark_case.valid_lengths),
                "total_tokens": args.batch_size * args.seq_len,
                "padding_fraction": 1.0 - sum(benchmark_case.valid_lengths) / (args.batch_size * args.seq_len),
                "latency": _stats_to_dict(latency),
                "baseline_memory_bytes": baseline_memory_bytes,
                "peak_memory_bytes": peak_memory_bytes,
                "peak_memory_increment_bytes": max(0, peak_memory_bytes - baseline_memory_bytes),
                "layers": layers,
                "profiler": profiler_result,
            }
            logger.info(
                "%s latency: mean=%.3f ms, p95=%.3f ms, peak_increment=%.2f MiB",
                benchmark_case.name,
                latency.mean_ms,
                latency.p95_ms,
                max(0, peak_memory_bytes - baseline_memory_bytes) / 1024**2,
            )

        summary["comparisons"] = _build_comparisons(summary["cases"])
        summary_path = args.output_dir / "summary.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        layer_csv_path = args.output_dir / "layer_times.csv"
        _write_layer_csv(layer_csv_path, summary["cases"])

        for comparison, values in summary["comparisons"].items():
            logger.info(
                "%s: delta=%+.3f ms (%+.2f%%), speedup=%.3fx",
                comparison,
                values["delta_ms"],
                values["delta_percent"],
                values["speedup"],
            )
        logger.info("Summary: %s", summary_path)
        logger.info("Per-layer CSV: %s", layer_csv_path)
    finally:
        _restore_functions(originals)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
