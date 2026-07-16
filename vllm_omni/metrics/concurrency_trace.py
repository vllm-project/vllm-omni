"""Opt-in JSONL observability for staged concurrency experiments.

The normal vLLM logs and Prometheus endpoint remain the production source of
truth. This module records a compact, request-correlatable trace only when
``VLLM_OMNI_CONCURRENCY_TRACE_PATH`` is set. It is intentionally dependency
light so model workers can emit events without importing NVML or HTTP clients.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from statistics import fmean
from typing import Any
from urllib.request import urlopen

TRACE_PATH_ENV = "VLLM_OMNI_CONCURRENCY_TRACE_PATH"
TRACE_RUN_ID_ENV = "VLLM_OMNI_CONCURRENCY_TRACE_RUN_ID"
_WRITE_LOCK = threading.Lock()


def trace_path() -> Path | None:
    raw_path = os.environ.get(TRACE_PATH_ENV)
    return Path(raw_path).expanduser() if raw_path else None


def emit_concurrency_trace(event: str, /, **fields: Any) -> None:
    """Append one JSONL event when experiment tracing is enabled.

    The caller must pass scalar or JSON-compatible values. Failures to write a
    diagnostic trace must never affect inference, so filesystem errors are
    intentionally swallowed.
    """
    path = trace_path()
    if path is None:
        return

    record = {
        "event": event,
        "pid": os.getpid(),
        "run_id": os.environ.get(TRACE_RUN_ID_ENV),
        "ts_unix_s": time.time(),
        "ts_monotonic_s": time.perf_counter(),
        **fields,
    }
    try:
        payload = (json.dumps(record, sort_keys=True, default=str) + "\n").encode("utf-8")
        path.parent.mkdir(parents=True, exist_ok=True)
        with _WRITE_LOCK:
            fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
            try:
                os.write(fd, payload)
            finally:
                os.close(fd)
    except OSError:
        # Trace collection is observability only; do not turn an unavailable
        # filesystem into a serving failure.
        return


def emit_stage_config_snapshot(stage_config: Mapping[str, Any], source: str) -> None:
    """Record the independently configured Stage 0/1 concurrency settings."""
    stages = stage_config.get("stages", [])
    if not isinstance(stages, list):
        raise ValueError("stage config must contain a list-valued 'stages' key")
    for stage in stages:
        if not isinstance(stage, Mapping):
            continue
        emit_concurrency_trace(
            "stage_config",
            source=source,
            stage_id=stage.get("stage_id"),
            devices=stage.get("devices"),
            max_num_seqs=stage.get("max_num_seqs"),
            max_model_len=stage.get("max_model_len"),
            max_num_batched_tokens=stage.get("max_num_batched_tokens"),
            gpu_memory_utilization=stage.get("gpu_memory_utilization"),
            tensor_parallel_size=stage.get("tensor_parallel_size", 1),
        )


_PROMETHEUS_LINE = re.compile(r"^(?P<name>[^\s{]+)(?P<labels>\{[^}]*\})?\s+(?P<value>[-+0-9.eE]+)$")


def _parse_labels(raw_labels: str | None) -> dict[str, str]:
    if not raw_labels:
        return {}
    return {
        match.group("key"): match.group("value")
        for match in re.finditer(r'(?P<key>[a-zA-Z_][a-zA-Z0-9_]*)="(?P<value>(?:\\.|[^\"])*)"', raw_labels)
    }


def _iter_prometheus_samples(payload: str) -> Iterable[tuple[str, dict[str, str], float]]:
    for line in payload.splitlines():
        if not line or line.startswith("#"):
            continue
        match = _PROMETHEUS_LINE.match(line)
        if match is None:
            continue
        name = match.group("name")
        if not (name.startswith("vllm:") or name.startswith("vllm_omni:")):
            continue
        yield name, _parse_labels(match.group("labels")), float(match.group("value"))


def sample_once(devices: Sequence[int], metrics_url: str | None, timeout_s: float) -> None:
    """Write one GPU and Prometheus snapshot to the active trace."""
    try:
        import pynvml

        pynvml.nvmlInit()
        for device in devices:
            handle = pynvml.nvmlDeviceGetHandleByIndex(int(device))
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            emit_concurrency_trace(
                "gpu_sample",
                gpu_index=int(device),
                gpu_utilization_pct=int(utilization.gpu),
                memory_utilization_pct=int(utilization.memory),
                memory_used_bytes=int(memory.used),
                memory_total_bytes=int(memory.total),
            )
    except Exception as exc:
        emit_concurrency_trace("gpu_sample_error", error_type=type(exc).__name__, message=str(exc))

    if not metrics_url:
        return
    try:
        with urlopen(metrics_url, timeout=timeout_s) as response:  # noqa: S310 - user supplied local endpoint
            payload = response.read().decode("utf-8")
        for name, labels, value in _iter_prometheus_samples(payload):
            emit_concurrency_trace("prometheus_sample", metric=name, labels=labels, value=value)
    except Exception as exc:
        emit_concurrency_trace("prometheus_sample_error", error_type=type(exc).__name__, message=str(exc))


def run_sampler(devices: Sequence[int], metrics_url: str | None, interval_s: float, timeout_s: float) -> None:
    while True:
        sample_once(devices, metrics_url, timeout_s)
        time.sleep(interval_s)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * percentile / 100.0
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = index - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _overlap_seconds(left: list[tuple[float, float]], right: list[tuple[float, float]]) -> float:
    overlap = 0.0
    left_index = right_index = 0
    while left_index < len(left) and right_index < len(right):
        left_start, left_end = left[left_index]
        right_start, right_end = right[right_index]
        overlap += max(0.0, min(left_end, right_end) - max(left_start, right_start))
        if left_end <= right_end:
            left_index += 1
        else:
            right_index += 1
    return overlap


def _device_indices(devices: object) -> list[int]:
    if isinstance(devices, str):
        raw_devices = re.split(r"[,\s]+", devices.strip())
    elif isinstance(devices, Sequence) and not isinstance(devices, (bytes, str)):
        raw_devices = devices
    else:
        return []

    indices: list[int] = []
    for device in raw_devices:
        try:
            indices.append(int(device))
        except (TypeError, ValueError):
            continue
    return indices


def build_summary(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate the trace fields needed by the concurrency RFC experiment."""
    records = list(records)
    completed = [record for record in records if record.get("event") == "request_completed"]
    latencies = [float(record["e2e_total_ms"]) for record in completed if record.get("e2e_total_ms") is not None]
    completion_times = [float(record["ts_unix_s"]) for record in completed if record.get("ts_unix_s") is not None]
    request_start_times = [
        float(record["ts_unix_s"]) - float(record["e2e_total_ms"]) / 1000.0
        for record in completed
        if record.get("ts_unix_s") is not None and record.get("e2e_total_ms") is not None
    ]
    gpu_samples: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    queue_samples: dict[str, list[float]] = defaultdict(list)
    stage_events: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    batch_composition_events: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    stage1_batch_events: list[Mapping[str, Any]] = []
    stage_postprocess_events: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    stage_intervals: dict[int, list[tuple[float, float]]] = defaultdict(list)
    stage_configurations: dict[str, Mapping[str, Any]] = {}
    tts_outcomes: dict[str, int] = defaultdict(int)
    stage1_input_times: dict[str, float] = {}
    stage1_queue_delays_ms: list[float] = []

    for record in records:
        event = record.get("event")
        if event == "gpu_sample" and record.get("gpu_index") is not None:
            gpu_samples[int(record["gpu_index"])].append(record)
        elif event == "prometheus_sample":
            metric = str(record.get("metric", ""))
            if "num_requests_running" in metric or "num_requests_waiting" in metric:
                labels = record.get("labels") or {}
                stage = labels.get("stage", "pipeline") if isinstance(labels, Mapping) else "pipeline"
                queue_samples[f"{metric}|stage={stage}"].append(float(record.get("value", 0.0)))
        elif event == "stage_completed" and record.get("stage_id") is not None:
            stage_events[int(record["stage_id"])].append(record)
            completed_at = float(record.get("ts_unix_s", 0.0))
            generation_s = float(record.get("stage_gen_time_ms", 0.0)) / 1000.0
            if completed_at > 0 and generation_s >= 0:
                stage_intervals[int(record["stage_id"])].append((completed_at - generation_s, completed_at))
        elif event == "batch_composition_changed" and record.get("stage_id") is not None:
            batch_composition_events[int(record["stage_id"])].append(record)
        elif event == "stage1_batch_started":
            stage1_batch_events.append(record)
        elif event == "stage_postprocess_completed" and record.get("stage_id") is not None:
            stage_postprocess_events[int(record["stage_id"])].append(record)
        elif event == "stage_config" and record.get("stage_id") is not None:
            stage_configurations[str(record["stage_id"])] = record
        elif event == "stage1_input_ready" and record.get("request_id") is not None:
            stage1_input_times[str(record["request_id"])] = float(record.get("ts_unix_s", 0.0))
        elif event == "tts_slot_started" and record.get("request_id") is not None:
            ready_at = stage1_input_times.get(str(record["request_id"]))
            started_at = float(record.get("ts_unix_s", 0.0))
            if ready_at is not None and started_at >= ready_at:
                stage1_queue_delays_ms.append((started_at - ready_at) * 1000.0)
        elif event == "tts_slot_completed":
            tts_outcomes[str(record.get("outcome", "unknown"))] += 1

    duration_s = max(completion_times) - min(request_start_times) if completion_times and request_start_times else 0.0
    summary: dict[str, Any] = {
        "completed_requests": len(completed),
        "completed_requests_per_s": len(completed) / duration_s if duration_s > 0 else 0.0,
        "e2e_latency_ms": {
            "p50": _percentile(latencies, 50),
            "p95": _percentile(latencies, 95),
            "mean": fmean(latencies) if latencies else 0.0,
        },
        "tts_output_outcomes": dict(sorted(tts_outcomes.items())),
        "stage_configurations": {
            stage_id: {
                key: value
                for key, value in config.items()
                if key
                in {
                    "devices",
                    "gpu_memory_utilization",
                    "max_model_len",
                    "max_num_batched_tokens",
                    "max_num_seqs",
                    "tensor_parallel_size",
                }
            }
            for stage_id, config in sorted(stage_configurations.items())
        },
        "stage1_queue_delay_ms": {
            "p50": _percentile(stage1_queue_delays_ms, 50),
            "p95": _percentile(stage1_queue_delays_ms, 95),
        },
        "stages": {},
        "gpus": {},
        "queue": {},
    }
    stage_ids = stage_events.keys() | batch_composition_events.keys()
    if stage1_batch_events:
        stage_ids |= {1}
    for stage_id in sorted(stage_ids):
        events = stage_events.get(stage_id, [])
        composition_events = batch_composition_events.get(stage_id, [])
        observed_batch_events = list(composition_events)
        if stage_id == 1:
            observed_batch_events.extend(stage1_batch_events)
        generation_times = [float(event.get("stage_gen_time_ms", 0.0)) for event in events]
        postprocess_times = [
            float(event.get("postprocess_time_ms", 0.0)) for event in stage_postprocess_events.get(stage_id, [])
        ]
        summary["stages"][str(stage_id)] = {
            "completed": len(events),
            "batch_sizes": [int(event.get("batch_size", 0)) for event in events],
            "batch_composition_sizes": [int(event.get("batch_size", 0)) for event in composition_events],
            "max_observed_batch_size": max(
                (int(event.get("batch_size", 0)) for event in observed_batch_events),
                default=0,
            ),
            "batch_composition_scheduled_tokens": [
                int(event.get("scheduled_tokens", 0)) for event in composition_events
            ],
            "generation_latency_ms": {
                "p50": _percentile(generation_times, 50),
                "p95": _percentile(generation_times, 95),
            },
            "postprocess_latency_ms": {
                "p50": _percentile(postprocess_times, 50),
                "p95": _percentile(postprocess_times, 95),
            },
        }
    for gpu, samples in sorted(gpu_samples.items()):
        used = [int(sample.get("memory_used_bytes", 0)) for sample in samples]
        utilization = [float(sample.get("gpu_utilization_pct", 0.0)) for sample in samples]
        summary["gpus"][str(gpu)] = {
            "peak_memory_used_bytes": max(used, default=0),
            "mean_utilization_pct": fmean(utilization) if utilization else 0.0,
        }
    for stage_id, stage_summary in summary["stages"].items():
        config = stage_configurations.get(stage_id, {})
        devices = _device_indices(config.get("devices"))
        stage_summary["devices"] = devices
        stage_summary["gpus"] = {str(device): summary["gpus"].get(str(device), {}) for device in devices}
    for queue_name, values in sorted(queue_samples.items()):
        summary["queue"][queue_name] = {"max": max(values, default=0.0), "mean": fmean(values) if values else 0.0}
    if 0 in stage_intervals and 1 in stage_intervals:
        summary["cross_stage_overlap_ms"] = (
            _overlap_seconds(_merge_intervals(stage_intervals[0]), _merge_intervals(stage_intervals[1])) * 1000.0
        )
    return summary


def _load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    sample_parser = subparsers.add_parser("sample", help="sample NVML and /metrics into the active JSONL trace")
    sample_parser.add_argument("--devices", default="0,1")
    sample_parser.add_argument("--metrics-url", default=None)
    sample_parser.add_argument("--interval-s", type=float, default=0.5)
    sample_parser.add_argument("--timeout-s", type=float, default=2.0)
    snapshot_parser = subparsers.add_parser("snapshot-config", help="record independent stage settings from YAML")
    snapshot_parser.add_argument("--stage-config", type=Path, required=True)
    summary_parser = subparsers.add_parser("summary", help="summarize a completed JSONL trace")
    summary_parser.add_argument("--trace-path", type=Path, required=True)
    summary_parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.command == "sample":
        devices = [int(device) for device in args.devices.split(",") if device.strip()]
        run_sampler(devices, args.metrics_url, args.interval_s, args.timeout_s)
        return 0
    if args.command == "snapshot-config":
        from omegaconf import OmegaConf

        config = OmegaConf.to_container(OmegaConf.load(args.stage_config), resolve=True)
        emit_stage_config_snapshot(config if isinstance(config, Mapping) else {}, str(args.stage_config))
        return 0

    summary = build_summary(_load_records(args.trace_path))
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
