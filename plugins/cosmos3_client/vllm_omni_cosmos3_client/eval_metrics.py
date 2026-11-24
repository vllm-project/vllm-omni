# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Client-side LIBERO success rates and explicitly scoped inference metrics."""

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

LATENCY_METRICS = (
    "policy_wall_ms",
    "client_ms",
    "ttfa_ms",
    "backend_roundtrip_ms",
    "client_time_to_first_action_ms",
    "client_time_to_last_action_ms",
    "client_total_ms",
    "client_infer_ms",
    "client_last_ready_action_ms",
    "client_preprocess_ms",
    "client_observation_to_completion_ms",
    "server_infer_ms",
    "policy_infer_ms",
    "server_ttfa_ms",
    "server_full_chunk_ms",
    "client_ttfa_ms",
    "http_total_ms",
    "server_preprocess_ms",
    "server_queue_wait_ms",
    "server_denormalization_ms",
    "worker_inference_ms",
)

TIMING_CONTRACT = "moh_sample_actions_v1"


def latency_stats(values: list[float]) -> dict[str, Any] | None:
    arr = np.asarray([v for v in values if math.isfinite(v) and v >= 0], dtype=np.float64)
    if not arr.size:
        return None
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        **{f"p{p}": float(np.percentile(arr, p)) for p in (25, 50, 75, 90, 95, 99)},
    }


def make_request_record(
    *,
    request: int,
    endpoint: str,
    batch_size: int,
    result: dict[str, Any],
    http_total_ms: float,
    started_at_s: float,
    client_preprocess_ms: float,
    client_ttfa_ms: float | None,
    partial_events: int,
    action_timings: list[dict[str, Any]],
    warmup: bool,
    execution_window: int = 0,
) -> dict[str, Any]:
    timing = result.get("timing_ms") or {}
    benchmark = result.get("benchmark") or {}
    performance = result.get("performance") or {}
    chunks = [result["action"]] if "action" in result else result.get("actions", [])
    generated_actions = sum(len(chunk) for chunk in chunks)
    executable_actions = sum(
        min(len(chunk), execution_window) if execution_window > 0 else len(chunk) for chunk in chunks
    )
    streaming = endpoint.endswith("_stream")
    ttfa = client_ttfa_ms if streaming else http_total_ms
    if not streaming:
        action_timings = [
            {"sample_index": sample, "action_index": index, "ready_ms": http_total_ms}
            for sample, chunk in enumerate(chunks)
            for index in range(len(chunk))
        ]
    ready_times = [row["ready_ms"] for row in action_timings]
    worker_metrics = performance.get("worker_metrics", [])
    worker_times = [row["worker_inference_ms"] for row in worker_metrics if row.get("worker_inference_ms") is not None]
    # Missing worker timing stays null rather than being replaced with HTTP time.
    worker_total = sum(worker_times) if len(worker_times) == len(chunks) and chunks else None
    record = {
        "request": request,
        "endpoint": endpoint,
        "batch_size": batch_size,
        "streaming": streaming,
        "warmup": warmup,
        "timing_schema": benchmark.get("timing_schema"),
        "timing_contract": result.get("timing_contract"),
        # Match PI's call boundary: observation preparation has already finished.
        "client_ms": http_total_ms,
        "ttfa_ms": ttfa,
        "policy_wall_ms": benchmark.get("policy_wall_ms"),
        "backend_roundtrip_ms": timing.get("backend_roundtrip"),
        "started_s": started_at_s,
        "completed_s": started_at_s + http_total_ms / 1000,
        "started_at_s": started_at_s,
        "completed_at_s": started_at_s + http_total_ms / 1000,
        "generated_chunks": len(chunks),
        "generated_actions": generated_actions,
        "emitted_valid_actions": generated_actions,
        "execution_window": execution_window,
        "executable_actions": executable_actions,
        "http_total_ms": http_total_ms,
        "client_ttfa_ms": ttfa,
        "client_time_to_first_action_ms": client_preprocess_ms + ttfa if ttfa is not None else None,
        # FASTER simple_client records last_action_time after infer_streaming returns.
        "client_time_to_last_action_ms": client_preprocess_ms + http_total_ms,
        "client_total_ms": client_preprocess_ms + http_total_ms,
        "client_infer_ms": client_preprocess_ms + http_total_ms if not streaming else None,
        "client_last_ready_action_ms": max(ready_times) if ready_times else None,
        "client_preprocess_ms": client_preprocess_ms,
        "client_observation_to_completion_ms": client_preprocess_ms + http_total_ms,
        "server_ttfa_ms": timing.get("ttfa"),
        "server_full_chunk_ms": timing.get("full_chunk"),
        "server_infer_ms": (result.get("server_timing") or {}).get("infer_ms"),
        "policy_infer_ms": (
            (result.get("policy_timing") or {}).get("infer_ms")
            if result.get("timing_contract") == TIMING_CONTRACT
            else None
        ),
        "server_preprocess_ms": timing.get("preprocess"),
        "server_queue_wait_ms": timing.get("queue_wait"),
        "server_denormalization_ms": timing.get("denormalization"),
        "worker_inference_ms": worker_total,
        "worker_metrics": worker_metrics,
        "partial_events": partial_events,
        "client_action_timings": action_timings,
        "server_action_timings": performance.get("action_timings", []),
        "policy_profile": performance.get("policy_profile"),
        "action_time_schedule": performance.get("action_time_schedule"),
    }
    return record


def _rate(count: int, seconds: float) -> float | None:
    return count / seconds if seconds > 0 else None


def summarize_model_throughput(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [
        row
        for row in rows
        if row.get("timing_contract") == TIMING_CONTRACT
        and not row.get("streaming")
        and isinstance(row.get("policy_infer_ms"), (int, float))
        and math.isfinite(row["policy_infer_ms"])
        and row["policy_infer_ms"] > 0
    ]
    seconds = sum(row["policy_infer_ms"] for row in valid) / 1000
    chunks = sum(row["generated_chunks"] for row in valid)
    actions = sum(row["generated_actions"] for row in valid)
    complete = bool(rows) and len(valid) == len(rows)
    return {
        "complete": complete,
        "timed_requests": len(valid),
        "timed_chunks": chunks,
        "timed_actions": actions,
        "sum_model_time_s": seconds,
        "requests_per_s": _rate(len(valid), seconds) if complete else None,
        "chunks_per_s": _rate(chunks, seconds) if complete else None,
        "actions_per_s": _rate(actions, seconds) if complete else None,
    }


def summarize_throughput(rows: list[dict[str, Any]]) -> dict[str, Any]:
    intervals = sorted((float(r["started_at_s"]), float(r["completed_at_s"])) for r in rows)
    merged: list[list[float]] = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    active = sum(end - start for start, end in merged)
    elapsed = merged[-1][1] - merged[0][0] if merged else 0.0
    chunks = sum(r["generated_chunks"] for r in rows)
    actions = sum(r["generated_actions"] for r in rows)
    # Old records do not contain the execution window: do not label their raw
    # generation count as executable capacity when reprocessing old logs.
    executable = sum(r["executable_actions"] for r in rows) if all("executable_actions" in r for r in rows) else None
    return {
        "request_count": len(rows),
        "generated_chunks": chunks,
        "generated_actions": actions,
        "timed_actions": actions,
        "executable_actions": executable,
        "http_active_s": active,
        "first_to_last_request_s": elapsed,
        "active_requests_per_s": _rate(len(rows), active),
        "active_chunks_per_s": _rate(chunks, active),
        "active_actions_per_s": _rate(actions, active),
        "active_executable_actions_per_s": _rate(executable, active) if executable is not None else None,
        "sustained_requests_per_s": _rate(len(rows), elapsed),
        "sustained_chunks_per_s": _rate(chunks, elapsed),
        "sustained_actions_per_s": _rate(actions, elapsed),
        "sustained_executable_actions_per_s": _rate(executable, elapsed) if executable is not None else None,
    }


def summarize_request_timings(records: list[dict[str, Any]]) -> dict[str, Any]:
    def summarize(rows):
        return {
            metric: latency_stats(
                [
                    float(row[metric])
                    for row in rows
                    if row.get(metric) is not None
                    and (metric != "policy_infer_ms" or row.get("timing_contract") == TIMING_CONTRACT)
                ]
            )
            for metric in LATENCY_METRICS
        }

    measured = [r for r in records if not r["warmup"]]
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for record in measured:
        groups.setdefault((record["endpoint"], record["batch_size"]), []).append(record)
    return {
        "request_count": len(records),
        "warmup_request_count": len(records) - len(measured),
        "measured_request_count": len(measured),
        "all_requests": summarize(records),
        "steady_state": summarize(measured),
        "throughput": summarize_throughput(measured),
        "by_endpoint_and_batch_size": {
            f"{endpoint}|batch={batch_size}": {
                "request_count": len(rows),
                **summarize(rows),
                "throughput": summarize_throughput(rows),
            }
            for (endpoint, batch_size), rows in sorted(groups.items())
        },
    }


def summarize_accuracy(task_results: list[dict[str, Any]], *, warmup_steps: int = 0) -> dict[str, Any]:
    def summarize(episodes):
        skipped = [e for e in episodes if str(e.get("error", "")).startswith("Skipped")]
        valid = [e for e in episodes if not str(e.get("error", "")).startswith("Skipped")]
        successes = sum(bool(e["success"]) for e in valid)
        rate = _rate(successes, len(valid))
        environment_steps = sum(int(e["steps"]) for e in valid)
        stabilization_actions = sum(int(e.get("warmup_actions", min(warmup_steps, int(e["steps"])))) for e in valid)
        return {
            "planned_episodes": len(episodes),
            "evaluated_episodes": len(valid),
            "skipped_episodes": len(skipped),
            "successful_episodes": successes,
            "failed_episodes": len(valid) - successes,
            "error_episodes": sum(bool(e.get("error")) for e in valid),
            "success_rate": rate,
            "success_rate_percent": rate * 100 if rate is not None else None,
            "success_rate_over_planned": _rate(successes, len(episodes)),
            "environment_steps": environment_steps,
            "warmup_actions": stabilization_actions,
            "executed_actions": environment_steps - stabilization_actions,
        }

    return {
        "metric": "LIBERO task success rate",
        "denominator": "evaluated episodes; runtime errors count as failures",
        "overall": summarize([e for task in task_results for e in task["episode_results"]]),
        "by_task": {str(task["task_id"]): summarize(task["episode_results"]) for task in task_results},
    }


def build_report(
    records: list[dict[str, Any]],
    task_results: list[dict[str, Any]],
    *,
    evaluation_wall_s: float,
    run: dict[str, Any],
) -> dict[str, Any]:
    timings = summarize_request_timings(records)
    accuracy = summarize_accuracy(task_results, warmup_steps=run.get("warmup_steps", 0))
    generated_actions = sum(r["generated_actions"] for r in records)
    generated_chunks = sum(r["generated_chunks"] for r in records)
    timed_rows = [r for r in records if not r["warmup"] and r.get("worker_inference_ms") is not None]
    worker_seconds = sum(r["worker_inference_ms"] for r in timed_rows) / 1000
    timed_chunks = sum(r["generated_chunks"] for r in timed_rows)
    timed_actions = sum(r["generated_actions"] for r in timed_rows)
    return {
        "schema_version": 3,
        "timing_contract": TIMING_CONTRACT,
        "run": run,
        "accuracy": accuracy,
        "request_timing_summary": timings,
        "throughput": {
            "steady_state_inference": summarize_model_throughput([r for r in records if not r["warmup"]]),
            "steady_state_http": timings["throughput"],
            "steady_state_worker_jobs": {
                "timed_requests": len(timed_rows),
                "timed_chunks": timed_chunks,
                "timed_actions": timed_actions,
                "sum_job_wall_s": worker_seconds,
                "chunks_per_s": _rate(timed_chunks, worker_seconds),
                "actions_per_s": _rate(timed_actions, worker_seconds),
            },
            "evaluation_wall": {
                "scope": run.get("evaluation_scope", "legacy_unaligned"),
                "elapsed_s": evaluation_wall_s,
                "completed_requests": len(records),
                "generated_chunks": generated_chunks,
                "generated_actions": generated_actions,
                "executed_actions": accuracy["overall"]["executed_actions"],
                "environment_steps": accuracy["overall"]["environment_steps"],
                "warmup_actions": accuracy["overall"]["warmup_actions"],
                "requests_per_s": _rate(len(records), evaluation_wall_s),
                "generated_chunks_per_s": _rate(generated_chunks, evaluation_wall_s),
                "generated_actions_per_s": _rate(generated_actions, evaluation_wall_s),
                "executed_actions_per_s": _rate(accuracy["overall"]["executed_actions"], evaluation_wall_s),
                "environment_steps_per_s": _rate(accuracy["overall"]["environment_steps"], evaluation_wall_s),
            },
        },
        "scopes": {
            "policy_infer_ms": "worker sampler wall time after input transforms/H2D and before output D2H; no extra device synchronization",
            "model_throughput": "counts / sum of versioned nonstreaming sampler durations; unavailable for missing/invalid or streaming timing",
            "policy_wall_ms": "complete policy call, matching PI benchmark; null for servers without this measurement",
            "client_ms": "HTTP submission to completed response; excludes client image preparation, matching PI client_ms",
            "ttfa_ms": "HTTP submission to first ready action; completed response for nonstream, matching PI ttfa_ms",
            "backend_roundtrip_ms": "backend job submission/polling/callbacks/cleanup; not GPU kernel time",
            "client_time_to_first_action_ms": "client predict entry, including image preparation, to first ready action; completed call for nonstream",
            "client_time_to_last_action_ms": "client predict entry to completed call, matching FASTER simple_client",
            "client_ttfa_ms": "HTTP submission to first ready action; excludes client image preparation",
            "client_last_ready_action_ms": "HTTP submission to last received partial action; response time for nonstream",
            "client_preprocess_ms": "observation image resize/PNG/base64 before HTTP submission",
            "client_action_timings": "per sample and position; ready_ms relative to HTTP submission",
            "server_action_timings": "per sample and position; ready_ms relative to policy entry",
            "worker_inference_ms": "sum of vLLM job generation/persistence wall times; not CUDA kernel time",
            "active_throughput": "counts / union of measured HTTP intervals; excludes environment gaps and warmup",
            "sustained_throughput": "counts / first-to-last measured HTTP interval, including environment gaps",
            "evaluation_wall": "libero_task_loop_v1: all task/env setup, inference, simulation and cleanup; excludes client setup and synthetic warmup",
            "generated_actions": "action positions, including unused chunk suffix; not scalar action dimensions",
            "executable_actions": "sum per chunk of min(valid returned actions, execution window); capacity, not actual execution",
            "executed_actions": "policy action positions actually executed; excludes stabilization dummy actions",
            "environment_steps": "all executed environment actions, including stabilization dummy actions",
            "warmup_actions": "environment stabilization dummy actions; independent of timing_warmup_requests",
            "requests": "successfully completed action HTTP calls; failed calls are reflected in episode errors",
            "accuracy": "LIBERO episode task success; no supervised action MSE is inferred",
        },
    }


def report_text(report: dict[str, Any]) -> str:
    accuracy = report["accuracy"]["overall"]
    pct = accuracy["success_rate_percent"]
    success = f"{pct:.2f}%" if pct is not None else "n/a"
    lines = [
        f"LIBERO success rate: {accuracy['successful_episodes']}/{accuracy['evaluated_episodes']} ({success}); "
        f"errors={accuracy['error_episodes']} skipped={accuracy['skipped_episodes']}",
        f"Timing: {report['request_timing_summary']['measured_request_count']} measured requests, "
        f"{report['request_timing_summary']['warmup_request_count']} warmup requests excluded",
        f"{'Metric (ms)':40s} {'mean':>10s} {'p50':>10s} {'p95':>10s} {'p99':>10s}",
    ]
    for metric, stats in report["request_timing_summary"]["steady_state"].items():
        if stats is not None:
            lines.append(
                f"{metric:40s} {stats['mean']:10.3f} {stats['p50']:10.3f} {stats['p95']:10.3f} {stats['p99']:10.3f}"
            )
    model = report["throughput"]["steady_state_inference"]
    for key in ("requests_per_s", "chunks_per_s", "actions_per_s"):
        value = model[key]
        lines.append(f"model_{key}: {value:.3f}" if value is not None else f"model_{key}: n/a")
    throughput = report["throughput"]["steady_state_http"]
    for key in (
        "active_requests_per_s",
        "active_chunks_per_s",
        "active_actions_per_s",
        "sustained_actions_per_s",
        "active_executable_actions_per_s",
        "sustained_executable_actions_per_s",
    ):
        value = throughput[key]
        lines.append(f"{key}: {value:.3f}" if value is not None else f"{key}: n/a")
    value = report["throughput"]["evaluation_wall"]["executed_actions_per_s"]
    lines.append(
        f"evaluation_executed_actions_per_s: {value:.3f}"
        if value is not None
        else "evaluation_executed_actions_per_s: n/a"
    )
    wall = report["throughput"]["evaluation_wall"]
    value = wall["environment_steps_per_s"]
    lines.append(
        f"evaluation_environment_steps_per_s: {value:.3f}"
        if value is not None
        else "evaluation_environment_steps_per_s: n/a"
    )
    lines.append(
        f"Executed actions: policy={wall['executed_actions']} stabilization={wall['warmup_actions']} "
        f"environment_total={wall['environment_steps']}"
    )
    return "\n".join(lines) + "\n"


def write_report(output_dir: Path, report: dict[str, Any], *, stem: str = "metrics") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / f"{stem}.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(output_dir / f"{stem}.json")
    (output_dir / f"{stem}.txt").write_text(report_text(report), encoding="utf-8")
    columns = (
        "category",
        "group",
        "metric",
        "unit",
        "value",
        "count",
        "mean",
        "std",
        "p25",
        "p50",
        "p75",
        "p90",
        "p95",
        "p99",
        "min",
        "max",
    )
    with (output_dir / f"{stem}.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for group in ("all_requests", "steady_state"):
            for metric, stats in report["request_timing_summary"][group].items():
                if stats is not None:
                    writer.writerow({"category": "latency", "group": group, "metric": metric, "unit": "ms", **stats})
        for group, values in report["throughput"].items():
            for metric, value in values.items():
                if metric.endswith("per_s"):
                    writer.writerow(
                        {"category": "throughput", "group": group, "metric": metric, "unit": "1/s", "value": value}
                    )
        for group, values in {"overall": report["accuracy"]["overall"], **report["accuracy"]["by_task"]}.items():
            writer.writerow(
                {
                    "category": "accuracy",
                    "group": group,
                    "metric": "success_rate",
                    "unit": "%",
                    "value": values["success_rate_percent"],
                    "count": values["evaluated_episodes"],
                }
            )
