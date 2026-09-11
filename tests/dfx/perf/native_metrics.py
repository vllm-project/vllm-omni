# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Pure validation and aggregation for native KV-append benchmark results."""

import math
import statistics
from typing import TypedDict


class _Stage0TokenMetrics(TypedDict):
    source: str
    output_token_count: int
    ttft_ms: float
    itls_ms: list[float]


def native_measurement(result: dict, *, sessions: int, turns: int, elapsed_s: float) -> dict:
    """Reject fallback/incomplete output before calculating native metrics."""
    if not math.isfinite(elapsed_s) or elapsed_s <= 0:
        raise AssertionError("invalid native workload elapsed time")
    assert result.get("ok") is True, result.get("failures", result)
    assert result.get("session_count") == sessions
    assert result.get("identity_isolation_ok") is True
    assert result.get("native_model_turn_end_ok") is True
    assert result.get("model_turn_end_count") == sessions * turns
    session_rows = result.get("sessions", [])
    assert len(session_rows) == sessions
    request_ids: set[str] = set()
    append_count = 0
    max_context = 0
    metrics = []
    stage0_metrics: list[_Stage0TokenMetrics] = []
    for row in session_rows:
        assert row.get("error_count") == 0
        assert row.get("done_count") == turns
        assert row.get("audio_delta_count", 0) > 0
        observations = row.get("native_append_observations", [])
        assert observations, "native KV append receipts missing (possible chat fallback)"
        ids = {item.get("request_id") for item in observations}
        assert len(ids) == 1 and all(isinstance(rid, str) and rid.startswith("duplex-s.") for rid in ids)
        assert not request_ids.intersection(ids), "physical request shared across sessions"
        request_ids.update(rid for rid in ids if isinstance(rid, str))
        # vLLM 0.28's resumable adapter exposes committed receipt counts and
        # prompt/computed offsets, not the older streaming_prompt flags.
        receipts = [item["omni_append_receipt_count"] for item in observations]
        assert receipts == sorted(receipts) and receipts[-1] > 0
        assert any(item["num_prompt_tokens"] > item["num_computed_tokens"] for item in observations)
        replicas = {item.get("replica_id") for item in observations}
        assert len(replicas) == 1 and all(type(replica) is int and replica >= 0 for replica in replicas)
        contexts = [item["omni_context_tokens"] for item in observations]
        assert contexts == sorted(contexts)
        assert all(0 < item["omni_context_tokens"] <= item["omni_context_limit"] for item in observations)
        max_context = max(max_context, max(contexts))
        append_count += len(observations)
        request_metrics = row.get("request_metrics", [])
        assert len(request_metrics) == turns
        metrics.extend(request_metrics)
        stage0_metrics.extend(
            timing["stage0_tokens"]
            for timing in row.get("response_timings", {}).values()
            if isinstance(timing.get("stage0_tokens"), dict)
            and timing["stage0_tokens"].get("source") == "engine_stage_metrics"
        )
    measured: dict[str, object] = {
        "elapsed_s": elapsed_s,
        "completed_sessions": sessions,
        "completed_responses": sessions * turns,
        "sessions_per_s": sessions / elapsed_s,
        "responses_per_s": sessions * turns / elapsed_s,
        "native_append_observations": append_count,
        "max_context_tokens": max_context,
    }
    for name in ("ttft_ms", "ttfp_ms", "rtf"):
        values = [item[name] for item in metrics]
        assert all(isinstance(v, int | float) and math.isfinite(v) and v >= 0 for v in values), name
        measured[f"mean_{name}"] = statistics.mean(values)
        measured[f"max_{name}"] = max(values)
    durations = [item["audio_duration_ms"] for item in metrics]
    assert all(isinstance(v, int | float) and math.isfinite(v) and v > 0 for v in durations)
    generated_audio_s = sum(durations) / 1000
    measured["generated_audio_s"] = generated_audio_s
    measured["audio_seconds_per_s"] = generated_audio_s / elapsed_s
    measured["stage0_token_metrics_available"] = len(stage0_metrics) == sessions * turns
    if measured["stage0_token_metrics_available"]:
        measured["stage0_output_tokens"] = sum(item["output_token_count"] for item in stage0_metrics)
        measured["stage0_mean_ttft_ms"] = statistics.mean(item["ttft_ms"] for item in stage0_metrics)
        itls = [itl for item in stage0_metrics for itl in item.get("itls_ms", [])]
        measured["stage0_mean_itl_ms"] = statistics.mean(itls) if itls else None
    return measured


def native_performance_gate(aggregate: dict, baselines: dict) -> str:
    """Use only a same-hardware measured baseline; never grade profiler runs."""
    if aggregate.get("profiling_run_not_performance_baseline") or aggregate.get(
        "instrumentation_run_not_performance_baseline"
    ):
        return "not_checked_instrumented"
    baseline = baselines.get(aggregate.get("hardware"))
    if baseline is None:
        return "unbaselined_hardware"
    tolerance = baseline["regression_tolerance"]
    assert 0 <= tolerance < 1
    assert aggregate["median_responses_per_s"] >= baseline["median_responses_per_s"] * (1 - tolerance)
    assert aggregate["median_ttfp_ms"] <= baseline["median_ttfp_ms"] * (1 + tolerance)
    return "passed"
