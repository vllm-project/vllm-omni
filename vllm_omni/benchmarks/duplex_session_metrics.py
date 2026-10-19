# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Assemble duplex serve metrics from a public EventCollector.

This module does not instrument the server. It reads the same
``EventCollector.timing_summary`` / ``global_timing_summary`` surfaces that
OmniInteract already uses, derives RTF with ``compute_audio_rtf``, and folds
per-session ``stream_*`` values into a run-level report.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from vllm_omni.metrics.definitions import compute_audio_rtf

if TYPE_CHECKING:
    from vllm_omni.clients.duplex import EventCollector

DUPLEX_METRICS_FILENAME = "duplex_metrics.json"

_REQUEST_MEASUREMENT_ORIGIN = {
    "tpot": "Stage-0 engine mean time per output token",
    "rtf": "response.created client receive to last audio packet divided by emitted audio duration",
}
_CLIENT_MEASUREMENT_ORIGIN = {
    "client_ttft": (
        "response.created client receive to first non-empty text delta; includes server prompt preparation "
        "and engine submission when they follow response creation, as in Qwen3-Omni"
    ),
    "client_ttfp": (
        "response.created client receive to first audio packet; includes server prompt preparation "
        "and engine submission when they follow response creation, as in Qwen3-Omni"
    ),
}
_TEXT_DELTA_TYPES = frozenset(
    {"response.output_audio_transcript.delta", "response.output_text.delta", "response.text.delta"}
)
_AUDIO_DELTA_TYPE = "response.output_audio.delta"

#: Per-response latencies exported as flat ``mean_/median_/p99_<name>`` keys,
#: the shape perf baselines read (like ``mean_ttft_ms``). Every response
#: counts once, however many responses its session had.
DUPLEX_RESPONSE_LATENCY_FIELDS = (
    ("ttft_ms", "duplex_response_ttft_ms", "server TTFT"),
    ("ttfp_ms", "duplex_response_ttfp_ms", "server TTFP"),
    ("client_ttft_ms", "duplex_client_ttft_ms", "client TTFT"),
    ("client_ttfp_ms", "duplex_client_ttfp_ms", "client TTFP"),
)
_STREAM_MEASUREMENT_ORIGIN = {
    "ttft": "input stream start to first non-empty text delta",
    "ttfp": "input stream start to first audio packet",
    "rtf": (
        "input stream start-to-last-audio receive time divided by total emitted audio duration; "
        "includes concurrent realtime input"
    ),
}


@dataclass(frozen=True)
class DuplexSessionMetricBundle:
    """One session's request rows plus the session/stream summary."""

    request_metrics: list[dict[str, object]]
    session_metrics: dict[str, object]
    output_tokens: int


def audio_rtf_from_raw_metric(raw_metric: Mapping[str, object]) -> float | None:
    """Derive audio RTF from client-reported generation and duration milliseconds."""
    generation_ms = raw_metric.get("audio_generation_ms")
    duration_ms = raw_metric.get("audio_duration_ms")
    if not isinstance(generation_ms, int | float) or not isinstance(duration_ms, int | float) or duration_ms <= 0:
        return None
    return round(compute_audio_rtf(float(generation_ms) / 1000.0, float(duration_ms) / 1000.0), 6)


def _client_response_latencies(
    collector: EventCollector,
    response_id: str,
    *,
    after_s: float,
) -> tuple[float | None, float | None]:
    """Client-observed ``response.created`` to first text and first audio, in ms.

    These timers start when the client learns a response exists. Qwen3-Omni
    creates the response before prompt preparation and engine submission, so
    that work is included. Model-native duplex may create the response only
    with the first output; its earlier preparation is then excluded.
    """
    created_at_s: float | None = None
    first_text_at_s: float | None = None
    first_audio_at_s: float | None = None
    for event, received_at_s in zip(collector.events, collector.event_received_at_s, strict=True):
        if received_at_s < after_s or collector.response_id(event) != response_id:
            continue
        event_type = event.get("type")
        if event_type == "response.created":
            created_at_s = received_at_s if created_at_s is None else created_at_s
        elif event_type in _TEXT_DELTA_TYPES and isinstance(event.get("delta"), str) and event["delta"]:
            first_text_at_s = received_at_s if first_text_at_s is None else first_text_at_s
        elif event_type == _AUDIO_DELTA_TYPE and isinstance(event.get("delta") or event.get("audio"), str):
            first_audio_at_s = received_at_s if first_audio_at_s is None else first_audio_at_s
    if created_at_s is None:
        return None, None
    started_at_s = created_at_s

    def elapsed_ms(at_s: float | None) -> float | None:
        return None if at_s is None else round(max(0.0, at_s - started_at_s) * 1000.0, 3)

    return elapsed_ms(first_text_at_s), elapsed_ms(first_audio_at_s)


def collect_duplex_session_metrics(
    collector: EventCollector,
    *,
    stream_start: float,
    session_id: str | None,
) -> DuplexSessionMetricBundle:
    """Build per-response and session/stream metrics for one duplex session.

    ``stream_start`` is the client monotonic time just before input media is
    pushed. Per-response TTFT/TTFP prefer server ``response_request_metrics``
    when present; RTF and the session ``stream_*`` window stay client-receive.
    """
    from vllm_omni.clients.duplex import summarize_session_request_metrics

    request_metrics: list[dict[str, object]] = []
    output_tokens = 0
    for request_index, response_id in enumerate(collector.response_ids):
        timing = collector.timing_summary(
            after_s=stream_start,
            input_committed_at_s=None,
            response_id=response_id,
            measurement_origin=_REQUEST_MEASUREMENT_ORIGIN,
        )
        raw_metric = timing.get("request_metrics")
        stages = timing.get("stages")
        stage0 = timing.get("stage0_tokens")
        metric: dict[str, object] = {
            "session_id": session_id,
            "request_index": request_index,
            "response_id": response_id,
        }
        if isinstance(raw_metric, dict):
            metric.update(raw_metric)
            metric["rtf"] = audio_rtf_from_raw_metric(raw_metric)
            metric["client_ttft_ms"], metric["client_ttfp_ms"] = _client_response_latencies(
                collector, response_id, after_s=stream_start
            )
            origin = metric.get("measurement_origin")
            metric["measurement_origin"] = {
                **(origin if isinstance(origin, dict) else {}),
                **_CLIENT_MEASUREMENT_ORIGIN,
            }
        copied_stages: dict[str, dict[str, object]] = {}
        if isinstance(stages, dict):
            for stage_id, stage_snapshot in stages.items():
                if isinstance(stage_snapshot, dict):
                    copied_stages[str(stage_id)] = dict(stage_snapshot)
        if copied_stages:
            metric["stages"] = copied_stages
        if isinstance(stage0, dict):
            metric["stage0_tokens"] = dict(stage0)
            output_tokens += int(stage0.get("output_token_count") or 0)
        if isinstance(raw_metric, dict) or copied_stages or isinstance(stage0, dict):
            request_metrics.append(metric)
    session_metrics = summarize_session_request_metrics(
        request_metrics,
        session_id=session_id,
    )
    from vllm_omni.clients.duplex import distribution_summary

    for field in ("client_ttft_ms", "client_ttfp_ms"):
        if (summary := distribution_summary(_finite_nonnegative(request_metrics, field))) is not None:
            session_metrics[field] = summary
    stream_metrics = collector.global_timing_summary(
        after_s=stream_start,
        window_started_at_s=stream_start,
        response_ids=list(collector.response_ids),
        measurement_origin=_STREAM_MEASUREMENT_ORIGIN,
    )
    if stream_metrics:
        session_metrics.update(
            {
                "stream_ttft_ms": stream_metrics.get("ttft_ms"),
                "stream_ttfp_ms": stream_metrics.get("ttfp_ms"),
                "stream_rtf": audio_rtf_from_raw_metric(stream_metrics),
                "stream_audio_generation_ms": stream_metrics.get("audio_generation_ms"),
                "stream_audio_duration_ms": stream_metrics.get("audio_duration_ms"),
                "stream_measurement_origin": stream_metrics.get("measurement_origin"),
            }
        )
    return DuplexSessionMetricBundle(
        request_metrics=request_metrics,
        session_metrics=session_metrics,
        output_tokens=output_tokens,
    )


def _finite_nonnegative(rows: Sequence[Mapping[str, object]], field: str) -> list[float]:
    return [
        float(value)
        for row in rows
        if isinstance((value := row.get(field)), int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    ]


def duplex_response_latency_metrics(request_metrics: Sequence[Mapping[str, object]]) -> dict[str, float | int]:
    """Flatten per-response latencies into ``num_/mean_/median_/p99_<name>`` keys."""
    from vllm_omni.clients.duplex import distribution_summary

    result: dict[str, float | int] = {}
    for field, name, _ in DUPLEX_RESPONSE_LATENCY_FIELDS:
        summary = distribution_summary(_finite_nonnegative(request_metrics, field))
        if summary is None:
            continue
        result[f"num_{name}_samples"] = summary["count"]
        result[f"mean_{name}"] = summary["mean"]
        result[f"median_{name}"] = summary["p50"]
        result[f"p99_{name}"] = summary["p99"]
    return result


def print_duplex_response_latency_metrics(metrics: Mapping[str, float | int]) -> None:
    """Print the per-response latency block of a benchmark summary."""
    if not metrics:
        return
    print("{s:{c}^{n}}".format(s=" Duplex Per-Response Latency ", n=50, c="="))
    for _, name, label in DUPLEX_RESPONSE_LATENCY_FIELDS:
        if f"mean_{name}" not in metrics:
            continue
        for stat, title in (("mean", "Mean"), ("median", "Median"), ("p99", "P99")):
            print("{:<40} {:<10.2f}".format(f"{title} {label} (ms):", metrics[f"{stat}_{name}"]))
    print("=" * 50)


def duplex_stream_metrics(session_metrics: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Summarize finite, non-negative per-session ``stream_*`` values."""
    from vllm_omni.clients.duplex import distribution_summary

    result: dict[str, object] = {}
    for session_key, result_key, digits in (
        ("stream_ttft_ms", "duplex_stream_ttft_ms", 3),
        ("stream_ttfp_ms", "duplex_stream_ttfp_ms", 3),
        ("stream_rtf", "duplex_stream_rtf", 6),
    ):
        values = [
            float(value)
            for metric in session_metrics
            if isinstance((value := metric.get(session_key)), int | float)
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value >= 0
        ]
        summary = distribution_summary(values, digits=digits)
        if summary is not None:
            result[result_key] = summary
    return result


def build_duplex_metrics_report(
    *,
    request_metrics: Sequence[Mapping[str, object]],
    session_metrics: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Flatten this generate run into the ``duplex_metrics.json`` payload."""
    report: dict[str, object] = {
        "duplex_request_metrics": [dict(metric) for metric in request_metrics],
        "duplex_session_metrics": [dict(metric) for metric in session_metrics],
    }
    report.update(duplex_stream_metrics(session_metrics))
    report.update(duplex_response_latency_metrics(request_metrics))
    return report


def sample_metric_key(metric: Mapping[str, object]) -> tuple[str, str] | None:
    """Identity used to merge resume rows: ``(split, sample_id)``."""
    split = metric.get("split")
    sample_id = metric.get("sample_id")
    if isinstance(split, str) and split and isinstance(sample_id, str) and sample_id:
        return (split, sample_id)
    return None


def merge_metric_rows(
    existing: Sequence[Mapping[str, object]],
    incoming: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Replace existing rows that share an incoming ``(split, sample_id)``.

    Incoming rows without a key are appended. Existing rows without a key are
    kept because they cannot be matched.
    """
    incoming_keys = {key for metric in incoming if (key := sample_metric_key(metric)) is not None}
    merged = [dict(metric) for metric in existing if (key := sample_metric_key(metric)) not in incoming_keys]
    merged.extend(dict(metric) for metric in incoming)
    return merged


def read_duplex_metrics_report(path: Path) -> dict[str, object] | None:
    """Load an on-disk report. Missing files return ``None``; corrupt JSON raises."""
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _metric_rows(value: object, *, field_name: str) -> list[Mapping[str, object]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a JSON list")
    rows: list[Mapping[str, object]] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError(f"{field_name} entries must be JSON objects")
        rows.append(item)
    return rows


def merge_duplex_metrics_report(
    existing: Mapping[str, object] | None,
    *,
    request_metrics: Sequence[Mapping[str, object]],
    session_metrics: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Merge this run into a previous ``duplex_metrics.json`` payload."""
    previous_requests = _metric_rows(
        None if existing is None else existing.get("duplex_request_metrics"),
        field_name="duplex_request_metrics",
    )
    previous_sessions = _metric_rows(
        None if existing is None else existing.get("duplex_session_metrics"),
        field_name="duplex_session_metrics",
    )
    return build_duplex_metrics_report(
        request_metrics=merge_metric_rows(previous_requests, request_metrics),
        session_metrics=merge_metric_rows(previous_sessions, session_metrics),
    )
