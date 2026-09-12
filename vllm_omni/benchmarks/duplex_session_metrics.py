# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Assemble duplex serve metrics from a public EventCollector.

This module does not instrument the server. It reads the same
``EventCollector.timing_summary`` / ``global_timing_summary`` surfaces that
OmniInteract already uses, derives RTF with ``compute_audio_rtf``, and folds
per-session ``global_*`` values into a run-level report.
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
_GLOBAL_MEASUREMENT_ORIGIN = {
    "ttft": "input stream start to first non-empty text delta",
    "ttfp": "input stream start to first audio packet",
    "rtf": (
        "input stream start-to-last-audio receive time divided by total emitted audio duration; "
        "includes concurrent realtime input"
    ),
}


@dataclass(frozen=True)
class DuplexSessionMetricBundle:
    """One session's request rows plus the session/global summary."""

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


def collect_duplex_session_metrics(
    collector: EventCollector,
    *,
    stream_start: float,
    session_id: str | None,
) -> DuplexSessionMetricBundle:
    """Build per-response and session/global metrics for one duplex session.

    ``stream_start`` is the client monotonic time just before input media is
    pushed. Per-response TTFT/TTFP prefer server ``response_request_metrics``
    when present; RTF and the session ``global_*`` window stay client-receive.
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
        stage0 = timing.get("stage0_tokens")
        metric: dict[str, object] = {
            "session_id": session_id,
            "request_index": request_index,
            "response_id": response_id,
        }
        if isinstance(raw_metric, dict):
            metric.update(raw_metric)
            metric["rtf"] = audio_rtf_from_raw_metric(raw_metric)
        if isinstance(stage0, dict):
            metric["stage0_tokens"] = dict(stage0)
            output_tokens += int(stage0.get("output_token_count") or 0)
        if isinstance(raw_metric, dict) or isinstance(stage0, dict):
            request_metrics.append(metric)
    session_metrics = summarize_session_request_metrics(
        request_metrics,
        session_id=session_id,
    )
    global_metrics = collector.global_timing_summary(
        after_s=stream_start,
        window_started_at_s=stream_start,
        response_ids=list(collector.response_ids),
        measurement_origin=_GLOBAL_MEASUREMENT_ORIGIN,
    )
    if global_metrics:
        session_metrics.update(
            {
                "global_ttft_ms": global_metrics.get("ttft_ms"),
                "global_ttfp_ms": global_metrics.get("ttfp_ms"),
                "global_rtf": audio_rtf_from_raw_metric(global_metrics),
                "global_audio_generation_ms": global_metrics.get("audio_generation_ms"),
                "global_audio_duration_ms": global_metrics.get("audio_duration_ms"),
                "global_measurement_origin": global_metrics.get("measurement_origin"),
            }
        )
    return DuplexSessionMetricBundle(
        request_metrics=request_metrics,
        session_metrics=session_metrics,
        output_tokens=output_tokens,
    )


def mean_duplex_global_metrics(session_metrics: Sequence[Mapping[str, object]]) -> dict[str, float]:
    """Average finite, non-negative per-session ``global_*`` values."""
    result: dict[str, float] = {}
    for session_key, result_key in (
        ("global_ttft_ms", "mean_duplex_global_ttft_ms"),
        ("global_ttfp_ms", "mean_duplex_global_ttfp_ms"),
        ("global_rtf", "mean_duplex_global_rtf"),
    ):
        values = [
            float(value)
            for metric in session_metrics
            if isinstance((value := metric.get(session_key)), int | float)
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value >= 0
        ]
        if values:
            result[result_key] = sum(values) / len(values)
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
    report.update(mean_duplex_global_metrics(session_metrics))
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
