# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Client-local helpers for duplex JSON metric blocks."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import NamedTuple

__all__ = [
    "STREAMING_OUTPUT_UNIT_TYPES",
    "StageModalityFlags",
    "distribution_summary",
    "metric_mean",
    "stage_modality_flags",
    "summarize_stage_metrics",
]

# Intermediate streaming units that are not the pipeline's text or audio
# final output (e.g. MiniCPM Talker codec tokens).
STREAMING_OUTPUT_UNIT_TYPES = frozenset({"text", "stream", "audio"})


class StageModalityFlags(NamedTuple):
    """How duplex JSON classifies one engine stage."""

    is_text_stage: bool
    is_audio_stage: bool
    is_image_stage: bool
    is_video_stage: bool
    is_internal_stream_stage: bool


def stage_modality_flags(
    final_output_type: object,
    output_unit_type: object,
) -> StageModalityFlags:
    """Classify a stage from wire ``final_output_type`` / ``output_unit_type``."""
    final_type = final_output_type if isinstance(final_output_type, str) else ""
    unit_type = output_unit_type if isinstance(output_unit_type, str) else ""
    is_text_stage = final_type == "text" or unit_type == "text"
    is_audio_stage = final_type == "audio" or unit_type == "audio"
    is_video_stage = final_type in {"video", "videos"} or unit_type == "video"
    # Video diffusion may still report output_unit_type="image" when frames are
    # stored in ``images``; prefer video when final_output_type says so.
    is_image_stage = (not is_video_stage) and (final_type in {"image", "images"} or unit_type == "image")
    is_internal_stream_stage = unit_type in STREAMING_OUTPUT_UNIT_TYPES and not is_text_stage and not is_audio_stage
    return StageModalityFlags(
        is_text_stage=is_text_stage,
        is_audio_stage=is_audio_stage,
        is_image_stage=is_image_stage,
        is_video_stage=is_video_stage,
        is_internal_stream_stage=is_internal_stream_stage,
    )


def _rounded_ms(value: float) -> float:
    return round(float(value), 3)


def _finite_number(value: object, *, nonnegative: bool = False) -> float | None:
    if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(float(value)):
        return None
    number = float(value)
    if nonnegative and number < 0:
        return None
    return number


def _interval_summary(values: list[float]) -> dict[str, float | int]:
    clean = sorted(_rounded_ms(value) for value in values if math.isfinite(value) and value >= 0)
    if not clean:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}

    def nearest_rank(percentile: float) -> float:
        index = max(0, math.ceil(percentile * len(clean)) - 1)
        return clean[min(index, len(clean) - 1)]

    return {
        "count": len(clean),
        "mean": _rounded_ms(sum(clean) / len(clean)),
        "p50": nearest_rank(0.50),
        "p95": nearest_rank(0.95),
        "max": clean[-1],
    }


def distribution_summary(values: Sequence[float], *, digits: int = 3) -> dict[str, float | int] | None:
    """Summarize values as ``{count, mean, p50, p99}`` for duplex report fields."""
    clean = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not clean:
        return None

    def nearest_rank(percentile: float) -> float:
        index = max(0, math.ceil(percentile * len(clean)) - 1)
        return clean[min(index, len(clean) - 1)]

    return {
        "count": len(clean),
        "mean": round(sum(clean) / len(clean), digits),
        "p50": round(nearest_rank(0.50), digits),
        "p99": round(nearest_rank(0.99), digits),
    }


def metric_mean(value: object) -> float | None:
    """Read a scalar mean, or the ``mean`` field of a distribution summary."""
    if isinstance(value, Mapping):
        nested = value.get("mean")
        if isinstance(nested, int | float) and not isinstance(nested, bool) and math.isfinite(float(nested)):
            return float(nested)
        return None
    if isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(float(value)):
        return float(value)
    return None


def _stage_id_sort_key(stage_id: str) -> tuple[int, int | str]:
    try:
        return (0, int(stage_id))
    except ValueError:
        return (1, stage_id)


def _stage_type_labels(stage_metrics: Mapping[str, object]) -> dict[str, object]:
    labels: dict[str, object] = {}
    for key in ("final_output_type", "output_unit_type"):
        value = stage_metrics.get(key)
        if isinstance(value, str) and value:
            labels[key] = value
    return labels


def _stage_ms_list(stage_metrics: Mapping[str, object], field_name: str) -> list[float]:
    raw = stage_metrics.get(field_name)
    return [float(value) for value in raw if isinstance(value, int | float)] if isinstance(raw, list) else []


def _has_legacy_vllm_token_metrics(stage_metrics: Mapping[str, object]) -> bool:
    if stage_metrics.get("vllm_ttft_ms") or stage_metrics.get("vllm_tpot_ms"):
        return True
    raw_itls = stage_metrics.get("vllm_itls_ms")
    return isinstance(raw_itls, list) and bool(raw_itls)


def _text_stage_engine_metrics_block(stage_metrics: Mapping[str, object]) -> dict[str, object]:
    itls = _stage_ms_list(stage_metrics, "vllm_itls_ms")
    return {
        "source": "engine_stage_metrics",
        **_stage_type_labels(stage_metrics),
        "output_token_count": int(stage_metrics.get("num_tokens_out") or 0),
        "ttft_ms": float(stage_metrics.get("vllm_ttft_ms") or 0.0),
        "tpot_ms": float(stage_metrics.get("vllm_tpot_ms") or 0.0),
        "itls_ms": itls,
        "inter_token_interval_ms": _interval_summary(itls),
    }


def _audio_stage_engine_metrics_block(stage_metrics: Mapping[str, object]) -> dict[str, object]:
    block: dict[str, object] = {
        "source": "engine_stage_metrics",
        **_stage_type_labels(stage_metrics),
        "output_unit_count": int(stage_metrics.get("output_unit_count") or 0),
        "audio_generated_frames": int(stage_metrics.get("audio_generated_frames") or 0),
        "audio_duration_s": float(stage_metrics.get("audio_duration_s") or 0.0),
    }
    serving_time = _finite_number(
        stage_metrics.get("serving_time_to_first_output_ms"),
        nonnegative=True,
    )
    if serving_time:
        block["ttfp_ms"] = serving_time
    return block


def _stream_stage_engine_metrics_block(stage_metrics: Mapping[str, object]) -> dict[str, object]:
    icls = _stage_ms_list(stage_metrics, "inter_output_latencies_ms")
    return {
        "source": "engine_stage_metrics",
        **_stage_type_labels(stage_metrics),
        "output_unit_count": int(stage_metrics.get("output_unit_count") or 0),
        "ttfc_ms": float(stage_metrics.get("serving_time_to_first_output_ms") or 0.0),
        "tpop_ms": float(stage_metrics.get("time_per_output_unit_ms") or 0.0),
        "icls_ms": icls,
        "inter_chunk_interval_ms": _interval_summary(icls),
    }


def _stage_engine_metrics_block(stage_metrics: Mapping[str, object]) -> dict[str, object]:
    flags = stage_modality_flags(
        stage_metrics.get("final_output_type"),
        stage_metrics.get("output_unit_type"),
    )
    if flags.is_text_stage or (
        not flags.is_audio_stage
        and not flags.is_internal_stream_stage
        and not flags.is_image_stage
        and not flags.is_video_stage
        and _has_legacy_vllm_token_metrics(stage_metrics)
    ):
        return _text_stage_engine_metrics_block(stage_metrics)
    if flags.is_audio_stage:
        return _audio_stage_engine_metrics_block(stage_metrics)
    if flags.is_internal_stream_stage:
        return _stream_stage_engine_metrics_block(stage_metrics)
    block: dict[str, object] = {"source": "engine_stage_metrics", **_stage_type_labels(stage_metrics)}
    gen_time_ms = stage_metrics.get("stage_gen_time_ms")
    if isinstance(gen_time_ms, int | float) and not isinstance(gen_time_ms, bool):
        block["stage_gen_time_ms"] = float(gen_time_ms)
    return block


def _finite_metric_values(
    rows: Sequence[Mapping[str, object]],
    metric: str,
    *,
    positive: bool = False,
) -> list[float]:
    return [
        float(row[metric])
        for row in rows
        if isinstance(row.get(metric), int | float)
        and not isinstance(row.get(metric), bool)
        and math.isfinite(float(row[metric]))
        and (not positive or float(row[metric]) > 0)
    ]


def summarize_stage_metrics(
    request_metrics: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]] | None:
    """Roll per-response engine stage blocks into ``{count, mean, p50, p99}``.

    Reads ``stages`` when present, otherwise ``stage0_tokens`` as stage ``"0"``.
    Zero or missing ``tpot_ms`` / ``tpop_ms`` values are omitted, matching
    session ``tpot_ms``.
    """
    buckets: dict[str, list[Mapping[str, object]]] = {}
    for request in request_metrics:
        stages = request.get("stages")
        if not isinstance(stages, dict):
            stage0 = request.get("stage0_tokens")
            stages = {"0": stage0} if isinstance(stage0, dict) else {}
        for stage_id, stage_snapshot in stages.items():
            if isinstance(stage_snapshot, dict):
                buckets.setdefault(str(stage_id), []).append(stage_snapshot)
    summary: dict[str, dict[str, object]] = {}
    for stage_id in sorted(buckets, key=_stage_id_sort_key):
        rows = buckets[stage_id]
        stage_summary: dict[str, object] = {}
        for metric, positive in (
            ("ttft_ms", False),
            ("tpot_ms", True),
            ("ttfc_ms", False),
            ("tpop_ms", True),
            ("ttfp_ms", False),
        ):
            rolled = distribution_summary(_finite_metric_values(rows, metric, positive=positive))
            if rolled is not None:
                stage_summary[metric] = rolled
        if stage_summary:
            summary[stage_id] = stage_summary
    return summary or None
