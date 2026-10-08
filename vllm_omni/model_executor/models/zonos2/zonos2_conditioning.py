# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Portions Copyright (c) 2026 Zyphra, Inc., originally licensed under MIT.
# Adapted from ZONOS2 194c0a3a, tts/conditioning.py. See LICENSE.zonos2.
"""Frozen official speaking-rate and quality bucket resolution."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import regex as re

_SPEAKING_RATE_FPS = 86.0 * (44070.0 / 44000.0)
_DEFAULT_SPEAKING_RATE_BYTES_PER_SECOND = 15.0
_SPEAKING_RATE_CLOSED_BUCKET_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*$")
_SPEAKING_RATE_OPEN_BUCKET_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*\+\s*$")
_QUALITY_NUMBER_RE = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
_QUALITY_EXACT_BUCKET_RE = re.compile(rf"^\s*({_QUALITY_NUMBER_RE})\s*$")
_QUALITY_CLOSED_BUCKET_RE = re.compile(rf"^\s*({_QUALITY_NUMBER_RE})\s*-\s*({_QUALITY_NUMBER_RE})\s*$")
_QUALITY_OPEN_BUCKET_RE = re.compile(rf"^\s*({_QUALITY_NUMBER_RE})\s*\+\s*$")
_QUALITY_METRIC_FIELDS = (
    "lufs",
    "estimated_snr",
    "max_pause",
    "estimated_bandlimit_hz",
    "leading_silence_s",
    "trailing_silence_s",
)
_DEFAULT_QUALITY_BUCKETS = {"trailing_silence_s": 3}


def _model_speaking_rate_num_buckets(config: Any) -> int:
    model_value = int(getattr(config.model_config, "speaking_rate_num_buckets", 0) or 0)
    server_value = int(getattr(config, "tts_speaking_rate_num_buckets", 0) or 0)
    return model_value or server_value


def _model_speaking_rate_buckets(config: Any) -> list[str]:
    raw = getattr(config.model_config, "speaking_rate_buckets", None)
    if not raw:
        raw = getattr(config, "tts_speaking_rate_buckets", ())
    return [str(item) for item in (raw or ())]


def _parse_speaking_rate_bucket(spec: str) -> tuple[float, float | None]:
    closed = _SPEAKING_RATE_CLOSED_BUCKET_RE.match(str(spec))
    if closed is not None:
        return float(closed.group(1)), float(closed.group(2))

    open_ended = _SPEAKING_RATE_OPEN_BUCKET_RE.match(str(spec))
    if open_ended is not None:
        return float(open_ended.group(1)), None

    raise ValueError(f"Invalid speaking-rate bucket {spec!r}; expected ranges like '0-3' or '60+'.")


def _speaking_rate_bucket_ranges(config: Any) -> list[tuple[float, float | None]]:
    ranges = [_parse_speaking_rate_bucket(spec) for spec in _model_speaking_rate_buckets(config)]
    if not ranges:
        return ranges

    first_low, _ = ranges[0]
    if not math.isclose(first_low, 0.0, abs_tol=1e-9):
        raise ValueError("speaking-rate buckets must start at 0.")

    previous_high: float | None = None
    for idx, (low, high) in enumerate(ranges):
        if low < 0.0:
            raise ValueError("speaking-rate buckets must use non-negative ranges.")
        if high is not None and high <= low:
            raise ValueError(f"speaking-rate bucket {idx} has an empty or inverted range.")
        if previous_high is None and idx > 0:
            raise ValueError("speaking-rate buckets cannot define ranges after an open-ended bucket.")
        if previous_high is not None and not math.isclose(low, previous_high, abs_tol=1e-9):
            raise ValueError("speaking-rate buckets must be contiguous and ordered.")
        previous_high = high

    if ranges[-1][1] is not None:
        raise ValueError("speaking-rate buckets must end with an open-ended range like '60+'.")
    return ranges


def _speaking_rate_bucket_for_rate(
    rate_bytes_per_second: float,
    *,
    num_buckets: int,
    ranges: list[tuple[float, float | None]],
) -> int:
    if rate_bytes_per_second <= 0:
        raise ValueError("speaking_rate must be positive.")

    if ranges:
        for idx, (_, high) in enumerate(ranges):
            if high is None or (
                rate_bytes_per_second < high
                and not math.isclose(rate_bytes_per_second, high, rel_tol=1e-12, abs_tol=1e-9)
            ):
                return idx
        return len(ranges) - 1

    rate_bytes_per_frame = rate_bytes_per_second / _SPEAKING_RATE_FPS
    bucket = int(rate_bytes_per_frame * num_buckets)
    return min(max(bucket, 0), num_buckets - 1)


def _neutral_speaking_rate_bytes_per_second(
    ranges: list[tuple[float, float | None]],
) -> float:
    if not ranges:
        return _DEFAULT_SPEAKING_RATE_BYTES_PER_SECOND

    low, high = ranges[len(ranges) // 2]
    if high is None:
        return max(low, _DEFAULT_SPEAKING_RATE_BYTES_PER_SECOND)
    return (low + high) / 2.0


def _resolve_speaking_rate_bucket(
    config: Any,
    *,
    speaking_rate_bucket: int | None = None,
    speaking_rate: float | None = None,
    speed: float | None = None,
    speaking_rate_enabled: bool = False,
) -> int | None:
    if not speaking_rate_enabled:
        return None

    supplied = [
        speaking_rate_bucket is not None,
        speaking_rate is not None,
        speed is not None,
    ]
    if sum(supplied) == 0:
        return None
    if sum(supplied) > 1:
        raise ValueError("Provide only one of speaking_rate_bucket, speaking_rate, or speed.")

    num_buckets = _model_speaking_rate_num_buckets(config)
    if num_buckets <= 0:
        if speed is not None and speaking_rate_bucket is None and speaking_rate is None:
            return None
        raise ValueError("Current model does not support speaking-rate conditioning.")

    if speaking_rate_bucket is not None:
        bucket = int(speaking_rate_bucket)
        if bucket < 0 or bucket >= num_buckets:
            raise ValueError(f"speaking_rate_bucket must be in [0, {num_buckets - 1}], got {bucket}.")
        return bucket

    ranges = _speaking_rate_bucket_ranges(config)
    if ranges and len(ranges) != num_buckets:
        raise ValueError(f"Model has {num_buckets} speaking-rate buckets, but config defines {len(ranges)} ranges.")

    if speaking_rate is not None:
        return _speaking_rate_bucket_for_rate(
            float(speaking_rate),
            num_buckets=num_buckets,
            ranges=ranges,
        )

    assert speed is not None
    speed_value = float(speed)
    if speed_value <= 0:
        raise ValueError("speed must be positive.")
    return _speaking_rate_bucket_for_rate(
        _neutral_speaking_rate_bytes_per_second(ranges) * speed_value,
        num_buckets=num_buckets,
        ranges=ranges,
    )


def _model_quality_features(config: Any) -> list[str]:
    raw = getattr(config.model_config, "quality_features", None)
    if not raw:
        model_buckets = getattr(config.model_config, "quality_buckets", None) or {}
        raw = model_buckets.keys()
    if not raw:
        raw = getattr(config, "tts_quality_features", ())
    if not raw:
        server_buckets = getattr(config, "tts_quality_buckets", {}) or {}
        raw = server_buckets.keys()
    if raw is None:
        raw = _QUALITY_METRIC_FIELDS
    if isinstance(raw, Mapping) or hasattr(raw, "items"):
        return [str(feature) for feature, enabled in raw.items() if bool(enabled)]
    if isinstance(raw, str):
        return [raw]
    return [str(item) for item in (raw or ())]


def _model_quality_buckets(config: Any) -> dict[str, list[str]]:
    raw = getattr(config.model_config, "quality_buckets", None)
    if not raw:
        raw = getattr(config, "tts_quality_buckets", {})
    features = _model_quality_features(config)
    return {feature: [str(item) for item in ((raw or {}).get(feature, ()) or ())] for feature in features}


def _model_quality_bucket_counts(config: Any) -> list[int]:
    buckets = _model_quality_buckets(config)
    return [len(buckets.get(feature, ())) for feature in _model_quality_features(config)]


def _model_quality_num_buckets(config: Any) -> int:
    model_value = int(getattr(config.model_config, "quality_num_buckets", 0) or 0)
    server_value = int(getattr(config, "tts_quality_num_buckets", 0) or 0)
    return model_value or server_value or sum(_model_quality_bucket_counts(config))


def _parse_quality_bucket(spec: str) -> tuple[str, float, float | None]:
    value = str(spec)
    exact = _QUALITY_EXACT_BUCKET_RE.match(value)
    if exact is not None:
        return "exact", float(exact.group(1)), None

    closed = _QUALITY_CLOSED_BUCKET_RE.match(value)
    if closed is not None:
        return "range", float(closed.group(1)), float(closed.group(2))

    open_ended = _QUALITY_OPEN_BUCKET_RE.match(value)
    if open_ended is not None:
        return "range", float(open_ended.group(1)), None

    raise ValueError(
        f"Invalid quality bucket {spec!r}; expected exact values like '0', "
        "ranges like '-30--25', or open-ended ranges like '22050+'."
    )


def _quality_bucket_specs(config: Any, feature: str) -> list[tuple[str, float, float | None]]:
    raw = _model_quality_buckets(config).get(feature, ())
    specs = [_parse_quality_bucket(spec) for spec in raw]
    for idx, (kind, low, high) in enumerate(specs):
        if not math.isfinite(low):
            raise ValueError(f"quality_buckets.{feature} must use finite bucket values.")
        if kind == "range":
            if high is not None and not math.isfinite(high):
                raise ValueError(f"quality_buckets.{feature} must use finite bucket values.")
            if high is not None and high <= low:
                raise ValueError(f"quality_buckets.{feature} has an empty or inverted range at index {idx}.")
    return specs


def _quality_bucket_for_value(value: float, config: Any, feature: str) -> int | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None

    specs = _quality_bucket_specs(config, feature)
    if not specs:
        return None

    for idx, (kind, low, _) in enumerate(specs):
        if kind == "exact" and math.isclose(value, low, rel_tol=1e-12, abs_tol=1e-9):
            return idx

    range_indexes = [idx for idx, (kind, _, _) in enumerate(specs) if kind == "range"]
    if not range_indexes:
        return None

    for idx in range_indexes:
        _, low, high = specs[idx]
        if high is None:
            if value >= low:
                return idx
        elif idx == range_indexes[-1]:
            if low <= value <= high:
                return idx
        elif low <= value < high:
            return idx

    _, first_low, _ = specs[range_indexes[0]]
    if value < first_low:
        return range_indexes[0]
    return range_indexes[-1]


def _quality_control_to_feature_list(value: Any, features: list[str]) -> list[Any]:
    if value is None:
        return [None] * len(features)
    if isinstance(value, dict):
        return [value.get(feature) for feature in features]
    if isinstance(value, (list, tuple)):
        return [value[idx] if idx < len(value) else None for idx in range(len(features))]
    raise ValueError("quality_buckets and quality_values must be a list or feature-name object.")


def _resolve_quality_buckets(
    config: Any,
    *,
    quality_buckets: Any = None,
    quality_values: Any = None,
    quality_enabled: bool = False,
) -> list[int | None] | None:
    if not quality_enabled:
        return None

    if quality_buckets is None and quality_values is None:
        return None
    if quality_buckets is not None and quality_values is not None:
        raise ValueError("Provide only one of quality_buckets or quality_values.")

    features = _model_quality_features(config)
    counts = _model_quality_bucket_counts(config)
    if not features or _model_quality_num_buckets(config) <= 0 or sum(counts) <= 0:
        raise ValueError("Current model does not support quality conditioning.")

    if any(count <= 0 for count in counts):
        raise ValueError("Every configured quality feature must define at least one bucket.")

    if quality_buckets is not None:
        raw_buckets = _quality_control_to_feature_list(quality_buckets, features)
        resolved: list[int | None] = []
        for feature, count, raw_bucket in zip(features, counts, raw_buckets, strict=True):
            if raw_bucket is None:
                resolved.append(None)
                continue
            bucket = int(raw_bucket)
            if bucket < 0 or bucket >= count:
                raise ValueError(f"quality_buckets.{feature} must be in [0, {count - 1}], got {bucket}.")
            resolved.append(bucket)
        return resolved

    raw_values = _quality_control_to_feature_list(quality_values, features)
    return [
        _quality_bucket_for_value(raw_value, config, feature) if raw_value is not None else None
        for feature, raw_value in zip(features, raw_values, strict=True)
    ]
