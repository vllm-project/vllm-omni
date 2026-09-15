# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from statistics import mean, median
from typing import Sequence

import numpy as np

ACTION_HORIZON = 24
DEFAULT_ACTION_DIM = 8


@dataclass(frozen=True)
class RequestTiming:
    index: int
    label: str
    latency_s: float
    action_count: int
    action_shape: tuple[int, ...]
    video_latent_frames: int


def validate_action_array(
    actions: np.ndarray,
    *,
    request_index: int,
    expected_action_horizon: int = ACTION_HORIZON,
    expected_action_dim: int = DEFAULT_ACTION_DIM,
) -> dict[str, object]:
    if actions.shape != (expected_action_horizon, expected_action_dim):
        raise AssertionError(
            f"Action {request_index} shape mismatch: expected "
            f"{(expected_action_horizon, expected_action_dim)}, got {actions.shape}"
        )
    if not np.isfinite(actions).all():
        raise AssertionError(f"Action {request_index} contains non-finite values")
    return {"shape": list(actions.shape), "count": int(actions.shape[0])}


def compare_action_sequences(
    actual: Sequence[np.ndarray],
    reference: Sequence[np.ndarray],
    *,
    atol: float,
) -> dict[str, object]:
    if len(actual) != len(reference):
        raise AssertionError(f"Expected {len(reference)} action chunks, got {len(actual)}")

    chunks: list[dict[str, object]] = []
    for index, (actual_chunk, reference_chunk) in enumerate(zip(actual, reference)):
        actual_arr = np.asarray(actual_chunk, dtype=np.float32)
        reference_arr = np.asarray(reference_chunk, dtype=np.float32)
        if actual_arr.shape != reference_arr.shape:
            raise AssertionError(
                f"Reference action {index} shape mismatch: expected "
                f"{reference_arr.shape}, got {actual_arr.shape}"
            )

        delta = actual_arr - reference_arr
        max_abs_error = float(np.max(np.abs(delta))) if delta.size else 0.0
        rmse = float(sqrt(float(np.mean(np.square(delta))))) if delta.size else 0.0
        chunks.append(
            {
                "index": index,
                "shape": list(actual_arr.shape),
                "max_abs_error": max_abs_error,
                "rmse": rmse,
                "passed": max_abs_error <= atol,
            }
        )

    return {
        "mode": "reference_actions",
        "atol": float(atol),
        "passed": all(bool(chunk["passed"]) for chunk in chunks),
        "chunks": chunks,
    }


def summarize_benchmark(
    request_timings: Sequence[RequestTiming],
    *,
    decoded_video_frames: int,
    decode_latency_s: float,
    write_latency_s: float,
    output_video_path: str,
    actions_path: str,
    accuracy: dict[str, object] | None = None,
) -> dict[str, object]:
    if not request_timings:
        raise ValueError("request_timings must not be empty")

    latencies = [float(timing.latency_s) for timing in request_timings]
    steady_latencies = latencies[1:]
    model_generate_total = sum(latencies)
    model_plus_decode_total = model_generate_total + float(decode_latency_s)
    total_actions = sum(int(timing.action_count) for timing in request_timings)

    return {
        "requests": {
            "count": len(request_timings),
            "items": [asdict(timing) for timing in request_timings],
        },
        "latency_s": {
            "first_request": latencies[0],
            "all_requests": _latency_stats(latencies),
            "steady_state": _latency_stats(steady_latencies),
            "model_generate_total": model_generate_total,
            "decode_video": float(decode_latency_s),
            "write_artifacts": float(write_latency_s),
            "model_plus_decode_total": model_plus_decode_total,
            "model_plus_decode_and_write_total": model_plus_decode_total + float(write_latency_s),
        },
        "throughput": {
            "model_request_hz": _safe_div(len(request_timings), model_generate_total),
            "steady_state_request_hz": _safe_div(len(steady_latencies), sum(steady_latencies)),
            "model_action_hz": _safe_div(total_actions, model_generate_total),
            "model_video_fps": _safe_div(decoded_video_frames, model_generate_total),
            "model_plus_decode_video_fps": _safe_div(decoded_video_frames, model_plus_decode_total),
        },
        "video": {
            "decoded_frames": int(decoded_video_frames),
        },
        "accuracy": accuracy
        or {
            "mode": "shape_finite_only",
            "passed": True,
        },
        "artifacts": {
            "output_video": output_video_path,
            "actions": actions_path,
        },
    }


def _latency_stats(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "p50": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    values_list = [float(value) for value in values]
    return {
        "count": len(values_list),
        "mean": float(mean(values_list)),
        "p50": float(median(values_list)),
        "p95": float(np.percentile(values_list, 95)),
        "min": min(values_list),
        "max": max(values_list),
    }


def _safe_div(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator / denominator)
