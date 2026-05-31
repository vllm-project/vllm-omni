# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

HEADER_SUPER_P95_SACRIFICIAL = "X-Super-P95-Sacrificial"
HEADER_SUPER_P95_ESTIMATED_SERVICE_S = "X-Super-P95-Estimated-Service-S"
HEADER_SUPER_P95_NORMAL_LOAD_S = "X-Super-P95-Normal-Load-S"
HEADER_SUPER_P95_SACRIFICIAL_LOAD_S = "X-Super-P95-Sacrificial-Load-S"

EXTRA_ARG_SUPER_P95_SACRIFICIAL = "_super_p95_sacrificial"
EXTRA_ARG_SUPER_P95_ESTIMATED_SERVICE_S = "_super_p95_estimated_service_s"

DEFAULT_SUPER_P95_HARDWARE_PROFILE = "910B2"
_KNOWN_SUPER_P95_HARDWARE_PROFILES = {"910B2", "910B3"}

_QWEN_IMAGE_HARDWARE_ANCHORS_S: dict[str, dict[tuple[int, int], tuple[int, float]]] = {
    "910B2": {
        (512, 512): (20, 8.60),
        (768, 768): (20, 8.94),
        (1024, 1024): (25, 14.22),
        (1536, 1536): (35, 43.22),
    },
    "910B3": {
        (512, 512): (20, 8.64),
        (768, 768): (20, 8.64),
        (1024, 1024): (25, 14.22),
        (1536, 1536): (35, 49.34),
    },
}

_WAN2_2_HARDWARE_ANCHORS_S: dict[str, dict[tuple[int, int, int, int], float]] = {
    "910B2": {
        (854, 480, 3, 80): 38.07,
        (854, 480, 4, 120): 71.34,
        (1280, 720, 6, 80): 119.71,
    },
    "910B3": {
        (854, 480, 3, 80): 38.07,
        (854, 480, 4, 120): 71.34,
        (1280, 720, 6, 80): 119.71,
    },
}

_QWEN_IMAGE_FALLBACK_BASE_RESOLUTION = (1024, 1024)
_QWEN_IMAGE_FALLBACK_BASE_STEPS = 25
_QWEN_IMAGE_PER_PIXEL_STEP_MS: dict[str, dict[tuple[int, int], float]] = {
    hardware_profile: {
        resolution: (anchor_latency_s * 1000.0) / (resolution[0] * resolution[1] * anchor_steps)
        for resolution, (anchor_steps, anchor_latency_s) in anchors.items()
    }
    for hardware_profile, anchors in _QWEN_IMAGE_HARDWARE_ANCHORS_S.items()
}
_WAN2_2_FALLBACK_BASE_KEY = (1280, 720, 6, 80)
_WAN2_2_PER_PIXEL_FRAME_STEP_MS: dict[str, float] = {
    hardware_profile: (
        anchors[_WAN2_2_FALLBACK_BASE_KEY]
        * 1000.0
        / (
            _WAN2_2_FALLBACK_BASE_KEY[0]
            * _WAN2_2_FALLBACK_BASE_KEY[1]
            * _WAN2_2_FALLBACK_BASE_KEY[2]
            * _WAN2_2_FALLBACK_BASE_KEY[3]
        )
    )
    for hardware_profile, anchors in _WAN2_2_HARDWARE_ANCHORS_S.items()
    if _WAN2_2_FALLBACK_BASE_KEY in anchors
}

_WAN2_2_VAE_SCALE_FACTOR_TEMPORAL = 4


@dataclass(frozen=True)
class SuperP95LoadSnapshot:
    normal_load_s: float = 0.0
    sacrificial_load_s: float = 0.0


def _parse_bool_header(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


def _parse_float_header(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def apply_super_p95_request_headers(extra_args: dict[str, Any], headers: Mapping[str, str] | None) -> None:
    if not headers:
        return
    sacrificial = _parse_bool_header(headers.get(HEADER_SUPER_P95_SACRIFICIAL))
    if sacrificial is not None:
        extra_args[EXTRA_ARG_SUPER_P95_SACRIFICIAL] = sacrificial
    estimated_service_s = _parse_float_header(headers.get(HEADER_SUPER_P95_ESTIMATED_SERVICE_S))
    if estimated_service_s is not None and estimated_service_s >= 0.0:
        extra_args[EXTRA_ARG_SUPER_P95_ESTIMATED_SERVICE_S] = estimated_service_s


def build_super_p95_response_headers(snapshot: SuperP95LoadSnapshot) -> dict[str, str]:
    return {
        HEADER_SUPER_P95_NORMAL_LOAD_S: f"{snapshot.normal_load_s:.6f}",
        HEADER_SUPER_P95_SACRIFICIAL_LOAD_S: f"{snapshot.sacrificial_load_s:.6f}",
    }


def parse_super_p95_load_headers(headers: Mapping[str, str] | None) -> SuperP95LoadSnapshot | None:
    if not headers:
        return None

    normal_load_s = _parse_float_header(headers.get(HEADER_SUPER_P95_NORMAL_LOAD_S))
    sacrificial_load_s = _parse_float_header(headers.get(HEADER_SUPER_P95_SACRIFICIAL_LOAD_S))
    if normal_load_s is None or sacrificial_load_s is None:
        return None
    return SuperP95LoadSnapshot(
        normal_load_s=normal_load_s,
        sacrificial_load_s=sacrificial_load_s,
    )


def get_super_p95_request_metadata(extra_args: Mapping[str, Any] | None) -> tuple[bool, float | None]:
    if not extra_args:
        return False, None
    sacrificial = bool(extra_args.get(EXTRA_ARG_SUPER_P95_SACRIFICIAL, False))
    estimated_service_s = extra_args.get(EXTRA_ARG_SUPER_P95_ESTIMATED_SERVICE_S)
    if isinstance(estimated_service_s, (int, float)):
        return sacrificial, float(estimated_service_s)
    return sacrificial, None


def normalize_super_p95_hardware_profile(hardware_profile: str | None) -> str:
    normalized = (hardware_profile or DEFAULT_SUPER_P95_HARDWARE_PROFILE).strip().upper()
    if normalized in _KNOWN_SUPER_P95_HARDWARE_PROFILES:
        return normalized
    return DEFAULT_SUPER_P95_HARDWARE_PROFILE


def estimate_service_time_s(
    request: Any,
    hardware_profile: str | None = None,
) -> float:
    params = request.sampling_params
    frames = max(int(params.num_frames or 1), 1)
    if frames > 1:
        return _estimate_wan2_2_service_time_s(request, hardware_profile=hardware_profile)

    params = request.sampling_params
    width = int(params.width or params.resolution or _QWEN_IMAGE_FALLBACK_BASE_RESOLUTION[0])
    height = int(params.height or params.resolution or _QWEN_IMAGE_FALLBACK_BASE_RESOLUTION[1])
    steps = max(int(params.num_inference_steps or _QWEN_IMAGE_FALLBACK_BASE_STEPS), 1)

    profile = normalize_super_p95_hardware_profile(hardware_profile)
    per_pixel_step_ms = _QWEN_IMAGE_PER_PIXEL_STEP_MS[profile].get((width, height))
    if per_pixel_step_ms is None:
        per_pixel_step_ms = _QWEN_IMAGE_PER_PIXEL_STEP_MS[profile][_QWEN_IMAGE_FALLBACK_BASE_RESOLUTION]
    return per_pixel_step_ms * width * height * steps / 1000.0


def _estimate_wan2_2_service_time_s(
    request: Any,
    hardware_profile: str | None = None,
) -> float:
    params = request.sampling_params
    width = int(params.width or 854)
    height = int(params.height or 480)
    steps = max(int(params.num_inference_steps or 1), 1)
    frames = max(int(params.num_frames or 1), 1)

    profile = normalize_super_p95_hardware_profile(hardware_profile)
    anchors = _WAN2_2_HARDWARE_ANCHORS_S.get(profile, _WAN2_2_HARDWARE_ANCHORS_S[DEFAULT_SUPER_P95_HARDWARE_PROFILE])
    exact_match = anchors.get((width, height, steps, frames))
    if exact_match is not None:
        return exact_match

    frames = normalize_wan2_2_num_frames(frames)
    per_pixel_frame_step_ms = _WAN2_2_PER_PIXEL_FRAME_STEP_MS[profile]
    return per_pixel_frame_step_ms * width * height * steps * frames / 1000.0


def normalize_wan2_2_num_frames(num_frames: int | None) -> int:
    frames = max(int(num_frames or 1), 1)
    if frames % _WAN2_2_VAE_SCALE_FACTOR_TEMPORAL != 1:
        frames = frames // _WAN2_2_VAE_SCALE_FACTOR_TEMPORAL * _WAN2_2_VAE_SCALE_FACTOR_TEMPORAL + 1
    return max(frames, 1)
