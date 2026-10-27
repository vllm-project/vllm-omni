# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Reference service estimates for normalized diffusion requests.

Qwen anchors depend on the selected Ascend profile. Wan uses the same reference
anchors for 910B2 and 910B3; these are not separately calibrated for each SKU.
Inputs are normalized sampling parameters, never HTTP.
"""

from __future__ import annotations

import math
from typing import Any

_QWEN_ANCHORS = {
    "910B2": {(512, 512): (20, 8.60), (768, 768): (20, 8.94), (1024, 1024): (25, 14.22), (1536, 1536): (35, 43.22)},
    "910B3": {(512, 512): (20, 8.64), (768, 768): (20, 8.64), (1024, 1024): (25, 14.22), (1536, 1536): (35, 49.34)},
}
# Shared Wan reference estimates for 910B2 and 910B3, not per-SKU calibration.
_WAN_ANCHORS = {
    "910B2": {(848, 480, 3, 81): 38.07, (848, 480, 4, 121): 71.34, (1280, 720, 6, 81): 119.71},
    "910B3": {(848, 480, 3, 81): 38.07, (848, 480, 4, 121): 71.34, (1280, 720, 6, 81): 119.71},
}


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a normalized positive integer")
    return value


def validate_service_time_profile(model_class_name: str | None, hardware_profile: str | None) -> str:
    """Reject unsupported model/profile pairs and return the validated profile."""
    if hardware_profile is None or hardware_profile not in _QWEN_ANCHORS:
        raise ValueError("central estimation requires hardware_profile '910B2' or '910B3'")
    if model_class_name not in {"QwenImagePipeline", "WanPipeline", "Wan22Pipeline"}:
        raise ValueError(f"Central scheduling is not supported for model class {model_class_name!r}")
    return hardware_profile


def estimate_service_time_s(sampling_params: Any, model_class_name: str, hardware_profile: str | None) -> float:
    """Estimate a Qwen image or Wan text-to-video request without mutation.

    Defaults match the native Qwen-Image / Wan2.2 T2V pipelines (spatial
    VAE factor 8, patch size 2). Custom VAE/model geometries are not calibrated.
    """
    hardware_profile = validate_service_time_profile(model_class_name, hardware_profile)
    if any(getattr(sampling_params, name, None) is not None for name in ("timesteps", "sigmas")):
        raise ValueError("central scheduling does not support custom timesteps or sigmas")
    qwen = model_class_name == "QwenImagePipeline"
    defaults = {
        "width": 1024 if qwen else 832,
        "height": 1024 if qwen else 480,
        "num_inference_steps": 50 if qwen else 40,
        "num_frames": 1 if qwen else 81,
    }
    values = {}
    for name, default in defaults.items():
        value = getattr(sampling_params, name, None)
        values[name] = _positive_int(default if value is None else value, name)
    width, height, steps = values["width"], values["height"], values["num_inference_steps"]
    frames = values["num_frames"]
    if qwen:
        if frames != 1:
            raise ValueError("QwenImagePipeline requires num_frames=1")
        width, height = max(16, width // 16 * 16), max(16, height // 16 * 16)
        area = width * height
        spatial_costs = sorted(
            (w * h, duration / reference_steps)
            for (w, h), (reference_steps, duration) in _QWEN_ANCHORS[hardware_profile].items()
        )
        lower_area, lower_cost = max((point for point in spatial_costs if point[0] <= area), default=spatial_costs[0])
        upper_area, upper_cost = min((point for point in spatial_costs if point[0] >= area), default=spatial_costs[-1])
        # Interpolate within the measured range; scale from an endpoint outside it.
        # Switching nearest anchors would introduce downward jumps between sizes.
        cost = (
            lower_cost * area / lower_area
            if lower_area == upper_area
            else lower_cost + (upper_cost - lower_cost) * (area - lower_area) / (upper_area - lower_area)
        )
        estimate = cost * steps
    else:
        wan_anchors = _WAN_ANCHORS[hardware_profile]
        # Match the standard T2V worker's spatial and temporal normalization.
        width, height = width // 16 * 16, height // 16 * 16
        if width == 0 or height == 0:
            raise ValueError("Wan dimensions must be at least 16 pixels")
        normalized_frames = frames if frames % 4 == 1 else frames // 4 * 4 + 1
        reference = min(
            wan_anchors,
            key=lambda key: (
                abs(width * height - key[0] * key[1]),
                abs(normalized_frames - key[3]),
                abs(steps - key[2]),
                key,
            ),
        )
        ref_width, ref_height, ref_steps, ref_frames = reference
        estimate = (
            wan_anchors[reference]
            * width
            * height
            * steps
            * normalized_frames
            / (ref_width * ref_height * ref_steps * ref_frames)
        )
    outputs = _positive_int(getattr(sampling_params, "num_outputs_per_prompt", 1), "num_outputs_per_prompt")
    if outputs != 1:
        raise ValueError("central scheduling currently supports one output per request")
    if not math.isfinite(estimate) or estimate <= 0:
        raise ValueError("service estimate must be finite and positive")
    return estimate
