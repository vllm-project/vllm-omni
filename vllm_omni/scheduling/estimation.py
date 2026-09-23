# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Calibrated service estimates for normalized diffusion requests.

Anchors retain the measurements used by the dispatcher prototype. They are
specific to the explicitly selected Ascend profile, not hardware-independent
latency predictions. Inputs are normalized sampling parameters, never HTTP.
"""

from __future__ import annotations

import math
from typing import Any

_QWEN_ANCHORS = {
    "910B2": {(512, 512): (20, 8.60), (768, 768): (20, 8.94), (1024, 1024): (25, 14.22), (1536, 1536): (35, 43.22)},
    "910B3": {(512, 512): (20, 8.64), (768, 768): (20, 8.64), (1024, 1024): (25, 14.22), (1536, 1536): (35, 49.34)},
}
_WAN_ANCHORS = {(854, 480, 3, 80): 38.07, (854, 480, 4, 120): 71.34, (1280, 720, 6, 80): 119.71}


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a normalized positive integer")
    return value


def estimate_service_time_s(sampling_params: Any, model_class_name: str, hardware_profile: str) -> float:
    """Estimate a Qwen image or Wan text-to-video request without mutation.

    Defaults match the calibrated Qwen-Image / Wan2.2 T2V pipelines (spatial
    VAE factor 8, patch size 2). Custom VAE/model geometries are not calibrated.
    """
    if hardware_profile not in _QWEN_ANCHORS:
        raise ValueError("tail-aware estimation requires hardware_profile '910B2' or '910B3'")
    if model_class_name not in {"QwenImagePipeline", "WanPipeline", "Wan22Pipeline"}:
        raise ValueError(f"Unsupported tail-aware model: {model_class_name!r}")
    if any(getattr(sampling_params, name, None) is not None for name in ("timesteps", "sigmas")):
        raise ValueError("tail-aware scheduling does not support custom timesteps or sigmas")
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
        anchors = _QWEN_ANCHORS[hardware_profile]
        base_width, base_height = (width, height) if (width, height) in anchors else (1024, 1024)
        anchor_steps, anchor_s = anchors[(base_width, base_height)]
        estimate = anchor_s * width * height * steps / (base_width * base_height * anchor_steps)
    else:
        exact = _WAN_ANCHORS.get((width, height, steps, frames))
        # The original exact 80/120-frame anchors precede VAE frame rounding.
        width, height = width // 16 * 16, height // 16 * 16
        if width == 0 or height == 0:
            raise ValueError("Wan dimensions must be at least 16 pixels")
        normalized_frames = frames if frames % 4 == 1 else frames // 4 * 4 + 1
        estimate = (
            exact if exact is not None else 119.71 * width * height * steps * normalized_frames / (1280 * 720 * 6 * 80)
        )
    outputs = _positive_int(getattr(sampling_params, "num_outputs_per_prompt", 1), "num_outputs_per_prompt")
    if outputs != 1:
        raise ValueError("tail-aware scheduling currently supports one output per request")
    if not math.isfinite(estimate) or estimate <= 0:
        raise ValueError("service estimate must be finite and positive")
    return estimate
