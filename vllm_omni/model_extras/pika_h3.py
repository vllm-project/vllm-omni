# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Lightweight Pika release identification and video API defaults."""

import json
from collections.abc import Mapping
from pathlib import Path

from vllm_omni.model_extras.video_generation import VideoGenerationDefaults

PIKA_MODEL_ID = "Pika-Labs/Pika-H3-Flex"
PIKA_REVISION = "3ff166447d5edb1406107ab8370c42000ce173e4"


def is_pika_h3_model(model: str) -> bool:
    if model.rstrip("/") == PIKA_MODEL_ID:
        return True
    root = Path(model)
    if not root.is_dir():
        return False
    for path in (root / "manifest.json", root / "fl2va/manifest.json", root / "ref2va/manifest.json"):
        if path.is_file():
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            if isinstance(manifest, dict) and manifest.get("model") == "pika-flex":
                return True
    return False


def get_pika_h3_video_generation_defaults(
    extra_body: Mapping[str, object] | None = None,
) -> VideoGenerationDefaults:
    del extra_body
    return VideoGenerationDefaults(
        width=832,
        height=480,
        num_frames=124,
        num_inference_steps=6,
        fps=24.0,
        flow_shift=12.0,
        dimension_multiple=32,
        output="pika.mp4",
        default_negative_prompt=None,
    )
