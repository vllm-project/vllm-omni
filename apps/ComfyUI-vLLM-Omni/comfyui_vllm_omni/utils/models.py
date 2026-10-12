# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
import re
from typing import Any

from .logger import get_logger
from .types import Modality, ModelMode, Spec

logger = get_logger(__name__)


def _bagel_payload_preprocessor(payload: dict) -> dict:
    try:
        for message in payload["messages"]:
            for content in message["content"]:
                if content["type"] == "text":
                    content["text"] = "<|im_start|>" + content["text"] + "<|im_end|>"
    except (KeyError, TypeError):
        raise RuntimeError("Internal Error: malformatted BAGEL payload")
    return payload


def _qwen25_payload_preprocessor(payload: dict) -> dict:
    if payload["messages"][0]["role"] != "system":
        payload["messages"] = [
            {
                "role": "system",
                "content": (
                    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group,"
                    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
                ),
            },
            *payload["messages"],
        ]
    return payload


MINIMAX_H3_ASPECT_RATIOS = {
    "21:9": 21.0 / 9.0,
    "16:9": 16.0 / 9.0,
    "4:3": 4.0 / 3.0,
    "1:1": 1.0,
    "3:4": 3.0 / 4.0,
    "9:16": 9.0 / 16.0,
}


def _nearest_minimaxh3_aspect_ratio(width: int, height: int) -> str:
    """Pick the supported named ratio closest to the requested frame size."""
    target = float(width) / float(height)
    return min(MINIMAX_H3_ASPECT_RATIOS, key=lambda name: abs(MINIMAX_H3_ASPECT_RATIOS[name] - target))


def _minimaxh3_params_builder(
    model_params: dict[str, Any],
    *,
    extra_params: dict[str, Any],
    width: int | None = None,
    height: int | None = None,
) -> dict[str, Any]:
    """Build multipart form fields for MiniMax-H3 from model_params + routed task."""
    params = dict(model_params)
    params.pop("type", None)
    form_fields: dict[str, Any] = {}
    merged_extra_params: dict[str, Any] = dict(extra_params or {})

    if "flow_shift" in params:
        form_fields["flow_shift"] = params.pop("flow_shift")
    for key in ("audio_flow_shift",):
        if key in params:
            merged_extra_params[key] = params.pop(key)
    if params:
        logger.warning("Unused MiniMax-H3 model params ignored: %s", sorted(params))

    # H3 refuses t2va without a named aspect ratio, and the Generate Video node
    # carries width/height rather than a ratio. fl2va takes its ratio from the
    # input image and ref2va defaults server-side, so only t2va needs this.
    if merged_extra_params.get("task") == "t2va" and "aspect_ratio" not in merged_extra_params and width and height:
        merged_extra_params["aspect_ratio"] = _nearest_minimaxh3_aspect_ratio(width, height)

    if merged_extra_params:
        form_fields["extra_params"] = json.dumps(merged_extra_params, ensure_ascii=False)
    return form_fields


_MODEL_PIPELINE_SPECS: dict[str, Spec] = {
    r"BAGEL-7B-MoT": {
        "stages": [
            "diffusion"  # The vLLM-Omni interface treats it as a single-stage diffusion model
        ],
        "modes": [
            {
                "mode": ModelMode.UNDERSTANDING,
                "input_modalities": [Modality.TEXT, Modality.IMAGE],
            }
        ],
        "payload_preprocessor": _bagel_payload_preprocessor,
    },
    r"Qwen2.5-Omni*": {
        "stages": ["autoregression", "autoregression", "autoregression"],
        "payload_preprocessor": _qwen25_payload_preprocessor,
        "modes": [
            {
                "mode": ModelMode.UNDERSTANDING,
                "input_modalities": [
                    Modality.TEXT,
                    Modality.IMAGE,
                    Modality.VIDEO,
                    Modality.AUDIO,
                ],
            }
        ],
    },
    r"Qwen3-Omni*": {
        "stages": ["autoregression", "autoregression", "autoregression"],
        "modes": [
            {
                "mode": ModelMode.UNDERSTANDING,
                "input_modalities": [
                    Modality.TEXT,
                    Modality.IMAGE,
                    Modality.VIDEO,
                    Modality.AUDIO,
                ],
            }
        ],
    },
    r"MiniMax-H3(/FL2VA|/Ref2VA)?": {
        "stages": ["diffusion"],
        "modes": [
            {
                "mode": ModelMode.VIDEO_GENERATION,
                "input_modalities": [
                    Modality.TEXT,
                    Modality.IMAGE,
                    Modality.VIDEO,
                    Modality.AUDIO,
                ],
            }
        ],
        "params_builder": _minimaxh3_params_builder,
    },
}
# Convert dict keys to regex patterns
MODEL_PIPELINE_SPECS: dict[re.Pattern, Spec] = {}
for k, v in _MODEL_PIPELINE_SPECS.items():
    MODEL_PIPELINE_SPECS[re.compile(k)] = v
del _MODEL_PIPELINE_SPECS


def lookup_model_spec(model: str) -> tuple[Spec | None, str | None]:
    normalized = model.rstrip("/").lstrip("/")
    lookup_key = normalized.split("/", 1)[-1]
    for pattern, spec in MODEL_PIPELINE_SPECS.items():
        if pattern.search(lookup_key):
            return spec, pattern.pattern
    return None, None


# ============== DEMONSTRATION ==============

if __name__ == "__main__":
    test_paths = [
        "Qwen/Qwen2.5-Omni-7B",
        "MyModels/Qwen2.5-Omni-3B",
        "/root/home/Qwen2.5-Omni-7B",
        "Qwen/Qwen3-Omni",
        "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "Custom/Path/UnknownModel-Instruct",
        "Not/Matching/Anything",
        "/path/to/MiniMax-H3/FL2VA",
        "/path/to/MiniMax-H3/Ref2VA",
        "MiniMaxAI/MiniMax-H3",
    ]

    test_payload = {"messages": [{"role": "user", "content": "prompt"}]}

    print("Testing registry lookups:\n")
    for path in test_paths:
        spec, _ = lookup_model_spec(path)
        if spec:
            if preprocessor := spec.get("payload_preprocessor"):
                result = preprocessor(test_payload)
                print(f"✓ {path:<40} → {result}")
            elif params_builder := spec.get("params_builder"):
                result = params_builder(
                    {"audio_flow_shift": 3.0, "flow_shift": 12.0, "type": "minimax_h3"},
                    extra_params={"task": "t2va"},
                )
                print(f"✓ {path:<40} → {result}")
            else:
                print(f"✓ {path:<40} → No preprocessor/params_builder")
        else:
            print(f"✗ {path:<40} → No match")
