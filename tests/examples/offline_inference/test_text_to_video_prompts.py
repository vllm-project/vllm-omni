# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest
import torch

from examples.offline_inference.text_to_video.text_to_video import (
    _MODEL_PRESETS,
    _detect_preset,
    _normalize_float_tensor,
    build_text_to_video_prompt,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    ("negative_prompt", "expected_negative_prompt"),
    [(None, None), ("", ""), ("blurry", "blurry")],
)
def test_text_to_video_builds_canonical_prompt(
    negative_prompt: str | None,
    expected_negative_prompt: str | None,
) -> None:
    result = build_text_to_video_prompt("a robot waves", negative_prompt)

    assert result["prompt"] == "a robot waves"
    assert result["modalities"] == ["video"]
    if expected_negative_prompt is None:
        assert "negative_prompt" not in result
    else:
        assert result["negative_prompt"] == expected_negative_prompt


@pytest.mark.parametrize(
    ("model", "model_class_name", "preset_name"),
    [
        ("robbyant/lingbot-video-dense-1.3b", None, "lingbot"),
        ("/models/custom-checkpoint", "LingBotVideoPipeline", "lingbot"),
        ("Lightricks/LTX-2", "LTX2DistilledPipeline", "ltx2_distilled"),
        ("BestWishYsh/Helios-Distilled", None, "helios"),
        ("/models/cosmos-edge", "Cosmos3EdgeVFMTransformer", "cosmos3_edge"),
        ("/models/vace", "WanVACEPipeline", "vace"),
    ],
)
def test_detect_preset_is_scoped_to_model_family(
    model: str,
    model_class_name: str | None,
    preset_name: str,
) -> None:
    assert _detect_preset(model, model_class_name) is _MODEL_PRESETS[preset_name]


def test_lingbot_preset_matches_lingbot_defaults() -> None:
    assert _MODEL_PRESETS["lingbot"] == {
        "model_class_name": "LingBotVideoPipeline",
        "height": 192,
        "width": 320,
        "num_frames": 9,
        "num_inference_steps": 2,
        "guidance_scale": 3.0,
        "fps": 24,
        "flow_shift": 3.0,
        "output": "lingbot_video_output.mp4",
    }


def test_normalize_float_tensor_preserves_zero_to_one_frames() -> None:
    frames = torch.tensor([0.0, 0.25, 1.0, 1.5])

    assert torch.equal(_normalize_float_tensor(frames, "zero_to_one"), torch.tensor([0.0, 0.25, 1.0, 1.0]))


def test_normalize_float_tensor_maps_negative_one_to_one_frames() -> None:
    frames = torch.tensor([-1.5, -1.0, 0.0, 1.0, 1.5])

    assert torch.equal(
        _normalize_float_tensor(frames, "negative_one_to_one"),
        torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0]),
    )


def test_normalize_float_tensor_uses_one_contract_for_ambiguous_frames() -> None:
    frames = [torch.tensor([0.0, 0.5, 1.0]), torch.tensor([-1.0, 0.0, 1.0])]

    normalized = [_normalize_float_tensor(frame, "negative_one_to_one") for frame in frames]

    assert torch.equal(normalized[0], torch.tensor([0.5, 0.75, 1.0]))
    assert torch.equal(normalized[1], torch.tensor([0.0, 0.5, 1.0]))
