# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest
from PIL import Image

from examples.offline_inference.image_to_video.image_to_video import build_image_to_video_prompt
from examples.offline_inference.text_to_image.text_to_image import build_text_to_image_prompt

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    ("negative_prompt", "expected_negative_prompt"),
    [(None, None), ("", ""), ("blurry", "blurry")],
)
def test_text_to_image_builds_canonical_prompt(
    negative_prompt: str | None,
    expected_negative_prompt: str | None,
) -> None:
    result = build_text_to_image_prompt("a red fox", negative_prompt)

    assert result["prompt"] == "a red fox"
    assert result["modalities"] == ["image"]
    if expected_negative_prompt is None:
        assert "negative_prompt" not in result
    else:
        assert result["negative_prompt"] == expected_negative_prompt


def test_image_to_video_builds_canonical_prompt() -> None:
    image = Image.new("RGB", (32, 16), "red")

    result = build_image_to_video_prompt(
        prompt="the fox turns toward the camera",
        negative_prompt="flicker",
        media_inputs={"image": image},
    )

    assert result == {
        "prompt": "the fox turns toward the camera",
        "modalities": ["video"],
        "multi_modal_data": {"image": image},
        "negative_prompt": "flicker",
    }
