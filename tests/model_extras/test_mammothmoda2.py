# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import pytest

from vllm_omni.model_extras import build_text_to_image_prompt

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.cpu,
    pytest.mark.parametrize(
        "model_class_name",
        ["MammothModa2DiTPipeline", "MammothModa2ForConditionalGeneration", "Mammothmoda2Model"],
    ),
]


@pytest.mark.parametrize(
    ("height", "width", "expected_height", "expected_width"),
    [
        (None, None, 1024, 1024),
        (None, 512, 1024, 512),
        (512, None, 512, 1024),
        (16, 16, 16, 16),
        (16, 32, 16, 32),
        (32, 16, 32, 16),
        (256, 256, 256, 256),
        (512, 768, 512, 768),
        (768, 512, 768, 512),
    ],
)
def test_text_to_image_dimensions(
    model_class_name: str, height: int | None, width: int | None, expected_height: int, expected_width: int
) -> None:
    result = build_text_to_image_prompt(
        model_class_name,
        {"prompt": "a cat", "modalities": ["image"]},
        height=height,
        width=width,
    )

    info = result["additional_information"]
    assert info["image_height"] == [expected_height]
    assert info["image_width"] == [expected_width]
    assert info["ar_height"] == [expected_height // 16]
    assert info["ar_width"] == [expected_width // 16]
    assert result["prompt"].endswith(f"<|image start|>{expected_width // 16}*{expected_height // 16}<|image token|>")


@pytest.mark.parametrize("dimension", ["height", "width"])
@pytest.mark.parametrize("value", [-16, 0, 1, 15, 17, 513])
def test_text_to_image_rejects_invalid_dimensions(model_class_name: str, dimension: str, value: int) -> None:
    dimensions = {"height": 512, "width": 512}
    dimensions[dimension] = value

    with pytest.raises(ValueError, match=rf"MammothModa2 {dimension} must be a positive multiple of 16, got {value}"):
        build_text_to_image_prompt(model_class_name, {"prompt": "a cat", "modalities": ["image"]}, **dimensions)


@pytest.mark.parametrize("dimension", ["height", "width"])
@pytest.mark.parametrize("value", [0, 513])
def test_text_to_image_rejects_invalid_dimension_when_other_is_omitted(
    model_class_name: str, dimension: str, value: int
) -> None:
    with pytest.raises(ValueError, match=rf"MammothModa2 {dimension} must be a positive multiple of 16, got {value}"):
        build_text_to_image_prompt(
            model_class_name,
            {"prompt": "a cat", "modalities": ["image"]},
            **{dimension: value},
        )
