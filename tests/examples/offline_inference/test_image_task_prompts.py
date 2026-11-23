# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import functools
import importlib.util

import pytest
from PIL import Image

from tests.examples.helpers import EXAMPLES

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@functools.cache
def _load_example_module(relative_path: str, module_name: str):
    path = EXAMPLES / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("negative_prompt", "expected_negative_prompt"),
    [(None, None), ("", ""), ("blurry", "blurry")],
)
def test_text_to_image_builds_canonical_prompt(
    negative_prompt: str | None,
    expected_negative_prompt: str | None,
) -> None:
    mod = _load_example_module(
        "offline_inference/text_to_image/text_to_image.py",
        "text_to_image_example",
    )
    result = mod.build_text_to_image_prompt("a red fox", negative_prompt)

    assert result["prompt"] == "a red fox"
    assert result["modalities"] == ["image"]
    if expected_negative_prompt is None:
        assert "negative_prompt" not in result
    else:
        assert result["negative_prompt"] == expected_negative_prompt


def test_image_to_video_builds_canonical_prompt() -> None:
    mod = _load_example_module(
        "offline_inference/image_to_video/image_to_video.py",
        "image_to_video_example",
    )
    image = Image.new("RGB", (32, 16), "red")

    result = mod.build_image_to_video_prompt(
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


@pytest.mark.parametrize(
    ("relative_path", "module_name"),
    [
        ("offline_inference/text_to_video/text_to_video.py", "text_to_video_output_example"),
        ("offline_inference/image_to_video/image_to_video.py", "image_to_video_output_example"),
    ],
)
@pytest.mark.parametrize("output_type", ["video", "image"])
def test_video_examples_accept_semantic_and_legacy_output_types(
    relative_path: str,
    module_name: str,
    output_type: str,
) -> None:
    mod = _load_example_module(relative_path, module_name)

    mod._validate_video_output_type(output_type)


@pytest.mark.parametrize(
    ("relative_path", "module_name"),
    [
        ("offline_inference/text_to_video/text_to_video.py", "text_to_video_invalid_output_example"),
        ("offline_inference/image_to_video/image_to_video.py", "image_to_video_invalid_output_example"),
    ],
)
def test_video_examples_reject_unrelated_output_types(
    relative_path: str,
    module_name: str,
) -> None:
    mod = _load_example_module(relative_path, module_name)

    with pytest.raises(ValueError, match="Unexpected output type 'audio'"):
        mod._validate_video_output_type("audio")
