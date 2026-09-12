# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest

from examples.offline_inference.speech_to_video.speech_to_video import validate_media_inputs

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_JSON_CASE = {"input_json": "multi_example_1.json"}


@pytest.mark.parametrize(
    ("model_type", "image", "audio", "extra_body"),
    [
        ("wan-s2v", "ref.png", "speech.wav", {}),
        # A LongCat JSON case supplies the reference image and speaker tracks.
        ("longcat-video-avatar", None, None, _JSON_CASE),
        # AT2V is audio-only, whether the stage is explicit or inferred.
        ("longcat-video-avatar", None, "speech.wav", {"stage": "at2v"}),
        ("longcat-video-avatar", None, "speech.wav", {}),
        ("longcat-video-avatar", "ref.png", "speech.wav", {"stage": "ai2v"}),
    ],
)
def test_validate_media_inputs_accepts_supported_combinations(
    model_type: str,
    image: str | None,
    audio: str | None,
    extra_body: dict,
) -> None:
    validate_media_inputs(model_type, image=image, audio=audio, extra_body=extra_body)


@pytest.mark.parametrize(
    ("model_type", "image", "audio", "extra_body", "expected"),
    [
        ("wan-s2v", None, "speech.wav", {}, "--image is required"),
        ("wan-s2v", "ref.png", None, {}, "--audio is required"),
        # input_json is a LongCat concept: it must not relax the Wan checks.
        ("wan-s2v", None, "speech.wav", _JSON_CASE, "--image is required"),
        ("wan-s2v", None, None, _JSON_CASE, "--audio is required"),
        ("longcat-video-avatar", None, "speech.wav", {"stage": "ai2v"}, "--image is required"),
        ("longcat-video-avatar", "ref.png", None, {}, "--audio is required"),
    ],
)
def test_validate_media_inputs_rejects_missing_media(
    model_type: str,
    image: str | None,
    audio: str | None,
    extra_body: dict,
    expected: str,
) -> None:
    with pytest.raises(ValueError, match=expected):
        validate_media_inputs(model_type, image=image, audio=audio, extra_body=extra_body)
