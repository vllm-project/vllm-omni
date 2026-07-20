# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
from PIL import Image

from vllm_omni.diffusion.models.lingbot_video.request_utils import (
    LINGBOT_RESOLUTION_PRESETS,
    LingBotGenerationMode,
    caption_from_lingbot_prompt,
    normalize_lingbot_request,
    resolve_lingbot_mode,
    resolve_lingbot_num_frames,
    resolve_lingbot_size,
    validate_lingbot_num_frames,
)
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _request(prompt, **sampling_overrides):
    return SimpleNamespace(
        prompt=prompt,
        sampling_params=OmniDiffusionSamplingParams(**sampling_overrides),
    )


def _normalize(prompt, **sampling_overrides):
    return normalize_lingbot_request(
        _request(prompt, **sampling_overrides),
        default_negative_prompt="video default",
        default_image_negative_prompt="image default",
    )


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ({"prompt": "still", "modalities": ["image"]}, LingBotGenerationMode.T2I),
        ({"prompt": "motion", "modalities": ["video"]}, LingBotGenerationMode.T2V),
        (
            {
                "prompt": "motion",
                "modalities": ["video"],
                "multi_modal_data": {"image": Image.new("RGB", (8, 8))},
            },
            LingBotGenerationMode.TI2V,
        ),
    ],
)
def test_resolve_lingbot_mode(prompt, expected):
    assert resolve_lingbot_mode(prompt) is expected


@pytest.mark.parametrize(
    ("prompt", "message"),
    [
        (
            {"prompt": "x", "modalities": ["image"], "multi_modal_data": {"image": Image.new("RGB", (8, 8))}},
            "does not accept",
        ),
        (
            {
                "prompt": "x",
                "modalities": ["video"],
                "multi_modal_data": {"image": [Image.new("RGB", (8, 8)), Image.new("RGB", (8, 8))]},
            },
            "exactly one",
        ),
        ({"prompt": "x", "modalities": ["video"], "multi_modal_data": {"video": [1]}}, "reference modalities"),
        ({"prompt": "x", "multi_modal_data": {"image": Image.new("RGB", (8, 8))}}, "explicitly set"),
        ({"prompt": "x", "modalities": ["audio"]}, "output modality"),
    ],
)
def test_resolve_lingbot_mode_rejects_invalid_media_combinations(prompt, message):
    with pytest.raises(ValueError, match=message):
        resolve_lingbot_mode(prompt)


def test_caption_from_lingbot_prompt_preserves_text_and_compact_json():
    assert caption_from_lingbot_prompt("plain text") == "plain text"
    assert caption_from_lingbot_prompt({"caption": "caption only", "duration": 4}) == "caption only"
    assert caption_from_lingbot_prompt({"caption": {"中文": True, "nested": {"value": 2}}}) == (
        '{"中文":true,"nested":{"value":2}}'
    )
    assert (
        caption_from_lingbot_prompt(
            {"scene": "海边", "camera": {"move": "pan"}, "duration": 5, "fps": 24, "size": "320x192"}
        )
        == '{"scene":"海边","camera":{"move":"pan"}}'
    )


@pytest.mark.parametrize("prompt", [{"duration": 4, "fps": 24}, {"caption": ""}, ""])
def test_caption_from_lingbot_prompt_rejects_empty_caption(prompt):
    with pytest.raises(ValueError, match="empty|must not be empty"):
        caption_from_lingbot_prompt(prompt)


@pytest.mark.parametrize("num_frames", [1, 5, 9, 121])
def test_validate_lingbot_num_frames_accepts_4n_plus_1(num_frames):
    assert validate_lingbot_num_frames(num_frames) == num_frames


@pytest.mark.parametrize("num_frames", [0, 2, 4, 120, 122])
def test_validate_lingbot_num_frames_rejects_invalid_values(num_frames):
    with pytest.raises(ValueError, match="num_frames"):
        validate_lingbot_num_frames(num_frames)


@pytest.mark.parametrize(
    ("duration", "fps", "expected"),
    [(4, 24, 97), (5, 24, 121), (8, 24, 193), (4, 12, 49), (4, 30, 121)],
)
def test_resolve_lingbot_num_frames_rounds_up(duration, fps, expected):
    assert resolve_lingbot_num_frames(duration, fps) == expected


@pytest.mark.parametrize(("duration", "fps"), [(0, 24), (-1, 24), (0.01, 24), (1, 0), (1, -1)])
def test_resolve_lingbot_num_frames_rejects_non_positive_or_empty_results(duration, fps):
    with pytest.raises(ValueError, match="duration|fps|at least one"):
        resolve_lingbot_num_frames(duration, fps)


def test_resolve_lingbot_size_supports_all_official_presets():
    for resolution, ratios in LINGBOT_RESOLUTION_PRESETS.items():
        for ratio, expected in ratios.items():
            assert resolve_lingbot_size(resolution=resolution, ratio=ratio) == expected


def test_resolve_lingbot_size_validates_sources_and_alignment():
    assert resolve_lingbot_size(size="320x192") == (192, 320)
    assert resolve_lingbot_size(width=320, height=192) == (192, 320)
    with pytest.raises(ValueError, match="provided together"):
        resolve_lingbot_size(width=320)
    with pytest.raises(ValueError, match="provided together"):
        resolve_lingbot_size(resolution="192p")
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_lingbot_size(width=320, height=192, resolution="192p", ratio="9:16")
    with pytest.raises(ValueError, match="multiples of 16"):
        resolve_lingbot_size(width=321, height=192)


def test_normalize_lingbot_request_uses_mode_defaults_and_preserves_empty_negative():
    t2i = _normalize(
        {"prompt": "still", "modalities": ["image"], "negative_prompt": ""},
        width=320,
        height=192,
        num_frames=1,
    )
    assert t2i.mode is LingBotGenerationMode.T2I
    assert t2i.num_frames == 1
    assert t2i.negative_prompt == ""

    t2v = _normalize(
        {"prompt": "motion", "modalities": ["video"]},
        width=320,
        height=192,
        num_frames=9,
    )
    assert t2v.mode is LingBotGenerationMode.T2V
    assert t2v.negative_prompt == "video default"


def test_normalize_lingbot_request_rejects_explicit_zero_flow_shift():
    with pytest.raises(ValueError, match="shift.*positive"):
        _normalize(
            {"prompt": "still", "modalities": ["image"]},
            width=320,
            height=192,
            num_frames=1,
            extra_args={"flow_shift": 0},
        )


def test_normalize_lingbot_request_rejects_explicit_zero_width():
    with pytest.raises(ValueError, match="width"):
        _normalize(
            {"prompt": "still", "modalities": ["image"]},
            width=0,
            height=192,
            num_frames=1,
        )


def test_normalize_lingbot_request_recomputes_seconds_and_prefers_explicit_frames():
    context = {
        "seconds": "4",
        "num_frames_explicit": False,
        "fps_explicit": True,
        "width_explicit": True,
        "height_explicit": True,
        "size_explicit": False,
    }
    config = _normalize(
        {"prompt": "motion", "modalities": ["video"]},
        width=320,
        height=192,
        num_frames=96,
        fps=24,
        frame_rate=24,
        extra_args={"_vllm_request_context": context},
    )
    assert config.num_frames == 97

    context["num_frames_explicit"] = True
    config = _normalize(
        {"prompt": "motion", "modalities": ["video"]},
        width=320,
        height=192,
        num_frames=121,
        fps=24,
        frame_rate=24,
        extra_args={"_vllm_request_context": context},
    )
    assert config.num_frames == 121


def test_normalize_lingbot_request_rejects_contract_conflicts():
    with pytest.raises(ValueError, match="text-to-image.*duration"):
        _normalize(
            {"prompt": {"caption": "still", "duration": 4}, "modalities": ["image"]},
            width=320,
            height=192,
            num_frames=1,
        )
    with pytest.raises(ValueError, match="text-to-image requires"):
        _normalize(
            {"prompt": "still", "modalities": ["image"]},
            width=320,
            height=192,
            num_frames=5,
        )
    with pytest.raises(ValueError, match="seconds.*duration"):
        _normalize(
            {"prompt": "motion", "modalities": ["video"]},
            width=320,
            height=192,
            num_frames=96,
            extra_args={
                "duration": 4,
                "_vllm_request_context": {
                    "seconds": "4",
                    "num_frames_explicit": False,
                    "fps_explicit": False,
                    "width_explicit": True,
                    "height_explicit": True,
                    "size_explicit": False,
                },
            },
        )


def test_normalize_lingbot_request_lets_preset_override_serving_defaults_only():
    config = _normalize(
        {"prompt": "motion", "modalities": ["video"]},
        width=480,
        height=480,
        num_frames=81,
        extra_args={
            "resolution": "192p",
            "ratio": "9:16",
            "_vllm_request_context": {
                "seconds": None,
                "num_frames_explicit": False,
                "fps_explicit": False,
                "width_explicit": False,
                "height_explicit": False,
                "size_explicit": False,
            },
        },
    )
    assert (config.height, config.width) == (192, 320)

    with pytest.raises(ValueError, match="mutually exclusive"):
        _normalize(
            {"prompt": "motion", "modalities": ["video"]},
            width=480,
            height=480,
            num_frames=81,
            extra_args={
                "resolution": "192p",
                "ratio": "9:16",
                "_vllm_request_context": {
                    "seconds": None,
                    "num_frames_explicit": False,
                    "fps_explicit": False,
                    "width_explicit": True,
                    "height_explicit": True,
                    "size_explicit": False,
                },
            },
        )
