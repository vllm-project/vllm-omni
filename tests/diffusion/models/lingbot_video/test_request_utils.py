# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
from PIL import Image

from vllm_omni.diffusion.models.lingbot_video.request_utils import (
    LingBotGenerationMode,
    caption_from_lingbot_prompt,
    normalize_lingbot_num_frames,
    normalize_lingbot_request,
    resolve_lingbot_mode,
    resolve_lingbot_num_frames,
    resolve_lingbot_output_dimensions,
    resolve_lingbot_size,
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


@pytest.mark.parametrize(
    ("num_frames", "expected"),
    [(1, 1), (2, 5), (4, 5), (5, 5), (96, 97), (98, 101), (121, 121)],
)
def test_normalize_lingbot_num_frames_rounds_up_to_4n_plus_1(num_frames, expected):
    assert normalize_lingbot_num_frames(num_frames) == expected


@pytest.mark.parametrize("num_frames", [0, -1, 1.5, "invalid"])
def test_normalize_lingbot_num_frames_rejects_invalid_values(num_frames):
    with pytest.raises(ValueError, match="num_frames"):
        normalize_lingbot_num_frames(num_frames)


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


@pytest.mark.parametrize(
    ("resolution", "ratio", "expected"),
    [
        ("192p", "1:1", (192, 192)),
        ("192p", "9:16", (192, 320)),
        ("192p", "16:9", (320, 192)),
        ("192p", "3:4", (192, 256)),
        ("192p", "4:3", (256, 192)),
        ("480p", "1:1", (480, 480)),
        ("480p", "9:16", (480, 832)),
        ("480p", "16:9", (832, 480)),
        ("480p", "3:4", (480, 640)),
        ("480p", "4:3", (640, 480)),
        ("720p", "1:1", (736, 736)),
        ("720p", "9:16", (736, 1280)),
        ("720p", "16:9", (1280, 736)),
        ("720p", "3:4", (736, 960)),
        ("720p", "4:3", (960, 736)),
        ("1080p", "1:1", (1088, 1088)),
        ("1080p", "9:16", (1088, 1920)),
        ("1080p", "16:9", (1920, 1088)),
        ("1080p", "3:4", (1088, 1440)),
        ("1080p", "4:3", (1440, 1088)),
        ("2k", "1:1", (1440, 1440)),
        ("2k", "9:16", (1440, 2560)),
        ("2k", "16:9", (2560, 1440)),
        ("2k", "3:4", (1440, 1920)),
        ("2k", "4:3", (1920, 1440)),
        ("4k", "1:1", (2176, 2176)),
        ("4k", "9:16", (2176, 3840)),
        ("4k", "16:9", (3840, 2176)),
        ("4k", "3:4", (2176, 2880)),
        ("4k", "4:3", (2880, 2176)),
    ],
)
def test_resolve_lingbot_size_supports_all_official_presets(resolution, ratio, expected):
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


def test_resolve_lingbot_output_dimensions_matches_request_priority():
    assert resolve_lingbot_output_dimensions(
        sampling_width=480,
        sampling_height=480,
        prompt_fields={"resolution": "720p", "ratio": "16:9"},
    ) == (736, 1280)

    assert resolve_lingbot_output_dimensions(
        sampling_width=480,
        sampling_height=480,
        prompt_fields={"width": 320, "height": 192},
    ) == (320, 192)

    assert resolve_lingbot_output_dimensions(
        sampling_width=320,
        sampling_height=192,
    ) == (320, 192)

    assert resolve_lingbot_output_dimensions() == (480, 480)


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


def test_normalize_lingbot_request_rounds_all_video_frame_counts():
    config = _normalize(
        {"prompt": "motion", "modalities": ["video"]},
        width=320,
        height=192,
        num_frames=96,
        fps=24,
        frame_rate=24,
    )
    assert config.num_frames == 97

    config = _normalize(
        {"prompt": "motion", "modalities": ["video"]},
        width=320,
        height=192,
        num_frames=121,
        fps=24,
        frame_rate=24,
    )
    assert config.num_frames == 121


def test_normalize_lingbot_request_prefers_explicit_frames_over_duration():
    online = _normalize(
        {"prompt": "motion", "modalities": ["video"]},
        width=320,
        height=192,
        num_frames=9,
        fps=24,
        extra_args={"duration": 5},
    )
    assert online.num_frames == 9

    offline = _normalize(
        {
            "prompt": {"caption": "motion", "duration": 5, "num_frames": 9},
            "modalities": ["video"],
        },
        width=320,
        height=192,
        fps=24,
    )
    assert offline.num_frames == 9


def test_normalize_lingbot_request_uses_duration_then_model_default():
    duration = _normalize(
        {"prompt": {"caption": "motion", "duration": 5}, "modalities": ["video"]},
        width=320,
        height=192,
        fps=24,
    )
    assert duration.num_frames == 121

    default = _normalize(
        {"prompt": "motion", "modalities": ["video"]},
        width=320,
        height=192,
    )
    assert default.num_frames == 81


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
            extra_args={"seconds": 4, "duration": 4},
        )


def test_normalize_lingbot_request_model_size_overrides_sampling_dimensions():
    # A prompt-provided size overrides the sampling width/height.
    config = _normalize(
        {"prompt": {"caption": "motion", "size": "320x192"}, "modalities": ["video"]},
        width=480,
        height=480,
        num_frames=81,
    )
    assert (config.height, config.width) == (192, 320)

    # Without a prompt size, the sampling dimensions are used.
    config = _normalize(
        {"prompt": "motion", "modalities": ["video"]},
        width=480,
        height=480,
        num_frames=81,
    )
    assert (config.height, config.width) == (480, 480)


def test_normalize_lingbot_request_reads_size_from_extra_args():
    """Regression: /v1/videos delivers `extra_params` via sampling.extra_args.

    The nightly TI2V case sends `extra_params={"size": "320x192"}` with no
    height/width form fields; the size used to be dropped and the video came
    out at the 480x480 default (build 2952, LingBot Video MoE Function Test).
    """
    config = _normalize(
        {
            "prompt": "motion",
            "modalities": ["video"],
            "multi_modal_data": {"image": Image.new("RGB", (320, 192))},
        },
        num_frames=9,
        extra_args={"size": "320x192"},
    )
    assert config.mode is LingBotGenerationMode.TI2V
    assert (config.height, config.width) == (192, 320)


def test_normalize_lingbot_request_extra_args_size_overrides_sampling_dimensions():
    config = _normalize(
        {"prompt": "motion", "modalities": ["video"]},
        width=480,
        height=480,
        num_frames=81,
        extra_args={"size": "320x192"},
    )
    assert (config.height, config.width) == (192, 320)


def test_normalize_lingbot_request_reads_width_height_from_extra_args():
    # Parity with the pre-#5311 pipeline, which honored extra_args height/width.
    config = _normalize(
        {"prompt": "motion", "modalities": ["video"]},
        num_frames=81,
        extra_args={"width": 320, "height": 192},
    )
    assert (config.height, config.width) == (192, 320)


def test_resolve_lingbot_output_dimensions_reads_extra_fields():
    assert resolve_lingbot_output_dimensions(extra_fields={"size": "320x192"}) == (320, 192)
    assert resolve_lingbot_output_dimensions(
        sampling_width=480,
        sampling_height=480,
        extra_fields={"resolution": "720p", "ratio": "16:9"},
    ) == (736, 1280)
