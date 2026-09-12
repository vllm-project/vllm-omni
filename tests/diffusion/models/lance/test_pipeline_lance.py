# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from vllm_omni.diffusion.models.lance.lance_transformer import LanceBagel
from vllm_omni.diffusion.models.lance.pipeline_lance import LancePipeline
from vllm_omni.model_executor.models.utils import normalize_decoded_video_frames

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class FramesWithMetadata(list[Image.Image]):
    def __init__(self, frames: list[Image.Image], **metadata: object) -> None:
        super().__init__(frames)
        for name, value in metadata.items():
            setattr(self, name, value)


class StopAfterVideoPreprocessError(Exception):
    pass


def test_normalize_decoded_video_frames_preserves_rgb_pixels_and_fps():
    red = Image.new("RGB", (3, 2), color=(255, 0, 0))
    rgba = Image.new("RGBA", (3, 2), color=(0, 255, 0, 64))
    rgba_bytes = rgba.tobytes()

    video, fps = normalize_decoded_video_frames(FramesWithMetadata([red, rgba], fps=8), default_fps=24.0)

    assert video.shape == (2, 2, 3, 3)
    assert video.dtype == np.uint8
    assert video.flags.c_contiguous
    np.testing.assert_array_equal(video[0, 0, 0], [255, 0, 0])
    np.testing.assert_array_equal(video[1, 0, 0], [0, 255, 0])
    assert fps == 8.0
    assert rgba.mode == "RGBA"
    assert rgba.tobytes() == rgba_bytes


def test_normalize_decoded_video_frames_uses_frame_rate_when_fps_is_missing():
    frames = FramesWithMetadata([Image.new("RGB", (2, 2))] * 2, frame_rate=12)

    _, fps = normalize_decoded_video_frames(frames, default_fps=24.0)

    assert fps == 12.0


def test_normalize_decoded_video_frames_prefers_valid_fps_over_frame_rate():
    frames = FramesWithMetadata([Image.new("RGB", (2, 2))] * 2, fps=8, frame_rate=12)

    _, fps = normalize_decoded_video_frames(frames, default_fps=24.0)

    assert fps == 8.0


def test_normalize_decoded_video_frames_uses_default_fps_for_plain_list_and_tuple():
    frames = [Image.new("RGB", (2, 2)), Image.new("RGB", (2, 2))]

    assert normalize_decoded_video_frames(frames, default_fps=24.0)[1] == 24.0
    assert normalize_decoded_video_frames(tuple(frames), default_fps=24.0)[1] == 24.0


def test_normalize_decoded_video_frames_accepts_numeric_string_fps():
    frames = FramesWithMetadata([Image.new("RGB", (2, 2))] * 2, fps="12.5")

    _, frame_rate = normalize_decoded_video_frames(frames, default_fps=24.0)

    assert frame_rate == 12.5


@pytest.mark.parametrize(
    ("fps", "frame_rate", "expected"),
    [
        (float("nan"), 12, 12.0),
        (0, 15, 15.0),
        (-1, 8, 8.0),
        (float("inf"), 10, 10.0),
        ("invalid", "6.5", 6.5),
    ],
)
def test_normalize_decoded_video_frames_uses_valid_frame_rate_after_invalid_fps(fps, frame_rate, expected):
    frames = FramesWithMetadata([Image.new("RGB", (2, 2))] * 2, fps=fps, frame_rate=frame_rate)

    _, frame_rate = normalize_decoded_video_frames(frames, default_fps=24.0)

    assert frame_rate == expected


def test_normalize_decoded_video_frames_uses_default_when_both_metadata_values_are_invalid():
    frames = FramesWithMetadata([Image.new("RGB", (2, 2))] * 2, fps="invalid", frame_rate=0)

    _, frame_rate = normalize_decoded_video_frames(frames, default_fps=24.0)

    assert frame_rate == 24.0


@pytest.mark.parametrize(
    ("frames", "match"),
    [
        ([], "empty decoded video frame sequence"),
        ([Image.new("RGB", (2, 2)), object()], "must be a PIL.Image.Image"),
        (
            [Image.new("RGB", (2, 2)), Image.new("RGB", (3, 2))],
            "must have identical dimensions",
        ),
    ],
)
def test_normalize_decoded_video_frames_rejects_invalid_sequences(frames, match):
    with pytest.raises(ValueError, match=match):
        normalize_decoded_video_frames(frames, default_fps=24.0)


@pytest.mark.parametrize(
    ("frames", "match"),
    [
        ([Image.new("RGB", (0, 16))], r"index 0.*0x16"),
        ([Image.new("RGB", (16, 0))], r"index 0.*16x0"),
        ([Image.new("RGB", (0, 0))], r"index 0.*0x0"),
        (
            [Image.new("RGB", (16, 16)), Image.new("RGB", (0, 16))],
            r"index 1.*0x16",
        ),
    ],
)
def test_normalize_decoded_video_frames_rejects_zero_dimensions(frames, match):
    with pytest.raises(ValueError, match=match):
        normalize_decoded_video_frames(frames, default_fps=24.0)


@pytest.mark.parametrize(
    ("video_input", "expected_fps"),
    [
        (FramesWithMetadata([Image.new("RGB", (3, 2))] * 2, fps=8), 8.0),
        (tuple([Image.new("RGBA", (3, 2))] * 2), 12.0),
    ],
)
def test_lance_video_edit_normalizes_decoded_frame_sequences(monkeypatch, video_input, expected_fps):
    captured = {}

    def capture_preprocessed_input(video, fps):
        captured["video"] = video
        captured["fps"] = fps
        raise StopAfterVideoPreprocessError

    monkeypatch.setattr(LanceBagel, "_lance_video_preprocess", staticmethod(capture_preprocessed_input))
    request = SimpleNamespace(
        prompts=[
            {
                "modalities": ["video"],
                "user_text": "edit the video",
                "multi_modal_data": {"video": video_input},
                "extra_args": {"origin_fps": 12.0},
            }
        ],
        sampling_params=SimpleNamespace(extra_args={}, num_inference_steps=1),
    )

    with pytest.raises(StopAfterVideoPreprocessError):
        object.__new__(LancePipeline)._forward_video_edit(request)

    video = captured["video"]
    assert isinstance(video, np.ndarray)
    assert video.shape == (2, 2, 3, 3)
    assert video.dtype == np.uint8
    assert video.flags.c_contiguous
    assert captured["fps"] == expected_fps
