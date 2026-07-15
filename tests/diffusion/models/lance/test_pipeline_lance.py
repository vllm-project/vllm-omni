# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
from PIL import Image

from vllm_omni.utils.video import normalize_decoded_video_frames

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class FramesWithMetadata(list[Image.Image]):
    def __init__(self, frames: list[Image.Image], **metadata: object) -> None:
        super().__init__(frames)
        for name, value in metadata.items():
            setattr(self, name, value)


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


def test_normalize_decoded_video_frames_uses_default_fps_for_plain_list_and_tuple():
    frames = [Image.new("RGB", (2, 2)), Image.new("RGB", (2, 2))]

    assert normalize_decoded_video_frames(frames, default_fps=24.0)[1] == 24.0
    assert normalize_decoded_video_frames(tuple(frames), default_fps=24.0)[1] == 24.0


@pytest.mark.parametrize(
    "fps",
    ["12.5", "invalid", 0, -1, float("nan"), float("inf")],
)
def test_normalize_decoded_video_frames_handles_fps_metadata(fps):
    frames = FramesWithMetadata([Image.new("RGB", (2, 2))] * 2, fps=fps)

    _, frame_rate = normalize_decoded_video_frames(frames, default_fps=24.0)

    expected = 12.5 if fps == "12.5" else 24.0
    assert frame_rate == expected


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
