# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from fractions import Fraction
from io import BytesIO
from types import SimpleNamespace

import av
import numpy as np
import pytest

from tests.helpers.assertions import assert_video_diffusion_response, assert_video_valid

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _encode_clip(fps=12.5, *, audio_rate=44100, audio_layout="stereo"):
    output = BytesIO()
    with av.open(output, "w", format="mp4") as container:
        video = container.add_stream("mpeg4", rate=Fraction(str(fps)))
        video.width = video.height = 32
        video.pix_fmt = "yuv420p"
        audio = None
        if audio_rate is not None:
            audio = container.add_stream("aac", rate=audio_rate)
            audio.layout = audio_layout
        for _ in range(5):
            frame = av.VideoFrame.from_ndarray(np.zeros((32, 32, 3), dtype=np.uint8), format="rgb24")
            for packet in video.encode(frame):
                container.mux(packet)
        for packet in video.encode():
            container.mux(packet)
        if audio is not None:
            channels = len(audio.layout.channels)
            frame = av.AudioFrame.from_ndarray(
                np.zeros((channels, 1024), dtype=np.float32), format="fltp", layout=audio_layout
            )
            frame.sample_rate = audio_rate
            for packet in audio.encode(frame):
                container.mux(packet)
            for packet in audio.encode():
                container.mux(packet)
    return output.getvalue()


@pytest.mark.parametrize(
    ("fps", "audio_rate", "audio_layout", "error"),
    [
        (12.5, 44100, "stereo", None),
        (12.0, 44100, "stereo", "Expected fps"),
        (13.0, 44100, "stereo", "Expected fps"),
        (12.5, None, "stereo", "does not contain an audio stream"),
        (12.5, 48000, "stereo", "Expected audio sample rate"),
        (12.5, 44100, "mono", "Expected audio channels"),
    ],
)
def test_fixed_rate_video_audio_response_contract(fps, audio_rate, audio_layout, error):
    response = SimpleNamespace(videos=[_encode_clip(fps, audio_rate=audio_rate, audio_layout=audio_layout)])
    request_config = {
        "form_data": {"fps": "12.5", "num_frames": 5, "width": 32, "height": 32},
        "fps_tolerance": 0.01,
        "expected_audio": {"sample_rate": 44100, "channels": 2},
    }
    if error is None:
        assert_video_diffusion_response(response, request_config)
    else:
        with pytest.raises(AssertionError, match=error):
            assert_video_diffusion_response(response, request_config)


def test_video_response_retains_default_fps_tolerance():
    response = SimpleNamespace(videos=[_encode_clip(12.0, audio_rate=None)])
    assert_video_diffusion_response(response, {"form_data": {"fps": 12.5}})


@pytest.mark.parametrize("reported_fps", [0.0, float("nan"), float("inf")])
def test_video_response_rejects_invalid_reported_fps(monkeypatch, tmp_path, reported_fps):
    import cv2

    path = tmp_path / "clip.mp4"
    path.write_bytes(_encode_clip(audio_rate=None))
    real_capture = cv2.VideoCapture

    class InvalidFpsCapture:
        def __init__(self, path):
            self.capture = real_capture(path)

        def isOpened(self):
            return self.capture.isOpened()

        def get(self, prop):
            return reported_fps if prop == cv2.CAP_PROP_FPS else self.capture.get(prop)

        def release(self):
            self.capture.release()

    monkeypatch.setattr(cv2, "VideoCapture", InvalidFpsCapture)
    with pytest.raises(AssertionError, match="Invalid video fps"):
        assert_video_valid(path, fps=12.5, fps_tolerance=0.01)
