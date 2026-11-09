# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Benchmark-local saving preserves encoded pictures and presentation times."""

import io
from fractions import Fraction

import av
import numpy as np
import pytest

from vllm_omni.benchmarks.realtime_video.media import finalize_streaming_video_bytes

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_remux_preserves_pixels_and_pts_without_product_media_patch():
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="mp4") as output:
        stream = output.add_stream("mpeg4", rate=16)
        stream.width, stream.height, stream.pix_fmt = 32, 24, "yuv420p"
        for index in range(3):
            frame = av.VideoFrame.from_ndarray(np.full((24, 32, 3), 40 + index * 40, dtype=np.uint8), format="rgb24")
            frame.pts, frame.time_base = index, Fraction(1, 16)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    original = buffer.getvalue()
    saved = finalize_streaming_video_bytes(original, input_format="m4s", fps=12)

    def decoded(data):
        with av.open(io.BytesIO(data)) as container:
            return [
                (frame.pts * frame.time_base, frame.to_ndarray(format="rgb24")) for frame in container.decode(video=0)
            ]

    first, second = decoded(original), decoded(saved)
    assert len(first) == len(second) == 3
    for (time_a, pixels_a), (time_b, pixels_b) in zip(first, second):
        assert time_a == time_b
        np.testing.assert_array_equal(pixels_a, pixels_b)
