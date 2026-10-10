# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The window client must distinguish video bytes from transport JSON."""

from __future__ import annotations

import io

import av
import numpy as np
import pytest
import requests

from vllm_omni.diffusion.models.seedvr2 import long_video
from vllm_omni.diffusion.utils.media_utils import mux_video_audio_bytes

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.mark.parametrize("content_type", ["application/json", "", "application/octet-stream"])
def test_restore_rejects_non_video_response_before_decoding(mocker, content_type: str) -> None:
    response = requests.Response()
    response.status_code = 200
    response.headers["Content-Type"] = content_type
    response._content = b'{"data":[{"b64_json":"not-video-bytes"}]}'
    mocker.patch.object(long_video.requests, "post", return_value=response)
    mocker.patch.object(long_video, "_segment", return_value=b"window")
    decoder = mocker.patch.object(long_video.av, "open", side_effect=AssertionError("JSON reached video decoder"))

    with pytest.raises(long_video.JobError, match="transport_mode='bytes'"):
        long_video._restore([object()], 32, 32, 0, "none", 8000, "")

    decoder.assert_not_called()


def test_restore_preserves_video_bytes_response(mocker) -> None:
    pixels = np.zeros((5, 32, 32, 3), dtype=np.uint8)
    encoded = mux_video_audio_bytes(pixels, fps=24)
    with av.open(io.BytesIO(encoded)) as container:
        frames = list(container.decode(video=0))
    response = requests.Response()
    response.status_code = 200
    response.headers["Content-Type"] = "video/mp4"
    response._content = encoded
    mocker.patch.object(long_video.requests, "post", return_value=response)
    mocker.patch.object(long_video, "_segment", return_value=b"window")

    restored = long_video._restore(frames, 32, 32, 0, "none", 8000, "")

    assert np.stack(restored).shape == pixels.shape
    assert np.stack(restored).dtype == np.uint8
