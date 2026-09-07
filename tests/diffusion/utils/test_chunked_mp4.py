# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import hashlib

import av
import numpy as np
import pytest

from vllm_omni.diffusion.utils.media_utils import ChunkedMP4Encoder, mux_av_video_audio_bytes

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _frames() -> np.ndarray:
    rng = np.random.default_rng(6872)
    return rng.integers(0, 256, size=(8, 16, 24, 3), dtype=np.uint8)


def test_chunked_mp4_is_byte_identical_to_whole_mux() -> None:
    frames = _frames()
    baseline = mux_av_video_audio_bytes(
        (av.VideoFrame.from_ndarray(frame, format="rgb24") for frame in frames),
        width=24,
        height=16,
        fps=24,
    )
    encoder = ChunkedMP4Encoder(width=24, height=16, fps=24, max_pending=2)
    for start in (0, 1, 5):
        stop = {0: 1, 1: 5, 5: len(frames)}[start]
        encoder.push(np.ascontiguousarray(frames[start:stop]))
    chunked = encoder.finish()
    assert chunked == baseline
    assert hashlib.sha256(chunked).digest() == hashlib.sha256(baseline).digest()

    audio = np.zeros((2, 320), dtype=np.float32)
    baseline_audio = mux_av_video_audio_bytes(
        (av.VideoFrame.from_ndarray(frame, format="rgb24") for frame in frames),
        width=24,
        height=16,
        fps=24,
        audio_waveform=audio,
        audio_sample_rate=32000,
    )
    encoder_audio = ChunkedMP4Encoder(
        width=24,
        height=16,
        fps=24,
        audio_waveform=audio,
        audio_sample_rate=32000,
    )
    encoder_audio.push(frames)
    assert encoder_audio.finish() == baseline_audio


def test_chunked_mp4_abort_joins_worker() -> None:
    encoder = ChunkedMP4Encoder(width=24, height=16, fps=24)
    encoder.push(_frames()[:1])
    encoder.abort()
    assert not encoder._thread.is_alive()
    with pytest.raises(RuntimeError, match="already closed"):
        encoder.push(_frames()[:1])


def test_chunked_mp4_validates_shape_and_dtype() -> None:
    encoder = ChunkedMP4Encoder(width=24, height=16, fps=24)
    with pytest.raises(ValueError, match="shape"):
        encoder.push(np.zeros((16, 24, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="dtype"):
        encoder.push(np.zeros((1, 16, 24, 3), dtype=np.float32))
    encoder.abort()
