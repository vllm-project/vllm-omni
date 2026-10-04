# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The fragmented MP4 stream can carry an AAC audio track next to the video."""

from __future__ import annotations

import io

import av
import numpy as np
import pytest

from vllm_omni.diffusion.utils.media_utils import FragmentedMP4Muxer
from vllm_omni.entrypoints.openai.video_api_utils import FragmentedMP4VideoEncoder

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _frames(count: int, height: int = 64, width: int = 96) -> np.ndarray:
    frames = np.zeros((count, height, width, 3), dtype=np.uint8)
    for index in range(count):
        frames[index, :, :, 0] = (index * 9) % 256
    return frames


def _sine(samples: int, rate: int = 32000) -> np.ndarray:
    t = np.arange(samples, dtype=np.float32) / rate
    return np.stack((np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 660 * t)), axis=1).astype(np.float32) * 0.2


def _probe(buffer: bytes) -> tuple[int, int, int]:
    """(stream count, decoded video frames, muxed audio samples).

    Audio is counted from the muxed AAC packets (1024 samples each): PyAV's
    decode iterator stops early on short fragmented streams, while ffmpeg
    decodes them completely.
    """
    with av.open(io.BytesIO(buffer)) as container:
        video = [stream for stream in container.streams if stream.type == "video"]
        audio = [stream for stream in container.streams if stream.type == "audio"]
        frames = sum(1 for _ in container.decode(video=0))
    samples = 0
    if audio:
        with av.open(io.BytesIO(buffer)) as container:
            samples = sum(1024 for packet in container.demux(container.streams.audio[0]) if packet.size)
    return len(video) + len(audio), frames, samples


@pytest.mark.parametrize("chunks", [1, 3])
def test_muxer_writes_video_and_audio_fragments(chunks: int) -> None:
    muxer = FragmentedMP4Muxer(width=96, height=64, fps=24, audio_sample_rate=32000, audio_channels=2)
    assert muxer.has_audio
    stream = b""
    for _ in range(chunks):
        stream += muxer.mux_video_frames(_frames(17))
        stream += muxer.mux_audio_samples(_sine(17 * 32000 // 24))
    stream += muxer.close()
    streams, frames, samples = _probe(stream)
    assert streams == 2
    assert frames == 17 * chunks
    # The first AAC priming length of content is dropped so that both
    # timelines start at zero; close() pads the tail one AAC frame past the
    # last video frame and the encoder rounds up to whole 1024-sample frames.
    expected = chunks * (17 * 32000 // 24) - 1024
    assert expected <= samples <= expected + 4 * 1024


def test_encoder_adds_audio_track_when_the_first_chunk_carries_audio() -> None:
    encoder = FragmentedMP4VideoEncoder(fps=24)
    stream = encoder.encode(_frames(34), _sine(34 * 32000 // 24), audio_sample_rate=32000)
    stream += encoder.encode(_frames(17), _sine(17 * 32000 // 24), audio_sample_rate=32000)
    stream += encoder.close()
    streams, frames, _ = _probe(stream)
    assert streams == 2 and frames == 51


def test_encoder_stays_video_only_without_audio() -> None:
    encoder = FragmentedMP4VideoEncoder(fps=24)
    stream = encoder.encode(_frames(8))
    stream += encoder.close()
    streams, frames, _ = _probe(stream)
    assert streams == 1 and frames == 8
    late = FragmentedMP4VideoEncoder(fps=24)
    late.encode(_frames(8))
    with pytest.raises(ValueError, match="audio track"):
        late.encode(_frames(8), _sine(1000), audio_sample_rate=32000)
    late.close()
