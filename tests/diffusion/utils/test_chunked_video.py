# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import io

import av
import numpy as np
import pytest
import torch

from vllm_omni.diffusion.utils.chunked_video import (
    ChunkedVideoMP4Session,
    chunk_to_uint8_frames,
    decode_to_mp4,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _FakeVAE:
    """Publish committed chunks the way a chunked-decode VAE does."""

    def __init__(self, value_range, *, batch=1, chunks=4, frames=2, height=16, width=32):
        self.chunk_value_range = value_range
        self._shape = (batch, 3, frames, height, width)
        self._chunks = chunks

    def decode_with_chunks(self, z, *, on_chunk):
        del z
        low, high = self.chunk_value_range
        for index in range(self._chunks):
            value = low + (high - low) * index / max(self._chunks - 1, 1)
            on_chunk(torch.full(self._shape, value))


def _decoded(data: bytes):
    with av.open(io.BytesIO(data)) as container:
        return np.stack([f.to_ndarray(format="rgb24") for f in container.decode(video=0)])


def test_quantization_maps_each_producer_range_onto_the_same_pixels():
    """A model's published range is what makes chunks comparable across models."""
    wan = torch.tensor([-1.0, 0.0, 1.0]).view(1, 1, 3, 1, 1).expand(1, 3, 3, 1, 1)
    h3 = torch.tensor([0.0, 0.5, 1.0]).view(1, 1, 3, 1, 1).expand(1, 3, 3, 1, 1)

    from_wan = chunk_to_uint8_frames(wan.contiguous(), (-1.0, 1.0))
    from_h3 = chunk_to_uint8_frames(h3.contiguous(), (0.0, 1.0))

    # Same physical pixels, different published ranges.
    assert np.array_equal(from_wan, from_h3)
    assert from_wan.reshape(-1, 3)[:, 0].tolist() == [0, 128, 255]


def test_quantization_rejects_a_degenerate_range():
    with pytest.raises(ValueError, match="increasing"):
        chunk_to_uint8_frames(torch.zeros(1, 3, 1, 2, 2), (1.0, 1.0))


@pytest.mark.parametrize("value_range", [(-1.0, 1.0), (0.0, 1.0)])
def test_decode_to_mp4_drives_any_declared_range(value_range):
    videos = decode_to_mp4(_FakeVAE(value_range), torch.zeros(1), fps=24, batch_frames=4)

    assert len(videos) == 1
    assert _decoded(videos[0]).shape[0] == 8


def test_decode_to_mp4_emits_one_container_per_batch_entry():
    videos = decode_to_mp4(_FakeVAE((-1.0, 1.0), batch=3), torch.zeros(1), fps=24)

    assert len(videos) == 3
    assert all(_decoded(v).shape[0] == 8 for v in videos)


def test_decode_to_mp4_rejects_a_vae_without_the_capability():
    class PlainVAE:
        def decode(self, z):
            del z

    with pytest.raises(TypeError, match="chunked VAE decode capability"):
        decode_to_mp4(PlainVAE(), torch.zeros(1), fps=24)


def test_decode_to_mp4_returns_nothing_for_a_rank_without_output():
    class SilentVAE:
        chunk_value_range = (-1.0, 1.0)

        def decode_with_chunks(self, z, *, on_chunk):
            del z, on_chunk

    assert decode_to_mp4(SilentVAE(), torch.zeros(1), fps=24) == []


def test_session_crops_the_decoder_padding_away():
    session = ChunkedVideoMP4Session(value_range=(0.0, 1.0), fps=24, crop=(8, 16))
    session.push(torch.zeros(1, 3, 4, 16, 32))

    assert _decoded(session.finish()[0]).shape[1:3] == (8, 16)


def test_session_muxes_one_waveform_per_batch_entry():
    audio = [np.zeros(8000, dtype=np.float32), np.full(8000, 0.5, dtype=np.float32)]
    session = ChunkedVideoMP4Session(value_range=(0.0, 1.0), fps=24, audio_waveforms=audio, audio_sample_rate=16000)
    session.push(torch.zeros(2, 3, 4, 16, 16))

    for data in session.finish():
        with av.open(io.BytesIO(data)) as container:
            assert container.streams.audio


def test_session_rejects_a_waveform_count_that_does_not_match_the_batch():
    session = ChunkedVideoMP4Session(value_range=(0.0, 1.0), fps=24, audio_waveforms=[np.zeros(8000, dtype=np.float32)])
    with pytest.raises(ValueError, match="one audio waveform per batch entry"):
        session.push(torch.zeros(2, 3, 4, 16, 16))


def test_session_rejects_a_non_positive_batch():
    with pytest.raises(ValueError, match="batch_frames"):
        ChunkedVideoMP4Session(value_range=(0.0, 1.0), fps=24, batch_frames=0)
