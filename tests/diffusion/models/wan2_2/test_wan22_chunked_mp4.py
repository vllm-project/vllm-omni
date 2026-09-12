# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import io
from types import SimpleNamespace

import av
import pytest
import torch

from vllm_omni.diffusion.models.wan2_2.chunked_mp4 import (
    WAN_DEFAULT_OUTPUT_FPS,
    resolve_wan_output_fps,
    resolve_wan_preencode_batch_frames,
    resolve_wan_preencode_mp4,
    resolve_wan_video_codec_options,
    wan_preencoded_mp4_payload,
)
from vllm_omni.diffusion.utils.chunked_video import decode_to_mp4

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _FakeWanVAE:
    """Publish temporal chunks the way the Wan VAE seam does."""

    chunk_value_range = (-1.0, 1.0)

    def __init__(self, *, batch: int, chunks: int, frames_per_chunk: int, height: int, width: int):
        self.batch = batch
        self.chunks = chunks
        self.frames_per_chunk = frames_per_chunk
        self.height = height
        self.width = width
        self.chunk_calls = 0

    def decode_with_chunks(self, latents, *, on_chunk):
        del latents
        assert on_chunk is not None
        for index in range(self.chunks):
            self.chunk_calls += 1
            value = -1.0 + 2.0 * index / max(self.chunks - 1, 1)
            on_chunk(torch.full((self.batch, 3, self.frames_per_chunk, self.height, self.width), value))


def _decoded_frame_count(data: bytes) -> int:
    with av.open(io.BytesIO(data)) as container:
        return sum(1 for _ in container.decode(video=0))


def test_decode_wan_latents_to_mp4_emits_one_playable_mp4_per_batch_entry():
    vae = _FakeWanVAE(batch=2, chunks=6, frames_per_chunk=2, height=16, width=32)

    videos = decode_to_mp4(vae, torch.zeros(1), fps=24, batch_frames=4)

    assert len(videos) == 2
    for data in videos:
        assert data[:4] and _decoded_frame_count(data) == 12


def test_decode_wan_latents_to_mp4_coalesces_transfers_without_losing_frames():
    """batch_frames only changes how many transfers carry the same frames."""
    unbatched = _FakeWanVAE(batch=1, chunks=8, frames_per_chunk=1, height=16, width=32)
    batched = _FakeWanVAE(batch=1, chunks=8, frames_per_chunk=1, height=16, width=32)

    per_chunk = decode_to_mp4(unbatched, torch.zeros(1), fps=24, batch_frames=1)
    coalesced = decode_to_mp4(batched, torch.zeros(1), fps=24, batch_frames=4)

    assert _decoded_frame_count(per_chunk[0]) == _decoded_frame_count(coalesced[0]) == 8


def test_decode_wan_latents_to_mp4_returns_nothing_for_a_rank_without_output():
    class SilentVAE:
        chunk_value_range = (-1.0, 1.0)

        def decode_with_chunks(self, latents, *, on_chunk):
            del latents, on_chunk

    assert decode_to_mp4(SilentVAE(), torch.zeros(1), fps=24) == []


def test_decode_wan_latents_to_mp4_rejects_a_non_positive_batch():
    with pytest.raises(ValueError, match="batch_frames"):
        decode_to_mp4(
            _FakeWanVAE(batch=1, chunks=1, frames_per_chunk=1, height=16, width=16), None, fps=24, batch_frames=0
        )


def test_decode_wan_latents_to_mp4_rejects_a_vae_without_the_capability():
    class PlainVAE:
        chunk_value_range = (-1.0, 1.0)

        def decode(self, z, return_dict=True):
            del z, return_dict

    with pytest.raises(TypeError, match="chunked VAE decode capability"):
        decode_to_mp4(PlainVAE(), torch.zeros(1), fps=24)


def test_preencode_flag_is_off_unless_the_request_asks_for_it():
    assert resolve_wan_preencode_mp4(SimpleNamespace(extra_args=None), output_type="np") is False
    assert resolve_wan_preencode_mp4(SimpleNamespace(extra_args={}), output_type="np") is False
    assert resolve_wan_preencode_mp4(SimpleNamespace(extra_args={"preencode_mp4": True}), output_type="np") is True


def test_preencode_rejects_combinations_it_cannot_serve():
    interpolating = SimpleNamespace(extra_args={"preencode_mp4": True}, enable_frame_interpolation=True)
    with pytest.raises(ValueError, match="enable_frame_interpolation"):
        resolve_wan_preencode_mp4(interpolating, output_type="np")

    with pytest.raises(ValueError, match="output_type"):
        resolve_wan_preencode_mp4(SimpleNamespace(extra_args={"preencode_mp4": True}), output_type="pil")


def test_output_fps_mirrors_the_serving_fallback():
    assert resolve_wan_output_fps(SimpleNamespace(fps=16)) == 16
    assert resolve_wan_output_fps(SimpleNamespace(fps=[16])) == 16
    assert resolve_wan_output_fps(SimpleNamespace(fps=None)) == WAN_DEFAULT_OUTPUT_FPS
    assert resolve_wan_output_fps(SimpleNamespace()) == WAN_DEFAULT_OUTPUT_FPS


def test_codec_options_reach_the_encoder_as_strings():
    params = SimpleNamespace(extra_args={"video_codec_options": {"preset": "ultrafast", "threads": 0}})
    assert resolve_wan_video_codec_options(params) == {"preset": "ultrafast", "threads": "0"}
    assert resolve_wan_video_codec_options(SimpleNamespace(extra_args={})) is None


def test_preencoded_payload_passes_bytes_through_and_ignores_tensors():
    assert wan_preencoded_mp4_payload(torch.zeros(1, 3, 2, 4, 4)) is None
    assert wan_preencoded_mp4_payload([b"one", b"two"]) == {"payload": {"video": [b"one", b"two"]}, "metadata": {}}
    assert wan_preencoded_mp4_payload(b"one") == {"payload": {"video": [b"one"]}, "metadata": {}}


@pytest.mark.parametrize("value", [1, 5, 33])
def test_request_batch_threshold_controls_transfers_without_losing_frames(monkeypatch, value):
    from vllm_omni.diffusion.utils import chunked_video

    transfers = []
    quantize = chunked_video.chunk_to_uint8_frames

    def capture(chunk, value_range):
        transfers.append(chunk.shape[2])
        return quantize(chunk, value_range)

    monkeypatch.setattr(chunked_video, "chunk_to_uint8_frames", capture)
    params = SimpleNamespace(extra_args={"preencode_mp4": True, "preencode_batch_frames": value})
    vae = _FakeWanVAE(batch=1, chunks=4, frames_per_chunk=3, height=16, width=32)
    videos = decode_to_mp4(vae, torch.zeros(1), fps=24, batch_frames=resolve_wan_preencode_batch_frames(params))

    assert transfers == {1: [3, 3, 3, 3], 5: [6, 6], 33: [12]}[value]
    assert _decoded_frame_count(videos[0]) == 12


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "17", None])
def test_request_batch_threshold_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="preencode_batch_frames"):
        resolve_wan_preencode_batch_frames(SimpleNamespace(extra_args={"preencode_batch_frames": value}))


def test_request_batch_threshold_preserves_model_defaults():
    params = SimpleNamespace(extra_args=None)
    assert resolve_wan_preencode_batch_frames(params) == 17
    assert resolve_wan_preencode_batch_frames(params, default=1) == 1
