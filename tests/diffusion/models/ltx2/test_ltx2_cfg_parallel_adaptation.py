from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import LTX2Pipeline

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_pipeline(sequence_parallel_size: int = 1) -> LTX2Pipeline:
    pipeline = object.__new__(LTX2Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.audio_vae_temporal_compression_ratio = 4
    pipeline.audio_vae_mel_compression_ratio = 4
    pipeline.od_config = SimpleNamespace(parallel_config=SimpleNamespace(sequence_parallel_size=sequence_parallel_size))
    # Mock audio_vae with identity normalization (mean=0, std=1) so
    # _normalize_audio_latents is a no-op and test values are preserved.
    pipeline.audio_vae = SimpleNamespace(
        latents_mean=torch.tensor(0.0),
        latents_std=torch.tensor(1.0),
    )
    return pipeline


def test_prepare_audio_latents_pads_packed_sequence_dim_for_provided_latents():
    pipeline = _make_pipeline(sequence_parallel_size=4)
    latents = torch.arange(40, dtype=torch.float32).view(1, 10, 4)

    padded, original_num_frames, padded_num_frames = pipeline.prepare_audio_latents(
        batch_size=1,
        num_channels_latents=2,
        num_mel_bins=8,
        audio_latent_length=10,
        dtype=torch.float32,
        device=torch.device("cpu"),
        latents=latents,
    )

    assert original_num_frames == 10
    assert padded_num_frames == 12
    assert padded.shape == (1, 12, 4)
    torch.testing.assert_close(padded[:, :10], latents)
    torch.testing.assert_close(padded[:, 10:], torch.zeros(1, 2, 4))


def test_unpad_audio_latents_restores_original_frames_before_unpack():
    pipeline = _make_pipeline()
    original = torch.arange(40, dtype=torch.float32).view(1, 10, 4)
    padded = torch.cat([original, torch.full((1, 2, 4), 999.0)], dim=1)

    unpadded = pipeline._unpad_audio_latents(padded, 10)
    unpacked = pipeline._unpack_audio_latents(unpadded, latent_length=10, num_mel_bins=2)
    expected = pipeline._unpack_audio_latents(original, latent_length=10, num_mel_bins=2)

    assert unpacked.shape == (1, 2, 10, 2)
    assert not (unpacked == 999.0).any()
    torch.testing.assert_close(unpacked, expected)
