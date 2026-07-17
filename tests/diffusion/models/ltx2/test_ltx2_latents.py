from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.ltx2 import ltx2_latents as latent_ops
from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import LTX2Pipeline, LTX23Pipeline

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_pipeline(pipeline_cls, sequence_parallel_size: int = 1):
    pipeline = object.__new__(pipeline_cls)
    torch.nn.Module.__init__(pipeline)
    pipeline.audio_vae_temporal_compression_ratio = 4
    pipeline.audio_vae_mel_compression_ratio = 4
    pipeline.od_config = SimpleNamespace(parallel_config=SimpleNamespace(sequence_parallel_size=sequence_parallel_size))
    # Mock audio_vae with identity normalization (mean=0, std=1).
    pipeline.audio_vae = SimpleNamespace(
        latents_mean=torch.tensor(0.0),
        latents_std=torch.tensor(1.0),
    )
    return pipeline


def test_prepare_audio_latents_pads_generated_dummy_length_for_sp():
    pipeline = _make_pipeline(LTX23Pipeline, sequence_parallel_size=2)

    latents, original_num_frames, padded_num_frames = pipeline.prepare_audio_latents(
        batch_size=1,
        num_channels_latents=8,
        num_mel_bins=64,
        audio_latent_length=1,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    assert original_num_frames == 1
    assert padded_num_frames == 2
    assert latents.shape == (1, 2, 128)


def test_prepare_audio_latents_pads_packed_sequence_dim_for_provided_latents():
    pipeline = _make_pipeline(LTX2Pipeline, sequence_parallel_size=4)
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
    original = torch.arange(40, dtype=torch.float32).view(1, 10, 4)
    padded = torch.cat([original, torch.full((1, 2, 4), 999.0)], dim=1)

    unpadded = latent_ops.unpad_audio_latents(padded, 10)
    unpacked = latent_ops.unpack_audio_latents(unpadded, latent_length=10, num_mel_bins=2)
    expected = latent_ops.unpack_audio_latents(original, latent_length=10, num_mel_bins=2)

    assert unpacked.shape == (1, 2, 10, 2)
    assert not (unpacked == 999.0).any()
    torch.testing.assert_close(unpacked, expected)


def test_prepare_audio_latents_accepts_already_padded_4d_latents_for_sp():
    pipeline = _make_pipeline(LTX23Pipeline, sequence_parallel_size=4)
    latents = torch.arange(96, dtype=torch.float32).view(1, 2, 12, 4)

    audio_latent_length = pipeline._resolve_audio_latent_length(10, latents)
    padded, original_num_frames, padded_num_frames = pipeline.prepare_audio_latents(
        batch_size=1,
        num_channels_latents=2,
        num_mel_bins=16,
        audio_latent_length=audio_latent_length,
        dtype=torch.float32,
        device=torch.device("cpu"),
        latents=latents,
    )

    assert audio_latent_length == 10
    assert original_num_frames == 10
    assert padded_num_frames == 12
    assert padded.shape == (1, 12, 8)
    torch.testing.assert_close(padded, latent_ops.pack_audio_latents(latents))


def test_resolve_audio_latent_length_preserves_legacy_4d_shape_inference():
    pipeline = _make_pipeline(LTX23Pipeline, sequence_parallel_size=4)
    latents = torch.zeros(1, 2, 13, 4)

    audio_latent_length = pipeline._resolve_audio_latent_length(10, latents)

    assert audio_latent_length == 13


def test_prepare_audio_latents_rejects_incompatible_provided_length():
    pipeline = _make_pipeline(LTX23Pipeline, sequence_parallel_size=4)
    latents = torch.zeros(1, 11, 4)

    with pytest.raises(ValueError, match="incompatible audio frame count"):
        pipeline.prepare_audio_latents(
            batch_size=1,
            num_channels_latents=2,
            num_mel_bins=8,
            audio_latent_length=10,
            dtype=torch.float32,
            device=torch.device("cpu"),
            latents=latents,
        )
