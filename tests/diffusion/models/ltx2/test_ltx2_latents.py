# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.ltx2 import ltx2_latents as latent_ops
from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import LTX2Pipeline

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


def test_prepare_video_latents_matches_official_values_and_token_major_layout():
    pipeline = _make_pipeline(LTX2Pipeline)
    pipeline.vae_spatial_compression_ratio = 8
    pipeline.vae_temporal_compression_ratio = 8
    pipeline.transformer_spatial_patch_size = 2
    pipeline.transformer_temporal_patch_size = 1

    expected_generator = torch.Generator().manual_seed(42)
    expected = torch.randn((1, 32, 16), generator=expected_generator)
    actual = pipeline.prepare_latents(
        batch_size=1,
        num_channels_latents=4,
        height=64,
        width=64,
        num_frames=9,
        dtype=torch.float32,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(42),
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.stride()[1:] == (1, actual.shape[1])


def test_prepare_video_latents_uses_diffusion_decoder_statistics():
    pipeline = _make_pipeline(LTX2Pipeline)
    pipeline.use_diffusion_decoder = True
    pipeline.vae_spatial_compression_ratio = 1
    pipeline.vae_temporal_compression_ratio = 1
    pipeline.transformer_spatial_patch_size = 1
    pipeline.transformer_temporal_patch_size = 1
    pipeline.vae = SimpleNamespace(
        latents_mean=torch.full((2,), 100.0),
        latents_std=torch.full((2,), 10.0),
        config=SimpleNamespace(scaling_factor=1.0),
    )
    pipeline.diffusion_decoder = SimpleNamespace(
        latents_mean=torch.tensor([1.0, 3.0]),
        latents_std=torch.tensor([2.0, 4.0]),
        config=SimpleNamespace(scaling_factor=2.0),
    )

    actual = pipeline.prepare_latents(
        batch_size=1,
        num_channels_latents=2,
        height=1,
        width=1,
        num_frames=1,
        noise_scale=0.0,
        dtype=torch.float32,
        device=torch.device("cpu"),
        latents=torch.tensor([[[[[3.0]]], [[[7.0]]]]]),
    )

    torch.testing.assert_close(actual, torch.full((1, 1, 2), 2.0))


def test_prepare_audio_latents_samples_directly_in_packed_token_space():
    pipeline = _make_pipeline(LTX2Pipeline)

    expected_generator = torch.Generator().manual_seed(42)
    expected = torch.randn((1, 3, 32), generator=expected_generator)
    actual, original_num_frames, padded_num_frames = pipeline.prepare_audio_latents(
        batch_size=1,
        num_channels_latents=2,
        num_mel_bins=64,
        audio_latent_length=3,
        dtype=torch.float32,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(42),
    )

    assert original_num_frames == 3
    assert padded_num_frames == 3
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_prepare_audio_latents_pads_generated_dummy_length_for_sp():
    pipeline = _make_pipeline(LTX2Pipeline, sequence_parallel_size=2)

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
    torch.testing.assert_close(latents[:, 1:], torch.zeros_like(latents[:, 1:]))


def test_prepare_audio_latents_request_rng_is_invariant_to_sp_padding():
    pipeline_sp1 = _make_pipeline(LTX2Pipeline, sequence_parallel_size=1)
    pipeline_sp4 = _make_pipeline(LTX2Pipeline, sequence_parallel_size=4)
    generator_sp1 = torch.Generator().manual_seed(42)
    generator_sp4 = torch.Generator().manual_seed(42)

    latents_sp1, _, _ = pipeline_sp1.prepare_audio_latents(
        batch_size=1,
        num_channels_latents=2,
        num_mel_bins=8,
        audio_latent_length=3,
        dtype=torch.float32,
        device=torch.device("cpu"),
        generator=generator_sp1,
    )
    latents_sp4, _, _ = pipeline_sp4.prepare_audio_latents(
        batch_size=1,
        num_channels_latents=2,
        num_mel_bins=8,
        audio_latent_length=3,
        dtype=torch.float32,
        device=torch.device("cpu"),
        generator=generator_sp4,
    )

    torch.testing.assert_close(latents_sp4[:, :3], latents_sp1, rtol=0, atol=0)
    torch.testing.assert_close(latents_sp4[:, 3:], torch.zeros_like(latents_sp4[:, 3:]))
    torch.testing.assert_close(
        torch.randn(8, generator=generator_sp4),
        torch.randn(8, generator=generator_sp1),
        rtol=0,
        atol=0,
    )


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
    unpacked = latent_ops.unpack_audio_latents(unpadded, num_mel_bins=2)
    expected = latent_ops.unpack_audio_latents(original, num_mel_bins=2)

    assert unpacked.shape == (1, 2, 10, 2)
    assert not (unpacked == 999.0).any()
    torch.testing.assert_close(unpacked, expected)


def test_prepare_audio_latents_accepts_already_padded_4d_latents_for_sp():
    pipeline = _make_pipeline(LTX2Pipeline, sequence_parallel_size=4)
    pipeline.preserve_sp_padded_audio_duration = True
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
    packed = latent_ops.pack_audio_latents(latents)
    torch.testing.assert_close(padded[:, :10], packed[:, :10])
    torch.testing.assert_close(padded[:, 10:], torch.zeros_like(padded[:, 10:]))


def test_resolve_audio_latent_length_preserves_legacy_4d_shape_inference():
    pipeline = _make_pipeline(LTX2Pipeline, sequence_parallel_size=4)
    latents = torch.zeros(1, 2, 13, 4)

    audio_latent_length = pipeline._resolve_audio_latent_length(10, latents)

    assert audio_latent_length == 13


def test_prepare_audio_latents_rejects_incompatible_provided_length():
    pipeline = _make_pipeline(LTX2Pipeline, sequence_parallel_size=4)
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


def test_create_noised_state_matches_official_fp32_lerp():
    latents = torch.linspace(-2, 2, 4096, dtype=torch.bfloat16).reshape(1, 32, 128)
    expected_generator = torch.Generator().manual_seed(42)
    noise = torch.randn(latents.shape, generator=expected_generator, dtype=latents.dtype)
    expected = torch.lerp(latents.float(), noise.float(), torch.tensor(0.15, dtype=torch.bfloat16).float()).to(
        latents.dtype
    )

    actual = latent_ops.create_noised_state(
        latents,
        torch.tensor(0.15, dtype=torch.bfloat16),
        torch.Generator().manual_seed(42),
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_create_conditioned_noised_state_matches_official_two_lerps():
    latents = torch.linspace(-2, 2, 24, dtype=torch.bfloat16).reshape(1, 3, 8)
    clean_latents = latents.clone()
    clean_latents[:, 0] = 3.0
    denoise_mask = torch.tensor([[[0.0], [1.0], [1.0]]])
    expected_generator = torch.Generator().manual_seed(42)
    noise = torch.randn(latents.shape, generator=expected_generator, dtype=latents.dtype)
    noised = torch.lerp(latents.float(), noise.float(), 0.909375)
    expected = torch.lerp(clean_latents.float(), noised, denoise_mask).to(latents.dtype)

    actual = latent_ops.create_conditioned_noised_state(
        latents,
        clean_latents,
        denoise_mask,
        0.909375,
        torch.Generator().manual_seed(42),
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_prepare_supplied_video_latents_uses_official_token_major_layout():
    pipeline = _make_pipeline(LTX2Pipeline)
    supplied = torch.arange(512, dtype=torch.float32).reshape(1, 32, 16)
    actual_video = pipeline.prepare_latents(
        batch_size=1,
        num_channels_latents=4,
        height=64,
        width=64,
        num_frames=9,
        dtype=torch.float32,
        device=torch.device("cpu"),
        latents=supplied,
    )

    torch.testing.assert_close(actual_video, supplied, rtol=0, atol=0)
    assert actual_video.stride()[1:] == (1, actual_video.shape[1])


def test_clear_audio_padding_keeps_padding_outside_sampler_state():
    updated = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [99.0, -99.0]]])

    actual = latent_ops.clear_audio_padding(updated, 2)

    torch.testing.assert_close(actual[:, :2], updated[:, :2])
    torch.testing.assert_close(actual[:, 2:], torch.zeros_like(actual[:, 2:]))
