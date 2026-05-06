import pytest
import torch

from . import two_stage_vae, vae_inference
from .two_stage_vae_test import _build_and_verify_two_stage_vae


def _inference_config(
    model: two_stage_vae.TwoStageVAE,
    *,
    frame_chunk_len: int = 8,
    valid_skip_n_blocks: list[int] | None = None,
) -> vae_inference.TwoStageVAEInferenceConfig:
    """Build inference config with the VAE passed directly (no checkpoint path)."""
    td = model.temporal_stage.time_downsample_factor
    assert frame_chunk_len % td == 0, (
        "frame_chunk_len must be divisible by temporal downsample factor"
    )
    return vae_inference.TwoStageVAEInferenceConfig(
        checkpoint=model,
        frame_chunk_len=frame_chunk_len,
        scaling_factor=1.0,
        bias_factor=0.0,
        valid_skip_n_blocks=list(valid_skip_n_blocks)
        if valid_skip_n_blocks is not None
        else [0],
    )


def test_two_stage_vae_inference_encode_matches_model_mean() -> None:
    model = _build_and_verify_two_stage_vae()
    cfg = _inference_config(model)
    wrapped = cfg.make(device=torch.device("cpu"))
    t, h, w = 8, 16, 24
    x = torch.randn(1, model.in_channels, t, h, w)
    expected = model.encode(x).mean
    actual = wrapped.encode(x)
    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected)


def test_two_stage_vae_inference_single_frame() -> None:
    model = _build_and_verify_two_stage_vae()
    cfg = _inference_config(model, frame_chunk_len=4)
    wrapped = cfg.make(device=torch.device("cpu"))
    t, h, w = 1, 16, 24
    x = torch.randn(1, model.in_channels, t, h, w)
    expected_z_size = model.input_to_latent_size((t, h, w))
    z = wrapped.encode(x)
    assert z.shape[-3:] == expected_z_size
    x_hat = wrapped.decode(z, num_frames=t, spatial_size=(h, w))
    assert x_hat.shape == x.shape


def test_two_stage_vae_inference_scaling_and_bias_roundtrip() -> None:
    model = _build_and_verify_two_stage_vae()
    scaling = 0.5
    bias = 0.25
    inf_cfg = vae_inference.TwoStageVAEInferenceConfig(
        checkpoint=model,
        frame_chunk_len=8,
        scaling_factor=scaling,
        bias_factor=bias,
        valid_skip_n_blocks=[0],
    )
    wrapped = inf_cfg.make(device=torch.device("cpu"))
    t, h, w = 8, 16, 24
    x = torch.randn(1, model.in_channels, t, h, w)
    z = wrapped.encode(x)
    mu = model.encode(x).mean
    expected_z = (mu + bias) * scaling
    assert torch.allclose(z, expected_z)
    x_hat = wrapped.decode(z, num_frames=t, spatial_size=(h, w))
    assert x_hat.shape == x.shape
    assert torch.isfinite(x_hat).all()


def test_two_stage_vae_inference_compression_and_expansion() -> None:
    model = _build_and_verify_two_stage_vae(spatial_block_out_channels=(32, 32, 32))
    cfg = _inference_config(model, valid_skip_n_blocks=[0, 1])
    wrapped = cfg.make(device=torch.device("cpu"))
    assert len(wrapped.compression_modes) == 2
    assert len(wrapped.expansion_modes) == 2
    bad = (999, 999, 999)
    with pytest.raises(ValueError, match="not a supported compression mode"):
        wrapped.encode(torch.randn(1, model.in_channels, 8, 16, 24), compression=bad)
    with pytest.raises(ValueError, match="not a supported expansion mode"):
        wrapped.decode(torch.randn(1, model.in_channels, 8, 16, 24), expansion=bad)
    # iterate over compression and expansion modes and ensure that the shapes are correct
    x = torch.randn(1, model.in_channels, 8, 16, 24)
    for compression in wrapped.compression_modes:
        for expansion in wrapped.expansion_modes:
            expected_z_size = wrapped.input_to_latent_size(
                x.shape[-3:],  # type: ignore[arg-type]
                compression=compression,
            )
            z = wrapped.encode(x, compression=compression)
            assert z.shape[-3:] == expected_z_size
            # ensure that last three axes of z.shape is x.shape divided by compression factors
            assert all(
                sz_z == sz_x // c
                for sz_z, sz_x, c in zip(z.shape[-3:], x.shape[-3:], compression)
            )
            x_hat = wrapped.decode(z, expansion=expansion)
            # ensure that last three axes of x_hat.shape is z.shape multiplied by expansion factors
            assert all(
                sz_x_hat == sz_z * e
                for sz_x_hat, sz_z, e in zip(x_hat.shape[-3:], z.shape[-3:], expansion)
            )


def test_num_frames_and_frame_chunk_len_compatibility() -> None:
    model = _build_and_verify_two_stage_vae()
    cfg = _inference_config(model, frame_chunk_len=8)
    wrapped = cfg.make(device=torch.device("cpu"))
    # ensure assertion error is raised for incompatible number of frames and frame_chunk_len
    with pytest.raises(
        ValueError, match="number of frames must be 1 or divisible by frame_chunk_len"
    ):
        wrapped.encode(torch.randn(1, model.in_channels, 12, 16, 24))
    with pytest.raises(
        ValueError,
        match="number of latent frames must be divisible by latent_chunk_len",
    ):
        wrapped.decode(
            torch.randn(1, model.latent_embed_dim, 3, 16, 24),
        )
    # verify it works with 1 frame
    wrapped.encode(torch.randn(1, model.in_channels, 1, 16, 24))
    # to decode a single frame, num_frames must be passed as 1
    wrapped.decode(torch.randn(1, model.latent_embed_dim, 1, 16, 24), num_frames=1)
    # verify it works with 8 frames
    wrapped.encode(torch.randn(1, model.in_channels, 8, 16, 24))
    wrapped.decode(torch.randn(1, model.latent_embed_dim, 8, 16, 24))
