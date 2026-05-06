from __future__ import annotations

import torch

from .tokenization_config import TokenizationConfig
from .vae_protocol import (
    DummyVAE,
    DummyVAEConfig,
    VAEProtocol,
)

_CPU = torch.device("cpu")


def test_dummy_protocol_conformance() -> None:
    """DummyVAE satisfies the VAEProtocol runtime protocol."""
    enc = DummyVAEConfig().make(device=_CPU)
    assert isinstance(enc, VAEProtocol)


def test_dummy_properties() -> None:
    tc = TokenizationConfig(visual_latent_dim=8, patch_size=32)
    enc = DummyVAEConfig(tokenization_config=tc).make(device=_CPU)
    assert enc.latent_dim == 8
    assert enc.compression_modes == ((4, 32, 32),)
    assert enc.temporal_chunk_size == 16


def test_dummy_encode_image_shape() -> None:
    """Single-frame (image) encoding: ``(1, 3, 1, H, W) -> (1, D, 1, H/ps, W/ps)``."""
    tc = TokenizationConfig(visual_latent_dim=4, patch_size=32)
    enc = DummyVAEConfig(tokenization_config=tc).make(device=_CPU)
    x = torch.randn(1, 3, 1, 128, 128)
    out = enc.encode(x)
    assert out.shape == (1, 4, 1, 4, 4)


def test_dummy_encode_video_shape() -> None:
    """Multi-frame (video) encoding with temporal compression."""
    tc = TokenizationConfig(
        visual_latent_dim=4,
        patch_size=32,
        vae_temporal_compression_factor=4,
    )
    enc = DummyVAEConfig(tokenization_config=tc).make(device=_CPU)
    x = torch.randn(2, 3, 8, 128, 64)
    out = enc.encode(x)
    assert out.shape == (2, 4, 2, 4, 2)


def test_dummy_encode_video_no_temporal_compression() -> None:
    """Temporal compression=1 preserves all frames."""
    tc = TokenizationConfig(
        visual_latent_dim=4,
        patch_size=32,
        vae_temporal_compression_factor=1,
    )
    enc = DummyVAEConfig(tokenization_config=tc).make(device=_CPU)
    x = torch.randn(2, 3, 8, 128, 64)
    out = enc.encode(x)
    assert out.shape == (2, 4, 8, 4, 2)


def test_dummy_decode_shape() -> None:
    """Decode produces the original spatial dimensions."""
    tc = TokenizationConfig(
        visual_latent_dim=4,
        patch_size=32,
        vae_temporal_compression_factor=1,
    )
    enc = DummyVAEConfig(tokenization_config=tc).make(device=_CPU)
    x = torch.randn(1, 3, 4, 128, 128)
    z = enc.encode(x)
    assert z.shape == (1, 4, 4, 4, 4)
    recon = enc.decode(z)
    assert recon.shape == (1, 3, 4, 128, 128)


def test_dummy_config_make_device() -> None:
    """``DummyVAEConfig.make(device=...)`` moves the encoder to the device."""
    enc = DummyVAEConfig().make(device=_CPU)
    assert isinstance(enc, DummyVAE)
    assert next(enc.parameters()).device == _CPU


def test_dummy_encode_deterministic() -> None:
    """Same input produces same output across two calls."""
    tc = TokenizationConfig(visual_latent_dim=4, patch_size=32)
    enc = DummyVAEConfig(tokenization_config=tc).make(device=_CPU)
    x = torch.randn(1, 3, 2, 64, 64)
    out1 = enc.encode(x)
    out2 = enc.encode(x)
    torch.testing.assert_close(out1, out2)


def test_tokenization_config_property() -> None:
    """DummyVAE exposes its tokenization_config."""
    tc = TokenizationConfig(patch_size=32, visual_latent_dim=8)
    enc = DummyVAEConfig(tokenization_config=tc).make(device=_CPU)
    assert enc.tokenization_config is tc
