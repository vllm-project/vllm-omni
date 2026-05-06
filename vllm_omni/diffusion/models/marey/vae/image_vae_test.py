import gc
from itertools import product

import pytest
import torch

from . import image_vae

_IN_CHANNELS = [3, 4, 5]
_LATENT_CHANNELS = [11, 16, 128]
_BLOCK_OUT_CHANNELS = [(32,), (32, 64, 128), (32, 64, 128, 256)]
_ACT_FNS: list[tuple[str, type[torch.nn.Module]]] = [
    ("silu", torch.nn.SiLU),
    ("relu", torch.nn.ReLU),
]

_PARAM_COMBOS = list(
    product(_IN_CHANNELS, _LATENT_CHANNELS, _BLOCK_OUT_CHANNELS, _ACT_FNS)
)


@pytest.mark.parametrize(
    "in_channels, latent_channels, block_out_channels, act_pair",
    _PARAM_COMBOS,
    ids=[
        f"in{ic}-lat{lc}-blocks{len(boc)}-{ap[0]}" for ic, lc, boc, ap in _PARAM_COMBOS
    ],
)
def test_image_vae(
    in_channels: int,
    latent_channels: int,
    block_out_channels: tuple[int, ...],
    act_pair: tuple[str, type[torch.nn.Module]],
) -> None:
    """Verify ImageVAE encode/decode/forward for a single config."""
    act_name, act_cls = act_pair
    num_blocks = len(block_out_channels)

    down_block_types = ("DownEncoderBlock2D",) * num_blocks
    up_block_types = ("UpDecoderBlock2D",) * num_blocks
    vae = image_vae.ImageVAE(
        in_channels=in_channels,
        out_channels=in_channels,
        block_out_channels=block_out_channels,
        latent_channels=latent_channels,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
        act_fn=act_name,
    )
    assert len(vae.encoder.down_blocks) == num_blocks
    assert len(vae.decoder.up_blocks) == num_blocks

    downsample_factor = num_blocks - 1
    expected_latent_shape = (1, latent_channels, 3, 4)
    input_shape = (
        1,
        in_channels,
        expected_latent_shape[2] * 2**downsample_factor,
        expected_latent_shape[3] * 2**downsample_factor,
    )
    x = torch.randn(input_shape)

    with torch.no_grad():
        z = vae.encode(x).sample()
        assert z.shape == expected_latent_shape
        x_reconstructed = vae.decode(z)
        assert x_reconstructed.shape == x.shape
        assert x_reconstructed.dtype == z.dtype
        assert x_reconstructed.dtype == x.dtype

        posterior, x_reconstructed = vae(x)
        assert posterior.mean.shape == expected_latent_shape
        assert x_reconstructed.shape == x.shape
        assert x_reconstructed.dtype == x.dtype

    assert isinstance(vae.encoder.conv_act, act_cls)
    assert isinstance(vae.decoder.conv_act, act_cls)
    assert isinstance(
        vae.encoder.down_blocks[0].resnets[0].nonlinearity,  # type: ignore[index, union-attr]
        act_cls,
    )
    assert isinstance(
        vae.decoder.up_blocks[0].resnets[0].nonlinearity,  # type: ignore[index, union-attr]
        act_cls,
    )

    del vae
    gc.collect()
