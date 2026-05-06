import torch

from . import video_vae


def _build_and_verify_video_vae(
    *,
    in_out_channels: int = 3,
    latent_embed_dim: int = 8,
    base_channels: int = 32,
    channel_multipliers: tuple[int, ...] = (1, 2, 2, 4),
    temporal_downsample: tuple[bool, ...] = (True, True, False),
    spatial_downsample: tuple[bool, ...] = (False, False, False),
    num_res_blocks: int = 2,
    activation_fn: str = "silu",
) -> video_vae.VideoVAE:
    vae = video_vae.VideoVAEConfig(
        in_out_channels=in_out_channels,
        latent_embed_dim=latent_embed_dim,
        base_channels=base_channels,
        main_kernel_size=(3, 3, 3),
        num_res_blocks=num_res_blocks,
        channel_multipliers=channel_multipliers,
        temporal_downsample=temporal_downsample,
        spatial_downsample=spatial_downsample,
        num_groups=8,
        activation_fn=activation_fn,
    ).make()
    num_blocks = len(channel_multipliers)
    assert len(vae.encoder.block_res_blocks) == num_blocks
    assert len(vae.encoder.conv_blocks) == num_blocks - 1

    t, h, w = 8, 16, 24
    x = torch.randn(1, in_out_channels, t, h, w)
    lt, lh, lw = vae.input_to_latent_size((t, h, w))
    assert lt is not None
    assert lh is not None
    assert lw is not None
    expected_latent_shape = (1, latent_embed_dim, lt, lh, lw)

    posterior = vae.encode(x)
    z = posterior.sample()
    assert z.shape == expected_latent_shape
    x_reconstructed = vae.decode(z, num_frames=t, spatial_size=(h, w))
    assert x_reconstructed.shape == x.shape
    assert x_reconstructed.dtype == z.dtype
    assert x_reconstructed.dtype == x.dtype

    posterior2, x_reconstructed2 = vae(x)
    assert posterior2.mean.shape == expected_latent_shape
    assert x_reconstructed2.shape == x.shape
    assert x_reconstructed2.dtype == x.dtype
    return vae


def test_video_vae() -> None:
    test_in_out_channels = [3, 4]
    test_latent_embed_dims = [8, 16]
    test_channel_setups: list[
        tuple[tuple[int, ...], tuple[bool, ...], tuple[bool, ...]]
    ] = [
        ((1, 2, 2, 4), (True, True, False), (False, False, False)),
        ((1, 2), (False,), (False,)),
    ]
    test_act_fns = [("silu", torch.nn.SiLU), ("relu", torch.nn.ReLU)]
    for in_out_channels in test_in_out_channels:
        for latent_embed_dim in test_latent_embed_dims:
            for (
                channel_multipliers,
                temporal_downsample,
                spatial_downsample,
            ) in test_channel_setups:
                for act_name, act_fn in test_act_fns:
                    vae = _build_and_verify_video_vae(
                        in_out_channels=in_out_channels,
                        latent_embed_dim=latent_embed_dim,
                        channel_multipliers=channel_multipliers,
                        temporal_downsample=temporal_downsample,
                        spatial_downsample=spatial_downsample,
                        activation_fn=act_name,
                    )
                    assert isinstance(vae.encoder.activate, act_fn)
                    assert isinstance(vae.decoder.activate, act_fn)
                    assert isinstance(
                        vae.encoder.block_res_blocks[0][0].activate,  # type: ignore[index]
                        act_fn,
                    )
                    assert isinstance(
                        vae.decoder.block_res_blocks[0][0].activate,  # type: ignore[index]
                        act_fn,
                    )
