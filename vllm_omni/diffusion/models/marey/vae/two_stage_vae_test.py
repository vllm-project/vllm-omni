import torch

from . import image_vae, two_stage_vae, video_vae


def _spatial_configs(
    *,
    in_channels: int,
    bridge_channels: int,
    block_out_channels: tuple[int, ...],
    act_fn: str,
) -> tuple[image_vae.EncoderConfig, image_vae.DecoderConfig]:
    num_blocks = len(block_out_channels)
    down_block_types = ("DownEncoderBlock2D",) * num_blocks
    up_block_types = ("UpDecoderBlock2D",) * num_blocks
    encoder = image_vae.EncoderConfig(
        in_channels=in_channels,
        out_channels=bridge_channels,
        down_block_types=down_block_types,
        block_out_channels=block_out_channels,
        norm_num_groups=32,
        act_fn=act_fn,
    )
    decoder = image_vae.DecoderConfig(
        in_channels=bridge_channels,
        out_channels=in_channels,
        up_block_types=up_block_types,
        block_out_channels=block_out_channels,
        norm_num_groups=32,
        act_fn=act_fn,
    )
    return encoder, decoder


def _build_and_verify_two_stage_vae(
    *,
    in_channels: int = 3,
    bridge_channels: int = 32,
    spatial_block_out_channels: tuple[int, ...] = (32, 64),
    latent_embed_dim: int = 8,
    base_channels: int = 32,
    channel_multipliers: tuple[int, ...] = (1, 2, 2, 4),
    temporal_downsample: tuple[bool, ...] = (True, True, False),
    spatial_downsample: tuple[bool, ...] = (False, False, False),
    num_res_blocks: int = 2,
    activation_fn: str = "silu",
    replicate_single_frames: bool = True,
) -> two_stage_vae.TwoStageVAE:
    spatial_encoder_cfg, spatial_decoder_cfg = _spatial_configs(
        in_channels=in_channels,
        bridge_channels=bridge_channels,
        block_out_channels=spatial_block_out_channels,
        act_fn=activation_fn,
    )
    temporal_stage_cfg = video_vae.VideoVAEConfig(
        in_out_channels=bridge_channels,
        latent_embed_dim=latent_embed_dim,
        base_channels=base_channels,
        main_kernel_size=(3, 3, 3),
        num_res_blocks=num_res_blocks,
        channel_multipliers=channel_multipliers,
        temporal_downsample=temporal_downsample,
        spatial_downsample=spatial_downsample,
        num_groups=8,
        activation_fn=activation_fn,
    )
    cfg = two_stage_vae.TwoStageVAEConfig(
        spatial_encoder=spatial_encoder_cfg,
        spatial_decoder=spatial_decoder_cfg,
        temporal_stage=temporal_stage_cfg,
        replicate_single_frames=replicate_single_frames,
        default_skip_n_blocks=0,
    )
    model = cfg.make()

    assert len(model.spatial_encoder.down_blocks) == len(spatial_block_out_channels)
    assert len(model.spatial_decoder.up_blocks) == len(spatial_block_out_channels)
    num_temporal_blocks = len(channel_multipliers)
    assert len(model.temporal_stage.encoder.block_res_blocks) == num_temporal_blocks

    assert model.in_channels == in_channels
    assert model.out_channels == in_channels
    assert model.latent_embed_dim == latent_embed_dim
    assert model.replicate_single_frames == replicate_single_frames

    ds = model.get_downsample_factors()
    us = model.get_upsample_factors()
    assert ds == us

    t, h, w = 8, 16, 24
    x = torch.randn(1, in_channels, t, h, w)
    lt, lh, lw = model.input_to_latent_size((t, h, w))
    assert lt is not None and lh is not None and lw is not None
    expected_latent_shape = (1, latent_embed_dim, lt, lh, lw)

    posterior = model.encode(x)
    z = posterior.sample()
    assert z.shape == expected_latent_shape
    x_reconstructed = model.decode(z, num_frames=t, spatial_size=(h, w))
    assert x_reconstructed.shape == x.shape
    assert x_reconstructed.dtype == z.dtype
    assert x_reconstructed.dtype == x.dtype

    return model


def test_two_stage_vae() -> None:
    test_in_channels = [3, 4]
    test_bridge_channels = [32, 64]
    test_latent_embed_dims = [8, 16]
    test_spatial_blocks: list[tuple[int, ...]] = [
        (32, 64),
        (32, 64, 128),
    ]
    test_channel_setups: list[
        tuple[tuple[int, ...], tuple[bool, ...], tuple[bool, ...]]
    ] = [
        ((1, 2, 2, 4), (True, True, False), (False, False, False)),
        ((1, 2), (False,), (False,)),
    ]
    test_act_fns = [("silu", torch.nn.SiLU), ("relu", torch.nn.ReLU)]
    for in_channels in test_in_channels:
        for bridge_channels in test_bridge_channels:
            for latent_embed_dim in test_latent_embed_dims:
                for spatial_block_out_channels in test_spatial_blocks:
                    for (
                        channel_multipliers,
                        temporal_downsample,
                        spatial_downsample,
                    ) in test_channel_setups:
                        for act_name, act_fn in test_act_fns:
                            model = _build_and_verify_two_stage_vae(
                                in_channels=in_channels,
                                bridge_channels=bridge_channels,
                                spatial_block_out_channels=spatial_block_out_channels,
                                latent_embed_dim=latent_embed_dim,
                                channel_multipliers=channel_multipliers,
                                temporal_downsample=temporal_downsample,
                                spatial_downsample=spatial_downsample,
                                activation_fn=act_name,
                            )
                            assert isinstance(model.spatial_encoder.conv_act, act_fn)
                            assert isinstance(model.spatial_decoder.conv_act, act_fn)
                            assert isinstance(
                                model.temporal_stage.encoder.activate, act_fn
                            )
                            assert isinstance(
                                model.temporal_stage.decoder.activate, act_fn
                            )
                            assert isinstance(
                                model.spatial_encoder.down_blocks[0]
                                .resnets[0]  # type: ignore[index]
                                .nonlinearity,  # type: ignore[index, union-attr]
                                act_fn,
                            )
                            assert isinstance(
                                model.spatial_decoder.up_blocks[0]
                                .resnets[0]  # type: ignore[index]
                                .nonlinearity,  # type: ignore[index, union-attr]
                                act_fn,
                            )
                            assert isinstance(
                                model.temporal_stage.encoder.block_res_blocks[0][
                                    0
                                ].activate,  # type: ignore[index]
                                act_fn,
                            )
                            assert isinstance(
                                model.temporal_stage.decoder.block_res_blocks[0][
                                    0
                                ].activate,  # type: ignore[index]
                                act_fn,
                            )
