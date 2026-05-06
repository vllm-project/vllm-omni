from __future__ import annotations

from dataclasses import dataclass
from typing import Type

import torch
from omegaconf import OmegaConf
from torch import nn
from torch.types import FileLike

from . import image_vae, utils, video_vae

# Configuration for importing from legacy checkpoints
#  - remove configuration keys that are no longer used
LEGACY_CKPT_DEL_CFG_KEYS = {  # module_key, param_key pairs
    ("spatial_vae", "force_upcast"),
    ("spatial_vae", "scaling_factor"),
    ("spatial_vae", "shift_factor"),
    ("spatial_vae", "use_quant_conv"),
    ("spatial_vae", "use_post_quant_conv"),
    ("temporal_vae", "type"),
    ("temporal_vae", "from_pretrained"),
}
#  - remove weights pertaining to losses, metrics and discriminator
#     (anything that starts with these prefixes will be removed)
LEGACY_CKPT_DEL_WEIGHTS_BY_PREFIX = {"fid.", "vae_loss_fn.", "discriminator."}
#  - map weights from legacy paths to new paths
#     (any key that starts with the prefix will be mapped to new prefix)
LEGACY_CKPT_MAP_WEIGHTS_BY_PREFIX = {
    "spatial_vae.module.encoder.": "spatial_encoder.",
    "spatial_vae.module.decoder.": "spatial_decoder.",
    "temporal_vae.": "temporal_stage.",
}


class TwoStageVAE(nn.Module):
    """
    A Pytorch Lightning module for a two-stage video VAE.

    This model contains a spatial-only stage plus a spatio-temporal stage.
    """

    def __init__(self, cfg: TwoStageVAEConfig):
        """
        Constructs a two-stage VAE module.

        Args:
            cfg (TwoStageVAEConfig): The configuration for the two-stage VAE.
        """
        super().__init__()
        self.cfg = cfg
        # build spatial encoder and decoder
        self.spatial_encoder: image_vae.Encoder = cfg.spatial_encoder.make()
        self.spatial_decoder: image_vae.Decoder = cfg.spatial_decoder.make()

        # create the temporal vae
        self.temporal_stage: video_vae.VideoVAE = cfg.temporal_stage.make()

    @classmethod
    def load_from_checkpoint(
        cls: Type[TwoStageVAE],
        checkpoint_path: FileLike,
        device: torch.device | str = "cpu",
        dtype: torch.dtype | None = None,
        strict: bool = True,
    ) -> TwoStageVAE:
        ckpt = torch.load(checkpoint_path, map_location=device)
        cfg = ckpt["hyper_parameters"]["cfg"]
        state_dict = ckpt["state_dict"]
        # -------- deal with legacy checkpoint format --------
        if "spatial_vae" in cfg:  # indicates this is a legacy checkpoint
            # delete certain keys that are no longer used
            cfg["spatial_vae"] = cfg["spatial_vae"]["cfg"]

            for module_key, param_key in LEGACY_CKPT_DEL_CFG_KEYS:
                if param_key in cfg[module_key]:
                    del cfg[module_key][param_key]
            # temporal_vae -> temporal_stage
            cfg["temporal_stage"] = cfg.pop("temporal_vae")
            cfg["temporal_stage"]["base_channels"] = cfg["temporal_stage"].pop(
                "filters"
            )
            # remove `conv_pad_mode` key, if it exists
            if "conv_pad_mode" in cfg["temporal_stage"]:
                del cfg["temporal_stage"]["conv_pad_mode"]
            # spatial_vae -> configs for spatial_encoder and spatial_decoder
            spatial_stage_cfg = cfg.pop("spatial_vae")
            spatial_down_block_types = spatial_stage_cfg.pop("down_block_types")
            spatial_up_block_types = spatial_stage_cfg.pop("up_block_types")
            in_channels = spatial_stage_cfg["in_channels"]
            out_channels = spatial_stage_cfg["out_channels"]
            spatial_io_channels = spatial_stage_cfg.pop("latent_channels")
            spatial_decoder_block_out_channels = spatial_stage_cfg.pop(
                "decoder_block_out_channels", spatial_stage_cfg["block_out_channels"]
            )
            cfg["spatial_encoder"] = {
                **spatial_stage_cfg,
                "in_channels": in_channels,
                "out_channels": spatial_io_channels,
                "down_block_types": spatial_down_block_types,
            }
            cfg["spatial_decoder"] = {
                **spatial_stage_cfg,
                "in_channels": spatial_io_channels,
                "out_channels": out_channels,
                "up_block_types": spatial_up_block_types,
                "block_out_channels": spatial_decoder_block_out_channels,
            }
            # build config class from dicts
            cfg = TwoStageVAEConfig(
                spatial_encoder=image_vae.EncoderConfig(**cfg["spatial_encoder"]),
                spatial_decoder=image_vae.DecoderConfig(**cfg["spatial_decoder"]),
                temporal_stage=video_vae.VideoVAEConfig(**cfg["temporal_stage"]),
                replicate_single_frames=cfg["replicate_single_frames"],
                default_skip_n_blocks=cfg.get("default_skip_n_blocks", 0),
            )
            for k in list(state_dict.keys()):
                for prefix in LEGACY_CKPT_DEL_WEIGHTS_BY_PREFIX:
                    if k.startswith(prefix):
                        del state_dict[k]
                # map weights
                for (
                    curr_prefix,
                    new_prefix,
                ) in LEGACY_CKPT_MAP_WEIGHTS_BY_PREFIX.items():
                    if k.startswith(curr_prefix):
                        new_key = k.replace(curr_prefix, new_prefix)
                        state_dict[new_key] = state_dict[k]
                        del state_dict[k]
            # strip redundant weights from spatial encoder conv_out
            state_dict["spatial_encoder.conv_out.weight"] = state_dict[
                "spatial_encoder.conv_out.weight"
            ][:spatial_io_channels]
            state_dict["spatial_encoder.conv_out.bias"] = state_dict[
                "spatial_encoder.conv_out.bias"
            ][:spatial_io_channels]
        # -------- end of dealing with legacy checkpoint format --------
        else:
            cfg = OmegaConf.to_object(
                OmegaConf.merge(OmegaConf.structured(TwoStageVAEConfig), cfg)
            )
        model = cls(cfg).to(device=device, dtype=dtype)
        model.load_state_dict(state_dict, strict=strict)
        return model

    @property
    def in_channels(self) -> int:
        return self.cfg.spatial_encoder.in_channels

    @property
    def out_channels(self) -> int:
        return self.cfg.spatial_decoder.out_channels

    @property
    def latent_embed_dim(self) -> int:
        return self.cfg.temporal_stage.latent_embed_dim

    @property
    def replicate_single_frames(self) -> bool:
        return self.cfg.replicate_single_frames

    def input_to_latent_size(
        self,
        input_size: tuple[int | None, int | None, int | None],
        skip_first_n_down_blocks: int = 0,
    ) -> tuple[int | None, int | None, int | None]:

        spatial_output_size = (
            input_size[0],
        ) + self.spatial_encoder.input_to_latent_size(
            input_size[1:], skip_first_n_down_blocks
        )
        return self.temporal_stage.input_to_latent_size(spatial_output_size)

    def get_downsample_factors(
        self, skip_first_n_down_blocks: int = 0
    ) -> tuple[int, int, int]:

        spatial_stage: tuple[int, int] = self.spatial_encoder.get_downsample_factors(
            skip_first_n_down_blocks
        )
        temporal_stage: tuple[int, int, int] = self.temporal_stage.downsample_factors
        return (
            temporal_stage[0],
            spatial_stage[0] * temporal_stage[1],
            spatial_stage[1] * temporal_stage[2],
        )

    def get_upsample_factors(
        self, skip_last_n_up_blocks: int = 0
    ) -> tuple[int, int, int]:
        spatial_stage: tuple[int, int] = self.spatial_decoder.get_upsample_factors(
            skip_last_n_up_blocks
        )
        # upsample factor same as downsample factor for temporal_stage
        temporal_stage: tuple[int, int, int] = self.temporal_stage.downsample_factors
        return (
            temporal_stage[0],
            temporal_stage[1] * spatial_stage[0],
            temporal_stage[2] * spatial_stage[1],
        )

    def maybe_replicate_single_frame(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[2] == 1 and self.replicate_single_frames:
            # repeat the single frame to match the temporal downsampling factor
            repeat = self.temporal_stage.time_downsample_factor
            x = x.tile((1, 1, repeat, 1, 1))
        return x

    def encode(
        self,
        x: torch.Tensor,
        skip_first_n_down_blocks: int = 0,
    ) -> utils.DiagonalGaussianDistribution:
        """
        Step to encode latents only (through the encoders of both stages).

        Args:
            x (torch.Tensor): The input tensor to encode.

        Returns:
            DiagonalGaussianDistribution: The encoded latent posterior.
        """
        # batch apply the spatial encoder to the input frames
        h = utils.WithAxisBatched(self.spatial_encoder, axis=2)(
            x, skip_first_n_down_blocks=skip_first_n_down_blocks
        )
        h = self.maybe_replicate_single_frame(h)
        return self.temporal_stage.encode(h)

    def decode(
        self,
        z: torch.Tensor,
        num_frames: int | None = None,
        spatial_size: tuple[int, int] | None = None,
        skip_last_n_up_blocks: int = 0,
    ) -> torch.Tensor:
        """
        Step to decode latents only (through the decoders of both stages).

        Args:
            z (torch.Tensor): The latent tensor to decode.

        Returns:
            torch.Tensor: The decoded tensor.
        """
        if spatial_size is not None:
            # deduce the target spatial size of the video_stage output / spatial stage input
            spatial_upsample_factors = self.spatial_decoder.get_upsample_factors(
                skip_last_n_up_blocks
            )
            assert spatial_size[0] % spatial_upsample_factors[0] == 0, (
                "target height not divisible by spatial decoder upsampling factor"
            )
            assert spatial_size[1] % spatial_upsample_factors[1] == 0, (
                "target width not divisible by spatial decoder upsampling factor"
            )
            temporal_stage_spatial_size = (
                spatial_size[0] // spatial_upsample_factors[0],
                spatial_size[1] // spatial_upsample_factors[1],
            )
        else:
            temporal_stage_spatial_size = None

        h = self.temporal_stage.decode(
            z, num_frames=num_frames, spatial_size=temporal_stage_spatial_size
        )
        # batch apply the spatial decoder to the input frames
        return utils.WithAxisBatched(self.spatial_decoder, axis=2)(
            h, skip_last_n_up_blocks=skip_last_n_up_blocks
        )


@dataclass
class TwoStageVAEConfig:
    spatial_encoder: image_vae.EncoderConfig
    spatial_decoder: image_vae.DecoderConfig
    temporal_stage: video_vae.VideoVAEConfig
    replicate_single_frames: bool = True
    default_skip_n_blocks: int = 0

    def __post_init__(self) -> None:
        # assert compatibility between spatial encoder and temporal encoder
        assert self.spatial_encoder.out_channels == self.temporal_stage.in_out_channels
        # assert compatibility between temporal decoder and spatial decoder
        assert self.temporal_stage.in_out_channels == self.spatial_decoder.in_channels
        num_spatial_down_blocks = len(self.spatial_encoder.down_block_types)
        assert self.default_skip_n_blocks < num_spatial_down_blocks, (
            "default_skip_n_blocks must be less than the number of spatial downblocks"
        )
        num_spatial_up_blocks = len(self.spatial_decoder.up_block_types)
        assert self.default_skip_n_blocks < num_spatial_up_blocks, (
            "default_skip_n_blocks must be less than the number of spatial upblocks"
        )

    def make(self) -> TwoStageVAE:
        return TwoStageVAE(self)
