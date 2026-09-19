# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Independent V1.2 LiDAR encoder and decoder loaded from the unified VAE."""

from __future__ import annotations

import inspect
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from typing_extensions import Self

_LIDAR_VAE_SUBFOLDER = "lidar_vae"
_LIDAR_VAE_WEIGHTS = "diffusion_pytorch_model.safetensors"


def validate_lidar_config(config: dict[str, Any]) -> None:
    if not isinstance(config, dict):
        raise ValueError("LiDAR metadata must be a JSON object.")
    required = {
        "version",
        "dtype",
        "sample_posterior",
        "apply_validity_mask",
        "fps",
        "latent_channels",
        "temporal_compression_factor",
        "spatial_compression",
        "network_config",
        "range_projection",
        "streaming_chunk_frames",
        "streaming_context_frames",
    }
    if missing := required - config.keys():
        raise ValueError(f"Incomplete joint artifact: missing LiDAR metadata {sorted(missing)}.")
    if config["version"] != "1.2":
        raise ValueError("Only the V1.2 LiDAR tokenizer is supported.")
    if config["dtype"] != "float32" or config["sample_posterior"] is not False:
        raise ValueError("LiDAR requires FP32 execution and posterior-mean encoding.")
    if not isinstance(config["apply_validity_mask"], bool):
        raise ValueError("LiDAR apply_validity_mask must be boolean.")
    projection = config["range_projection"]
    if not isinstance(projection, dict) or not isinstance(config["network_config"], dict):
        raise ValueError("LiDAR range_projection and network_config must be JSON objects.")
    expected = {
        "semantic_width": 1800,
        "model_width": 1808,
        "native_height": 128,
        "model_width_transform": "circular_pad",
        "intensity_encoding": "unit",
    }
    if any(projection.get(key) != value for key, value in expected.items()):
        raise ValueError("V1.2 requires the 128x1800 metric/unit-intensity grid with circular padding to 1808.")
    minimum, maximum = projection.get("min_range_m"), projection.get("max_range_m")
    if not all(isinstance(v, int | float) and math.isfinite(v) for v in (minimum, maximum)) or maximum <= minimum:
        raise ValueError("LiDAR range normalization requires finite min_range_m < max_range_m.")
    network = config["network_config"]
    if (
        network.get("resolution") != [128, 1808]
        or network.get("patch_size") != [2, 2]
        or len(network.get("depths", [])) != 4
        or config["spatial_compression"] != [16, 16]
        or config["temporal_compression_factor"] != 1
        or network.get("z_dim") != config["latent_channels"]
        or network.get("in_channels") != 3
        or any(network.get("temporal_downsample", [True]))
    ):
        raise ValueError("LiDAR architecture and V1.2 compression metadata disagree.")
    if not math.isfinite(config["fps"]) or config["fps"] <= 0:
        raise ValueError("LiDAR FPS must be finite and positive.")
    chunk, context = config["streaming_chunk_frames"], config["streaming_context_frames"]
    if type(chunk) is not int or chunk < 1 or (context is not None and (type(context) is not int or context < chunk)):
        raise ValueError("LiDAR streaming context must be at least the positive chunk length.")


def _validate_lidar_vae_config(component: dict[str, Any], deployment: dict[str, Any]) -> None:
    validate_lidar_config(deployment)
    validate_lidar_config(component)
    for key, expected in deployment.items():
        if key == "network_config":
            # The VAE resolves constructor defaults that transformer metadata may omit.
            if any(name not in component[key] or component[key][name] != value for name, value in expected.items()):
                raise ValueError("LiDAR VAE network_config metadata disagrees with transformer deployment metadata.")
        elif key not in component or component[key] != expected:
            raise ValueError(f"LiDAR VAE {key} metadata disagrees with transformer deployment metadata.")


def prepare_lidar_encoder_input(frames: torch.Tensor, projection: dict[str, Any]) -> torch.Tensor:
    """Reference physical normalization after symmetric circular width padding."""
    frames = frames.float().unsqueeze(0)
    padding = projection["model_width"] - projection["semantic_width"]
    if padding < 0 or padding % 2:
        raise ValueError("LiDAR model width must allow symmetric circular padding of the semantic width.")
    half_padding = padding // 2
    if half_padding:
        frames = torch.cat((frames[..., -half_padding:], frames, frames[..., :half_padding]), dim=-1)
    minimum, maximum = projection["min_range_m"], projection["max_range_m"]
    ranges, intensities, validity = frames.split(1, dim=1)
    valid = (validity >= 0.5) & (ranges >= minimum) & (ranges <= maximum)
    ranges = (ranges.clamp(minimum, maximum) - minimum) / (maximum - minimum) * 2 - 1
    intensities = intensities.clamp(0, 1) * 2 - 1
    return torch.cat((ranges.masked_fill(~valid, -1), intensities.masked_fill(~valid, -1), valid.float()), dim=1)


class _LidarComponent(nn.Module):
    _unused_prefixes: tuple[str, ...]
    _component_name: str

    @classmethod
    def from_pretrained(cls, model_path: str, config: dict[str, Any], device: torch.device) -> Self:
        from safetensors import safe_open

        checkpoint_path = Path(model_path)
        if not checkpoint_path.exists():
            from vllm_omni.transformers_utils.repo_utils import hf_api

            checkpoint_path = Path(
                hf_api().snapshot_download(
                    model_path,
                    allow_patterns=[
                        f"{_LIDAR_VAE_SUBFOLDER}/config.json",
                        f"{_LIDAR_VAE_SUBFOLDER}/{_LIDAR_VAE_WEIGHTS}",
                    ],
                )
            )
        folder = checkpoint_path / _LIDAR_VAE_SUBFOLDER
        if not (folder / "config.json").is_file() or not (folder / _LIDAR_VAE_WEIGHTS).is_file():
            raise ValueError(
                f"Incomplete joint artifact: {_LIDAR_VAE_SUBFOLDER}/config.json and {_LIDAR_VAE_WEIGHTS} are required."
            )
        component_config = json.loads((folder / "config.json").read_text())
        _validate_lidar_vae_config(component_config, config)
        # Pipeline construction may set the default parameter dtype to BF16.
        # Convert before loading so FP32 checkpoint values are never rounded.
        model = cls(component_config).float()
        expected = model.state_dict()
        with safe_open(folder / _LIDAR_VAE_WEIGHTS, framework="pt", device="cpu") as weights:
            # Each component reads only its own parameters and the shared buffers.
            keys = [name for name in weights.keys() if not name.startswith(cls._unused_prefixes)]
            if unexpected := set(keys) - expected.keys():
                raise ValueError(f"Unexpected LiDAR {cls._component_name} tensors: {sorted(unexpected)}.")
            state = {}
            for name in keys:
                if expected[name].is_floating_point() and weights.get_slice(name).get_dtype() != "F32":
                    raise ValueError(f"LiDAR {cls._component_name} tensor {name} must be FP32.")
                state[name] = weights.get_tensor(name)
            model.load_state_dict(state, strict=True)
        if (
            not torch.isfinite(model.latent_mean).all()
            or not torch.isfinite(model.latent_std).all()
            or (model.latent_std <= 0).any()
        ):
            raise ValueError("LiDAR latent statistics must be finite with positive standard deviations.")
        return model.eval().requires_grad_(False).to(device=device, dtype=torch.float32)


class Cosmos3LidarEncoder(_LidarComponent):
    _unused_prefixes = ("decoder.", "post_quant_conv.")
    _component_name = "encoder"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        validate_lidar_config(config)
        from .lidar_encoder.encoding import generate_polar_coords
        from .lidar_encoder.transformer_vae import Encoder

        self.config = config
        network = config["network_config"]
        encoder_args = set(inspect.signature(Encoder).parameters)
        self.encoder = Encoder(**{key: value for key, value in network.items() if key in encoder_args})
        channels = config["latent_channels"]
        self.quant_conv = nn.Conv2d(channels * 2, channels * 2, 1)
        self.register_buffer("coords", generate_polar_coords(*network["resolution"]))
        self.register_buffer("latent_mean", torch.empty(1, channels, 1, 1, 1))
        self.register_buffer("latent_std", torch.empty(1, channels, 1, 1, 1))

    @torch.inference_mode()
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # Keep normalization, network execution and the latent affine in FP32,
        # including when the parent DiT is under BF16 autocast.
        with torch.autocast(device_type=self.coords.device.type, enabled=False):
            state = prepare_lidar_encoder_input(frames.to(self.coords.device), self.config["range_projection"])
            chunk = self.config["streaming_chunk_frames"]
            context = self.config["streaming_context_frames"]
            outputs, cache = [], None
            for start in range(0, state.shape[2], chunk):
                pixels = state[:, :, start : start + chunk]
                if cache is not None and context is not None:
                    keep = context - pixels.shape[2]
                    cache = (
                        {key: (k[:, :, -keep:], v[:, :, -keep:]) for key, (k, v) in cache.items()} if keep > 0 else None
                    )
                parameters, cache = self.encoder.forward_stream(pixels, self.coords, cache)
                batch, _, time, height, width = parameters.shape
                parameters = parameters.permute(0, 2, 1, 3, 4).flatten(0, 1)
                mean = self.quant_conv(parameters).chunk(2, dim=1)[0]
                outputs.append(mean.reshape(batch, time, -1, height, width).permute(0, 2, 1, 3, 4))
            latent = torch.cat(outputs, dim=2)
            return (latent - self.latent_mean) / self.latent_std


def lidar_decoder_args(network: dict[str, Any]) -> dict[str, Any]:
    """Resolve the decoder topology exactly as the reference TransformerVAE."""
    from .lidar_encoder.transformer_vae import Decoder

    args = {key: value for key, value in network.items() if key in inspect.signature(Decoder).parameters}
    # TransformerVAE never enables the standalone decoder's external-encoder stem.
    args.pop("stem_patchify", None)
    for field in ("depths", "num_heads", "dilation"):
        if network.get(f"decoder_{field}") is not None:
            args[field] = network[f"decoder_{field}"]
    if network.get("decoder_depths") is not None:
        temporal = network.get("decoder_temporal_upsample")
        if temporal is None:
            raise ValueError("An asymmetric LiDAR decoder requires decoder_temporal_upsample.")
        args.update(temporal_downsample=temporal, temporal_upsample=temporal)
    return args


def validate_lidar_decoder_config(config: dict[str, Any]) -> None:
    validate_lidar_config(config)
    network = config["network_config"]
    if (
        network.get("out_channels") != 3
        or network.get("predict_validity", False)
        or network.get("mask_as_input", False)
        or network.get("formulation", "VAE") != "VAE"
    ):
        raise ValueError("V1.2 LiDAR decoding requires three output channels with embedded validity logits.")
    args = lidar_decoder_args(network)
    depths = args["depths"]
    patch = args.get("out_patch_size") or args["patch_size"]
    temporal = args.get("temporal_upsample")
    if temporal is None:
        temporal = args["temporal_downsample"]
    if (
        len(depths) < 1
        or len(patch) != 2
        or [size * 2 ** (len(depths) - 1) for size in patch] != config["spatial_compression"]
        or len(temporal) != len(depths) - 1
        or any(temporal)
        or any(len(args.get(field, ())) != len(depths) for field in ("num_heads", "dilation"))
    ):
        raise ValueError("LiDAR decoder topology must preserve V1.2 spatial and temporal compression.")
    if args.get("temporal_mixer", "attention") != "attention" or (
        args.get("bottleneck_3d", False)
        and not (args.get("bottleneck_3d_causal_time", False) and args.get("bottleneck_3d_rope", False))
    ):
        raise ValueError("LiDAR streaming decode requires causal temporal attention with RoPE for a 3D bottleneck.")
    threshold = config["range_projection"].get("validity_threshold", 0.5)
    if isinstance(threshold, bool) or not isinstance(threshold, int | float) or not 0 < threshold < 1:
        raise ValueError("LiDAR validity_threshold must lie in (0, 1).")


def postprocess_lidar_decoder_output(output: torch.Tensor, config: dict[str, Any]) -> torch.Tensor:
    """Convert raw network predictions to the unpadded metric sensor grid."""
    projection = config["range_projection"]
    if (
        output.ndim != 5
        or output.shape[1] != 3
        or tuple(output.shape[-2:])
        != (
            projection["native_height"],
            projection["model_width"],
        )
    ):
        raise ValueError(f"Unexpected LiDAR decoder output shape: {tuple(output.shape)}.")
    output = output.float()
    minimum, maximum = projection["min_range_m"], projection["max_range_m"]
    ranges = (output[:, :1].clamp(-1, 1) + 1) * 0.5 * (maximum - minimum) + minimum
    intensity = (output[:, 1:2].clamp(-1, 1) + 1) * 0.5
    validity = output[:, 2:3].sigmoid()
    if config["apply_validity_mask"]:
        valid = validity >= projection.get("validity_threshold", 0.5)
        ranges = ranges.masked_fill(~valid, 0)
        intensity = intensity.masked_fill(~valid, 0)
        validity = valid.float()
    frames = torch.cat((ranges, intensity, validity), dim=1)
    offset = (projection["model_width"] - projection["semantic_width"]) // 2
    return frames[..., offset : offset + projection["semantic_width"]].contiguous()


class Cosmos3LidarDecoder(_LidarComponent):
    """Inference-only, FP32 decoder for normalized V1.2 diffusion latents."""

    _unused_prefixes = ("encoder.", "quant_conv.")
    _component_name = "decoder"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        validate_lidar_decoder_config(config)
        from .lidar_encoder.encoding import generate_polar_coords
        from .lidar_encoder.transformer_vae import Decoder

        self.config = config
        self.decoder = Decoder(**lidar_decoder_args(config["network_config"]))
        channels = config["latent_channels"]
        self.post_quant_conv = nn.Conv2d(channels, channels, 1)
        self.register_buffer("coords", generate_polar_coords(*config["network_config"]["resolution"]))
        self.register_buffer("latent_mean", torch.empty(1, channels, 1, 1, 1))
        self.register_buffer("latent_std", torch.empty(1, channels, 1, 1, 1))

    @torch.inference_mode()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Return [B,3,T,128,1800] metric frames, keeping each request's cache local."""
        expected_hw = tuple(
            size // factor
            for size, factor in zip(self.config["network_config"]["resolution"], self.config["spatial_compression"])
        )
        if (
            latents.ndim != 5
            or latents.shape[1] != self.config["latent_channels"]
            or tuple(latents.shape[-2:]) != expected_hw
            or latents.shape[0] < 1
            or latents.shape[2] < 1
        ):
            raise ValueError(f"Expected nonempty LiDAR latents [B,{self.config['latent_channels']},T,{expected_hw}].")
        with torch.autocast(device_type=self.coords.device.type, enabled=False):
            latents = latents.to(device=self.coords.device, dtype=torch.float32)
            latents = latents * self.latent_std + self.latent_mean
            chunk = self.config["streaming_chunk_frames"]
            context = self.config["streaming_context_frames"]
            outputs, cache = [], None
            for start in range(0, latents.shape[2], chunk):
                latent = latents[:, :, start : start + chunk]
                batch, _, time, height, width = latent.shape
                if cache is not None and context is not None:
                    keep = context - time
                    cache = (
                        {key: (k[:, :, -keep:], v[:, :, -keep:]) for key, (k, v) in cache.items()} if keep > 0 else None
                    )
                latent = latent.permute(0, 2, 1, 3, 4).flatten(0, 1)
                latent = self.post_quant_conv(latent)
                latent = latent.reshape(batch, time, -1, height, width).permute(0, 2, 1, 3, 4)
                output, cache = self.decoder(
                    latent, self.coords, temporal_kv_cache=cache, return_temporal_kv_cache=True
                )
                outputs.append(postprocess_lidar_decoder_output(output, self.config))
            return torch.cat(outputs, dim=2)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decode(latents)
