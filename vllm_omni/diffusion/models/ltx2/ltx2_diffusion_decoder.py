# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# SPDX-FileCopyrightText: Copyright 2026 Lightricks and The HuggingFace Team. All rights reserved.
#
# Native conversion rules and decoder behavior copied and modified from
# Diffusers commit d035dcd7cc7c88e0a154609b62887d50bba9fdc2 (Apache-2.0).

"""LTX-2.5 Native-weight adapter for Diffusers' diffusion VAE decoder."""

from collections.abc import Mapping
from typing import Any

import torch
from diffusers.models.autoencoders import (
    LTX2VideoDiffusionDecoderModel as DiffusersLTX2VideoDiffusionDecoderModel,
)
from diffusers.models.autoencoders.ltx2_diffusion_decoder import (
    LTX2VideoDiffusionDecoder3d as DiffusersLTX2VideoDiffusionDecoder3d,
)
from diffusers.models.autoencoders.vae import DecoderOutput
from safetensors import safe_open

LTX25_NATIVE_DIFFUSION_DECODER_REPO_ID = "Lightricks/LTX-2.5"
LTX25_NATIVE_ARTIFACT_REVISION = "8a4ff96f581e72bedc1b44367581c49d544a05f1"
LTX25_NATIVE_DIFFUSION_DECODER_FILENAME = "vae/ltx-2.5-video-vae-bf16.safetensors"

_NATIVE_DECODER_PREFIXES = ("vae.decoder.", "decoder.")
_NATIVE_STATISTICS_KEYS = {
    "per_channel_statistics.mean-of-means": "latents_mean",
    "per_channel_statistics.std-of-means": "latents_std",
}
_NATIVE_KEY_REPLACEMENTS = (
    ("t_embedder.mlp.0.", "t_embedder.timestep_embedder.linear_1."),
    ("t_embedder.mlp.2.", "t_embedder.timestep_embedder.linear_2."),
    (".attn.proj.", ".attn.to_out.0."),
    (".attn.q_norm.", ".attn.norm_q."),
    (".attn.k_norm.", ".attn.norm_k."),
)
_GATE_FOLD_TARGETS = {
    ".attn.to_out.0.weight": ".gate_msa",
    ".attn.to_out.0.bias": ".gate_msa",
    ".mlp.w_down.weight": ".gate_mlp",
    ".context_proj.weight": ".gate_ctx",
    ".context_proj.bias": ".gate_ctx",
}
_GATE_SUFFIXES = tuple(_GATE_FOLD_TARGETS.values())


def _strip_native_decoder_prefix(key: str) -> str | None:
    for prefix in _NATIVE_DECODER_PREFIXES:
        if key.startswith(prefix):
            return key.removeprefix(prefix)
    return None


def _native_statistics_target(key: str) -> str | None:
    return _NATIVE_STATISTICS_KEYS.get(key.removeprefix("vae."))


def _rename_native_decoder_key(key: str) -> str:
    for source, target in _NATIVE_KEY_REPLACEMENTS:
        key = key.replace(source, target)
    return key


def _fold_native_gate(key: str, value: torch.Tensor, gates: Mapping[str, torch.Tensor]) -> torch.Tensor:
    target_suffix = next((suffix for suffix in _GATE_FOLD_TARGETS if key.endswith(suffix)), None)
    if target_suffix is None:
        return value
    gate_key = key[: -len(target_suffix)] + _GATE_FOLD_TARGETS[target_suffix]
    gate = gates.get(gate_key)
    if gate is None:
        return value
    gate = gate.to(device=value.device, dtype=torch.float32)
    value_float = value.to(dtype=torch.float32)
    folded = gate.unsqueeze(1) * value_float if value.ndim == 2 else gate * value_float
    return folded.to(dtype=value.dtype)


def convert_ltx25_native_diffusion_decoder_state_dict(
    native_state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Convert the canonical Native LTX-2.5 DiffVAE state dict to Diffusers."""
    decoder_state_dict: dict[str, torch.Tensor] = {}
    converted: dict[str, torch.Tensor] = {}
    for key, value in native_state_dict.items():
        statistics_target = _native_statistics_target(key)
        if statistics_target is not None:
            converted[statistics_target] = value
            continue
        decoder_key = _strip_native_decoder_prefix(key)
        if decoder_key is not None:
            decoder_state_dict[decoder_key] = value

    gates = {key: value for key, value in decoder_state_dict.items() if key.endswith(_GATE_SUFFIXES)}
    for key, value in decoder_state_dict.items():
        # ``type_emb`` is an unused shipping-checkpoint residue. Preview heads
        # and legacy static gates are not parameters of the runtime decoder.
        if key == "type_emb" or key.startswith("coarse_") or ".coarse_" in key or key in gates:
            continue

        converted_key = _rename_native_decoder_key(key)
        value = _fold_native_gate(converted_key, value, gates)
        if converted_key.endswith((".qkv.weight", ".qkv.bias")):
            leaf = "weight" if converted_key.endswith(".weight") else "bias"
            prefix = converted_key[: -len(f"qkv.{leaf}")]
            if value.shape[0] % 3 != 0:
                raise ValueError(
                    f"Fused LTX-2.5 DiffVAE parameter {key!r} has leading dimension "
                    f"{value.shape[0]}, which is not divisible by 3."
                )
            chunk = value.shape[0] // 3
            converted[f"decoder.{prefix}to_q.{leaf}"] = value[:chunk].clone()
            converted[f"decoder.{prefix}to_k.{leaf}"] = value[chunk : 2 * chunk].clone()
            converted[f"decoder.{prefix}to_v.{leaf}"] = value[2 * chunk :].clone()
            continue

        converted[f"decoder.{converted_key}"] = value

    return converted


def load_ltx25_native_diffusion_decoder_state_dict(path: str) -> dict[str, torch.Tensor]:
    """Read and convert only DiffVAE tensors from the canonical Native VAE file."""
    with safe_open(path, framework="pt", device="cpu") as handle:
        native_state_dict = {
            key: handle.get_tensor(key)
            for key in handle.keys()
            if _strip_native_decoder_prefix(key) is not None or _native_statistics_target(key) is not None
        }
    return convert_ltx25_native_diffusion_decoder_state_dict(native_state_dict)


class LTX2VideoDiffusionDecoder3d(DiffusersLTX2VideoDiffusionDecoder3d):
    """Diffusers decoder core with the short-clip NATTEN context fix."""

    def forward_stage_4(
        self,
        hidden_states: torch.Tensor,
        drop_leading_frame: bool = True,
        crop_trailing_ghost: bool = True,
    ) -> torch.Tensor:
        blocks = self.det_stages[-1]
        block_mask = blocks[0].attn.build_block_mask(hidden_states)
        for block in blocks:
            hidden_states = block(hidden_states, block_mask)
        hidden_states = self.upsamples[-1](hidden_states, drop_leading_frame=drop_leading_frame)

        num_pad = self.trailing_pad_latent_frames
        if crop_trailing_ghost and num_pad > 0:
            content_frames = max(hidden_states.shape[1] - num_pad * self.temporal_compression_ratio, 1)
            keep_frames = min(hidden_states.shape[1], max(content_frames, self.stage5_kernel[0]))
            hidden_states = hidden_states[:, :keep_frames]
        return hidden_states


class LTX2VideoDiffusionDecoderModel(DiffusersLTX2VideoDiffusionDecoderModel):
    """Diffusers 0.40 decoder plus canonical Native loading and short-clip support."""

    def __init__(
        self,
        out_channels: int = 3,
        latent_channels: int = 128,
        patch_size: int = 4,
        scaling_factor: float = 1.0,
        decoder_head_dim: int = 64,
        decoder_stage_channels: tuple[int, ...] = (2048, 1024, 512, 512, 256),
        decoder_stage_depths: tuple[int, ...] = (4, 6, 4, 2, 8),
        decoder_stage_kernels: tuple[tuple[int, int, int], ...] = ((3, 7, 7), (3, 7, 7), (3, 5, 5), (3, 5, 5)),
        decoder_upsample_strides: tuple[tuple[int, int, int], ...] = ((1, 2, 2), (2, 1, 1), (2, 2, 2), (2, 2, 2)),
        decoder_upsample_channel_reductions: tuple[int, ...] = (2, 2, 1, 2),
        decoder_stage5_kernel: tuple[int, int, int] = (11, 11, 11),
        decoder_t_emb_dim: int = 384,
        decoder_timestep_scale_multiplier: float = 1000.0,
        decoder_model_output_type: str = "x0",
        decoder_num_inference_steps: int = 1,
        spatial_compression_ratio: int = 32,
        temporal_compression_ratio: int = 8,
    ) -> None:
        super().__init__(
            out_channels=out_channels,
            latent_channels=latent_channels,
            patch_size=patch_size,
            scaling_factor=scaling_factor,
            decoder_head_dim=decoder_head_dim,
            decoder_stage_channels=decoder_stage_channels,
            decoder_stage_depths=decoder_stage_depths,
            decoder_stage_kernels=decoder_stage_kernels,
            decoder_upsample_strides=decoder_upsample_strides,
            decoder_upsample_channel_reductions=decoder_upsample_channel_reductions,
            decoder_stage5_kernel=decoder_stage5_kernel,
            decoder_t_emb_dim=decoder_t_emb_dim,
            decoder_timestep_scale_multiplier=decoder_timestep_scale_multiplier,
            decoder_model_output_type=decoder_model_output_type,
            decoder_num_inference_steps=decoder_num_inference_steps,
            spatial_compression_ratio=spatial_compression_ratio,
            temporal_compression_ratio=temporal_compression_ratio,
        )
        # Diffusers constructs its private decoder core internally. Switching
        # to this behavior-only subclass preserves parameters and state-dict keys.
        self.decoder.__class__ = LTX2VideoDiffusionDecoder3d
        self.decoder.stage5_kernel = tuple(decoder_stage5_kernel)

    @classmethod
    def from_ltx25_native_checkpoint(
        cls,
        checkpoint_path: str,
        config: Mapping[str, Any],
        dtype: torch.dtype,
    ) -> "LTX2VideoDiffusionDecoderModel":
        """Construct this decoder class directly from the canonical Native checkpoint."""
        with torch.device("meta"):
            model = cls.from_config(dict(config))
        state_dict = load_ltx25_native_diffusion_decoder_state_dict(checkpoint_path)
        try:
            model.load_state_dict(state_dict, strict=True, assign=True)
        except RuntimeError as exc:
            raise ValueError(f"Invalid LTX-2.5 Native DiffVAE checkpoint {checkpoint_path!r}.") from exc
        model.to(device="cpu", dtype=dtype)
        return model

    def decode(
        self,
        z: torch.Tensor,
        generator: torch.Generator | None = None,
        num_inference_steps: int | None = None,
        return_dict: bool = True,
    ) -> DecoderOutput | tuple[torch.Tensor]:
        decoded = super().decode(
            z,
            generator=generator,
            num_inference_steps=num_inference_steps,
            return_dict=False,
        )[0]

        # Short clips retain replicated trailing context through stage 5. Do
        # not expose those context pixels to the caller.
        target_num_frames = (z.shape[2] - 1) * self.temporal_compression_ratio + 1
        target_height = z.shape[3] * self.spatial_compression_ratio
        target_width = z.shape[4] * self.spatial_compression_ratio
        decoded = decoded[:, :, :target_num_frames, :target_height, :target_width]

        if not return_dict:
            return (decoded,)
        return DecoderOutput(sample=decoded)
