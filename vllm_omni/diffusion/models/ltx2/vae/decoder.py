# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# SPDX-FileCopyrightText: Copyright 2026 Lightricks and The HuggingFace Team. All rights reserved.
#
# Native conversion rules and decoder behavior copied and modified from
# Diffusers commit d035dcd7cc7c88e0a154609b62887d50bba9fdc2 (Apache-2.0).

"""LTX-2.5 Native-weight adapter for Diffusers' diffusion VAE decoder."""

import math
from collections.abc import Mapping
from functools import cache
from typing import Any

import torch
from diffusers.models.autoencoders import (
    LTX2VideoDiffusionDecoderModel as DiffusersLTX2VideoDiffusionDecoderModel,
)
from diffusers.models.autoencoders import ltx2_diffusion_decoder as diffusers_decoder
from diffusers.models.autoencoders.ltx2_diffusion_decoder import (
    LTX2VideoDiffusionDecoder3d as DiffusersLTX2VideoDiffusionDecoder3d,
)
from diffusers.models.autoencoders.vae import DecoderOutput
from safetensors import safe_open

from ..ops import is_ltx2_fna_eligible, is_ltx2_fusion_eligible, resolve_ltx2_vae_operators

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


@cache
def _load_natten_token_permutation():
    from kernels import get_kernel

    natten = get_kernel("shi-labs/natten", version=1)
    return natten.token_permute.token_permute_operation, natten.token_permute.token_unpermute_operation


class LTX2VideoVaeNeighborhoodNattenProcessor(diffusers_decoder.LTX2VideoVaeNeighborhoodNattenProcessor):
    """NATTEN dispatch with Omni FNA for eligible stage-5 grids."""

    def __call__(
        self, attn: "LTX2VideoVaeNeighborhoodAttention", hidden_states: torch.Tensor, block_mask=None
    ) -> torch.Tensor:
        batch_size, num_frames, height, width, channels = hidden_states.shape
        query, key, value = attn.project_qkv(hidden_states)
        # NATTEN's CUTLASS kernels silently produce wrong output for non-contiguous inputs.
        query, key, value = query.contiguous(), key.contiguous(), value.contiguous()
        operators = resolve_ltx2_vae_operators(hidden_states.device) if is_ltx2_fna_eligible(hidden_states) else None
        # `scale=1.0`: the query is already scaled in `project_qkv`, as in the reference.
        if (
            tuple(attn.kernel_size) == (11, 11, 11)
            and hidden_states.dtype == torch.bfloat16
            and channels == 256
            and attn.head_dim == 64
            and operators is not None
            and operators.fna is not None
            # The native metadata encodes KV tile indices in 15 bits.
            and math.prod((n + tile - 1) // tile for n, tile in zip((num_frames, height, width), (4, 4, 8))) <= 1 << 15
        ):
            token_permute, token_unpermute = _load_natten_token_permutation()
            query_permuted, token_shape, _ = token_permute(query, (4, 4, 4), flip_tiled_dims=True)
            key_permuted, _, _ = token_permute(key, (4, 4, 8), flip_tiled_dims=True)
            value_permuted, _, _ = token_permute(value, (4, 4, 8), flip_tiled_dims=True)
            output = operators.fna(
                query_permuted,
                key_permuted,
                value_permuted,
                shape=(num_frames, height, width),
                window=tuple(attn.kernel_size),
            )
            hidden_states = token_unpermute(
                output,
                token_layout_shape=token_shape,
                tile_shape=(4, 4, 4),
                flip_tiled_dims=True,
            )
        else:
            hidden_states = self._na3d(
                query,
                key,
                value,
                kernel_size=attn.kernel_size,
                scale=1.0,
                backend=self.backend,
            )
        hidden_states = hidden_states.reshape(batch_size, num_frames, height, width, channels)
        return attn.to_out[0](hidden_states)


class LTX2VideoVaeNeighborhoodAttention(diffusers_decoder.LTX2VideoVaeNeighborhoodAttention):
    _available_processors = [
        *diffusers_decoder.LTX2VideoVaeNeighborhoodAttention._available_processors,
        LTX2VideoVaeNeighborhoodNattenProcessor,
    ]

    def project_qkv(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_frames, height, width, _ = hidden_states.shape
        shape = (batch_size, num_frames, height, width, self.heads, self.head_dim)
        query = self.to_q(hidden_states).view(shape)
        key = self.to_k(hidden_states).view(shape)
        value = self.to_v(hidden_states).view(shape)

        operators = resolve_ltx2_vae_operators(query.device) if is_ltx2_fusion_eligible(query) else None
        optimized = (
            None
            if operators is None
            else operators.qk_norm_rope(
                query,
                key,
                self.norm_q.weight,
                self.norm_k.weight,
                self.norm_q.eps,
                self.scale,
                self.rope.rope_dim_split,
                self.rope.base,
            )
        )
        if optimized is not None:
            query, key = optimized
            return query, key, value

        query = self.norm_q(query)
        key = self.norm_k(key)
        query = query * self.scale
        return self.rope(query), self.rope(key), value


class LTX2VideoVaeSwiGLU(diffusers_decoder.LTX2VideoVaeSwiGLU):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        operators = resolve_ltx2_vae_operators(hidden_states.device) if is_ltx2_fusion_eligible(hidden_states) else None
        optimized = (
            None
            if operators is None
            else operators.swiglu(hidden_states, self.w_gate.weight, self.w_up.weight, self.w_down.weight)
        )
        return super().forward(hidden_states) if optimized is None else optimized


class LTX2VideoVaeDiffusionNABlock(diffusers_decoder.LTX2VideoVaeDiffusionNABlock):
    def forward(
        self,
        hidden_states: torch.Tensor,
        latent_context: torch.Tensor,
        modulation: tuple[torch.Tensor, ...],
        block_mask=None,
    ) -> torch.Tensor:
        if hidden_states.dtype is not torch.bfloat16 or not is_ltx2_fusion_eligible(hidden_states):
            return super().forward(hidden_states, latent_context, modulation, block_mask)
        operators = resolve_ltx2_vae_operators(hidden_states.device)
        assert operators is not None

        scale_msa, shift_msa, _, scale_mlp, shift_mlp, _, _ = [
            modulation[i] + self.scale_shift_table[i].view(1, 1, 1, 1, -1) for i in range(self.num_mod_params)
        ]
        context_output = self.context_proj(latent_context)
        attention_input = operators.residual_norm(
            hidden_states,
            context_output,
            None,
            self.norm1.weight,
            scale_msa,
            shift_msa,
            self.norm1.eps,
        )
        if attention_input is None:
            hidden_states = hidden_states + context_output
            hidden_states = hidden_states + self.attn(
                self.norm1(hidden_states) * (1 + scale_msa) + shift_msa, block_mask
            )
            hidden_states = hidden_states + self.mlp(self.norm2(hidden_states) * (1 + scale_mlp) + shift_mlp)
            return hidden_states

        attention_output = self.attn(attention_input, block_mask)
        mlp_input = operators.residual_norm(
            hidden_states,
            context_output,
            attention_output,
            self.norm2.weight,
            scale_mlp,
            shift_mlp,
            self.norm2.eps,
        )
        if mlp_input is None:
            residual = hidden_states + context_output
            residual = residual + attention_output
            mlp_input = self.norm2(residual) * (1 + scale_mlp) + shift_mlp
        mlp_output = self.mlp(mlp_input)
        output = operators.residual_add(
            hidden_states,
            context_output,
            attention_output,
            mlp_output,
        )
        if output is not None:
            return output
        residual = hidden_states + context_output
        residual = residual + attention_output
        return residual + mlp_output


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
        overrides = {
            diffusers_decoder.LTX2VideoVaeNeighborhoodAttention: LTX2VideoVaeNeighborhoodAttention,
            diffusers_decoder.LTX2VideoVaeSwiGLU: LTX2VideoVaeSwiGLU,
            diffusers_decoder.LTX2VideoVaeDiffusionNABlock: LTX2VideoVaeDiffusionNABlock,
        }
        # Preserve every module and parameter object; only execution methods change.
        for module in self.decoder.modules():
            if replacement := overrides.get(type(module)):
                module.__class__ = replacement

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
