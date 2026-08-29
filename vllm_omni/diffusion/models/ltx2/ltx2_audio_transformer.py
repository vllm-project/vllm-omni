# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Audio-only projection of the full LTX Transformer checkpoint."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import torch
from diffusers.models.embeddings import PixArtAlphaTextProjection
from torch import nn
from torch.utils.checkpoint import checkpoint
from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.distributed.hsdp_utils import is_transformer_block_module
from vllm_omni.diffusion.distributed.sp_plan import SequenceParallelInput, SequenceParallelOutput

from .ltx2_transformer import (
    LTX2AdaLayerNormSingle,
    LTX2Attention,
    LTX2AudioVideoRotaryPosEmbed,
    LTX2FeedForward,
    _make_rms_norm,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig


@dataclass(frozen=True)
class LTX2AudioStaticConditioning:
    """Immutable Transformer inputs prepared once for one audio request."""

    encoder_hidden_states: torch.Tensor
    rotary_emb: tuple[torch.Tensor, torch.Tensor]


class LTX2AudioTransformerBlock(nn.Module):
    """The self-attention, text cross-attention and FFN audio path of an LTX block."""

    def __init__(
        self,
        *,
        audio_dim: int,
        audio_num_attention_heads: int,
        audio_attention_head_dim: int,
        audio_cross_attention_dim: int,
        audio_gated_attn: bool = False,
        audio_cross_attn_adaln: bool = False,
        qk_norm: str = "rms_norm_across_heads",
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = True,
        attention_out_bias: bool = True,
        eps: float = 1e-6,
        elementwise_affine: bool = False,
        rope_type: str = "interleaved",
        perturbed_attn: bool = False,
        audio_ff_bias: bool = True,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.audio_cross_attn_adaln = audio_cross_attn_adaln
        self.perturbed_attn = perturbed_attn

        self.audio_norm1 = _make_rms_norm(audio_dim, eps=eps, elementwise_affine=elementwise_affine)
        self.audio_attn1 = LTX2Attention(
            query_dim=audio_dim,
            heads=audio_num_attention_heads,
            kv_heads=audio_num_attention_heads,
            dim_head=audio_attention_head_dim,
            bias=attention_bias,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
            rope_type=rope_type,
            apply_gated_attention=audio_gated_attn,
            pack_qkv=False,
            quant_config=quant_config,
            prefix=f"{prefix}.audio_attn1" if prefix else "audio_attn1",
        )
        self.audio_norm2 = _make_rms_norm(audio_dim, eps=eps, elementwise_affine=elementwise_affine)
        self.audio_attn2 = LTX2Attention(
            query_dim=audio_dim,
            cross_attention_dim=audio_cross_attention_dim,
            heads=audio_num_attention_heads,
            kv_heads=audio_num_attention_heads,
            dim_head=audio_attention_head_dim,
            bias=attention_bias,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
            rope_type=rope_type,
            apply_gated_attention=audio_gated_attn,
            quant_config=quant_config,
            prefix=f"{prefix}.audio_attn2" if prefix else "audio_attn2",
            disable_kv_quant=True,
        )
        self.audio_norm3 = _make_rms_norm(audio_dim, eps=eps, elementwise_affine=elementwise_affine)
        self.audio_ff = LTX2FeedForward(
            audio_dim,
            activation_fn=activation_fn,
            bias=audio_ff_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.audio_ff" if prefix else "audio_ff",
        )

        num_mod_params = 9 if audio_cross_attn_adaln else 6
        self.audio_scale_shift_table = nn.Parameter(torch.randn(num_mod_params, audio_dim) / audio_dim**0.5)
        if audio_cross_attn_adaln:
            self.audio_prompt_scale_shift_table = nn.Parameter(torch.randn(2, audio_dim))

    @staticmethod
    def _get_mod_params(
        scale_shift_table: torch.Tensor,
        temb: torch.Tensor,
        batch_size: int,
    ) -> tuple[torch.Tensor, ...]:
        count = scale_shift_table.shape[0]
        values = scale_shift_table[None, None].to(temb.device) + temb.reshape(batch_size, temb.shape[1], count, -1)
        return values.unbind(dim=2)

    def forward(
        self,
        audio_hidden_states: torch.Tensor,
        audio_encoder_hidden_states: torch.Tensor,
        *,
        temb_audio: torch.Tensor,
        temb_prompt_audio: torch.Tensor | None = None,
        audio_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        audio_encoder_attention_mask: torch.Tensor | None = None,
        audio_self_attention_mask: torch.Tensor | None = None,
        audio_self_attention_perturbation_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = audio_hidden_states.shape[0]
        params = self._get_mod_params(self.audio_scale_shift_table, temb_audio, batch_size)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = params[:6]

        norm_states = self.audio_norm1(audio_hidden_states) * (1 + scale_msa) + shift_msa
        attended = self.audio_attn1(
            norm_states,
            query_rotary_emb=audio_rotary_emb,
            attention_mask=audio_self_attention_mask,
            perturbation_mask=audio_self_attention_perturbation_mask,
        )
        audio_hidden_states = audio_hidden_states + attended * gate_msa

        norm_states = self.audio_norm2(audio_hidden_states)
        if self.audio_cross_attn_adaln:
            shift_text_q, scale_text_q, gate_text_q = params[6:9]
            norm_states = norm_states * (1 + scale_text_q) + shift_text_q
            if temb_prompt_audio is None:
                shift_text_kv, scale_text_kv = (
                    self.audio_prompt_scale_shift_table[None, None]
                    .to(device=audio_hidden_states.device, dtype=audio_hidden_states.dtype)
                    .unbind(dim=2)
                )
            else:
                shift_text_kv, scale_text_kv = self._get_mod_params(
                    self.audio_prompt_scale_shift_table,
                    temb_prompt_audio,
                    batch_size,
                )
            audio_encoder_hidden_states = audio_encoder_hidden_states * (1 + scale_text_kv) + shift_text_kv

        attended = self.audio_attn2(
            norm_states,
            encoder_hidden_states=audio_encoder_hidden_states,
            attention_mask=audio_encoder_attention_mask,
        )
        if self.audio_cross_attn_adaln:
            attended = attended * gate_text_q
        audio_hidden_states = audio_hidden_states + attended

        norm_states = self.audio_norm3(audio_hidden_states) * (1 + scale_mlp) + shift_mlp
        return audio_hidden_states + self.audio_ff(norm_states) * gate_mlp


class LTX2AudioTransformerModel(nn.Module):
    """LTX full-checkpoint Transformer with every video path removed."""

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["norm"]
    _repeated_blocks = ["LTX2AudioTransformerBlock"]
    _layerwise_offload_blocks_attrs = ["transformer_blocks"]
    _hsdp_shard_conditions = [is_transformer_block_module]
    stacked_params_mapping = (
        (".audio_attn1.to_qkv", ".audio_attn1.to_q", "q"),
        (".audio_attn1.to_qkv", ".audio_attn1.to_k", "k"),
        (".audio_attn1.to_qkv", ".audio_attn1.to_v", "v"),
    )
    packed_modules_mapping = {"to_qkv": ["to_q", "to_k", "to_v"]}

    @staticmethod
    def _build_sp_plan(rope_type: str) -> dict[str, Any]:
        rope_expected_dims, rope_split_dim = (4, 2) if rope_type == "split" else (3, 1)
        return {
            "": {
                "audio_hidden_states": SequenceParallelInput(split_dim=1, expected_dims=3, split_output=False),
                "audio_encoder_hidden_states": SequenceParallelInput(split_dim=1, expected_dims=3, split_output=False),
                "audio_timestep": SequenceParallelInput(split_dim=1, expected_dims=2, split_output=False),
            },
            "audio_rope": {
                0: SequenceParallelInput(split_dim=rope_split_dim, expected_dims=rope_expected_dims, split_output=True),
                1: SequenceParallelInput(split_dim=rope_split_dim, expected_dims=rope_expected_dims, split_output=True),
            },
            "audio_proj_out": SequenceParallelOutput(gather_dim=1, expected_dims=3),
        }

    def __init__(
        self,
        audio_in_channels: int = 128,
        audio_out_channels: int | None = 128,
        audio_patch_size: int = 1,
        audio_patch_size_t: int = 1,
        audio_num_attention_heads: int = 32,
        audio_attention_head_dim: int = 64,
        audio_cross_attention_dim: int = 2048,
        audio_scale_factor: int = 4,
        audio_pos_embed_max_pos: int = 20,
        audio_sampling_rate: int = 16000,
        audio_hop_length: int = 160,
        num_layers: int = 48,
        activation_fn: str = "gelu-approximate",
        qk_norm: str = "rms_norm_across_heads",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        caption_channels: int = 3840,
        attention_bias: bool = True,
        attention_out_bias: bool = True,
        rope_theta: float = 10000.0,
        rope_double_precision: bool = True,
        causal_offset: int = 1,
        timestep_scale_multiplier: int = 1000,
        rope_type: str = "interleaved",
        use_prompt_embeddings: bool = True,
        perturbed_attn: bool = False,
        audio_gated_attn: bool = False,
        audio_cross_attn_mod: bool = False,
        audio_ff_bias: bool = True,
        use_prompt_adaln_single: bool = True,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        audio_out_channels = audio_out_channels or audio_in_channels
        audio_inner_dim = audio_num_attention_heads * audio_attention_head_dim
        self.config = SimpleNamespace(
            audio_in_channels=audio_in_channels,
            audio_out_channels=audio_out_channels,
            audio_patch_size=audio_patch_size,
            audio_patch_size_t=audio_patch_size_t,
            audio_num_attention_heads=audio_num_attention_heads,
            audio_attention_head_dim=audio_attention_head_dim,
            audio_cross_attention_dim=audio_cross_attention_dim,
            audio_scale_factor=audio_scale_factor,
            audio_pos_embed_max_pos=audio_pos_embed_max_pos,
            audio_sampling_rate=audio_sampling_rate,
            audio_hop_length=audio_hop_length,
            timestep_scale_multiplier=timestep_scale_multiplier,
            rope_type=rope_type,
            use_prompt_adaln_single=use_prompt_adaln_single,
        )
        self.prompt_modulation = audio_cross_attn_mod
        self.perturbed_attn = perturbed_attn

        self.audio_proj_in = nn.Linear(audio_in_channels, audio_inner_dim)
        if use_prompt_embeddings:
            self.audio_caption_projection = PixArtAlphaTextProjection(
                in_features=caption_channels,
                hidden_size=audio_inner_dim,
            )
        num_mod_params = 9 if audio_cross_attn_mod else 6
        self.audio_time_embed = LTX2AdaLayerNormSingle(
            audio_inner_dim,
            num_mod_params=num_mod_params,
            use_additional_conditions=False,
        )
        if audio_cross_attn_mod and use_prompt_adaln_single:
            self.audio_prompt_adaln = LTX2AdaLayerNormSingle(
                audio_inner_dim,
                num_mod_params=2,
                use_additional_conditions=False,
            )
        self.audio_scale_shift_table = nn.Parameter(torch.randn(2, audio_inner_dim) / audio_inner_dim**0.5)
        self.audio_rope = LTX2AudioVideoRotaryPosEmbed(
            dim=audio_inner_dim,
            patch_size=audio_patch_size,
            patch_size_t=audio_patch_size_t,
            base_num_frames=audio_pos_embed_max_pos,
            sampling_rate=audio_sampling_rate,
            hop_length=audio_hop_length,
            scale_factors=[audio_scale_factor],
            theta=rope_theta,
            causal_offset=causal_offset,
            modality="audio",
            double_precision=rope_double_precision,
            rope_type=rope_type,
            num_attention_heads=audio_num_attention_heads,
        )
        self.transformer_blocks = nn.ModuleList(
            [
                LTX2AudioTransformerBlock(
                    audio_dim=audio_inner_dim,
                    audio_num_attention_heads=audio_num_attention_heads,
                    audio_attention_head_dim=audio_attention_head_dim,
                    audio_cross_attention_dim=audio_cross_attention_dim,
                    audio_gated_attn=audio_gated_attn,
                    audio_cross_attn_adaln=audio_cross_attn_mod,
                    qk_norm=qk_norm,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    attention_out_bias=attention_out_bias,
                    eps=norm_eps,
                    elementwise_affine=norm_elementwise_affine,
                    rope_type=rope_type,
                    perturbed_attn=perturbed_attn,
                    audio_ff_bias=audio_ff_bias,
                    quant_config=quant_config,
                    prefix=f"transformer_blocks.{index}",
                )
                for index in range(num_layers)
            ]
        )
        self.audio_norm_out = nn.LayerNorm(audio_inner_dim, eps=1e-6, elementwise_affine=False)
        self.audio_proj_out = nn.Linear(audio_inner_dim, audio_out_channels)
        self.gradient_checkpointing = False
        self._sp_plan = self._build_sp_plan(rope_type)

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = False

    def prepare_static_conditioning(
        self,
        audio_encoder_hidden_states: torch.Tensor,
        audio_coords: torch.Tensor,
        *,
        hidden_dtype: torch.dtype,
    ) -> LTX2AudioStaticConditioning:
        """Project prompt context and build RoPE once for one request."""
        batch_size = audio_encoder_hidden_states.shape[0]
        if hasattr(self, "audio_caption_projection"):
            audio_encoder_hidden_states = self.audio_caption_projection(audio_encoder_hidden_states)
            audio_encoder_hidden_states = audio_encoder_hidden_states.view(
                batch_size,
                -1,
                self.audio_proj_in.out_features,
            )
        audio_rotary_emb = self.audio_rope(
            audio_coords,
            device=audio_encoder_hidden_states.device,
            out_dtype=hidden_dtype,
        )
        return LTX2AudioStaticConditioning(
            encoder_hidden_states=audio_encoder_hidden_states,
            rotary_emb=audio_rotary_emb,
        )

    def forward(
        self,
        audio_hidden_states: torch.Tensor,
        audio_encoder_hidden_states: torch.Tensor,
        audio_timestep: torch.Tensor,
        *,
        audio_sigma: torch.Tensor | None = None,
        audio_encoder_attention_mask: torch.Tensor | None = None,
        audio_attention_mask: torch.Tensor | None = None,
        audio_num_frames: int | None = None,
        audio_coords: torch.Tensor | None = None,
        audio_static_conditioning: LTX2AudioStaticConditioning | None = None,
        attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = False,
    ) -> torch.Tensor:
        del return_dict
        if audio_sigma is None:
            scaled_sigma = audio_timestep[:, 0] if audio_timestep.ndim > 1 else audio_timestep
            audio_sigma = scaled_sigma / self.config.timestep_scale_multiplier
        if audio_encoder_attention_mask is not None and audio_encoder_attention_mask.ndim == 2:
            audio_encoder_attention_mask = (1 - audio_encoder_attention_mask.to(audio_hidden_states.dtype)) * -10000
            audio_encoder_attention_mask = audio_encoder_attention_mask.unsqueeze(1)

        batch_size = audio_hidden_states.shape[0]
        if audio_static_conditioning is None:
            if audio_coords is None:
                if audio_num_frames is None:
                    raise ValueError("`audio_num_frames` is required when `audio_coords` is not provided.")
                audio_coords = self.audio_rope.prepare_audio_coords(
                    batch_size,
                    audio_num_frames,
                    audio_hidden_states.device,
                )
            audio_static_conditioning = self.prepare_static_conditioning(
                audio_encoder_hidden_states,
                audio_coords,
                hidden_dtype=audio_hidden_states.dtype,
            )
        audio_encoder_hidden_states = audio_static_conditioning.encoder_hidden_states
        audio_rotary_emb = audio_static_conditioning.rotary_emb
        audio_hidden_states = self.audio_proj_in(audio_hidden_states)
        temb_audio, embedded_timestep = self.audio_time_embed(
            audio_timestep.flatten(),
            batch_size=batch_size,
            hidden_dtype=audio_hidden_states.dtype,
        )
        temb_audio = temb_audio.view(batch_size, -1, temb_audio.shape[-1])
        embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.shape[-1])

        if self.prompt_modulation and self.config.use_prompt_adaln_single:
            # Match the official preprocessor: prompt-side AdaLN consumes the
            # scheduler sigma on the same 0..1000 scale as token timesteps.
            temb_prompt_audio, _ = self.audio_prompt_adaln(
                audio_sigma.flatten() * self.config.timestep_scale_multiplier,
                batch_size=batch_size,
                hidden_dtype=audio_hidden_states.dtype,
            )
            temb_prompt_audio = temb_prompt_audio.view(batch_size, -1, temb_prompt_audio.shape[-1])
        else:
            temb_prompt_audio = None
        perturbation_kwargs = (attention_kwargs or {}).get("ltx_perturbation_kwargs", {})
        for index, block in enumerate(self.transformer_blocks):
            perturbation_mask = perturbation_kwargs.get("audio_self_attention_mask")
            blocks = perturbation_kwargs.get("audio_self_attention_blocks")
            if blocks is not None and index not in blocks:
                perturbation_mask = None
            kwargs = {
                "temb_audio": temb_audio,
                "temb_prompt_audio": temb_prompt_audio,
                "audio_rotary_emb": audio_rotary_emb,
                "audio_encoder_attention_mask": audio_encoder_attention_mask,
                "audio_self_attention_mask": audio_attention_mask,
                "audio_self_attention_perturbation_mask": perturbation_mask,
            }
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                audio_hidden_states = checkpoint(
                    block,
                    audio_hidden_states,
                    audio_encoder_hidden_states,
                    use_reentrant=False,
                    **kwargs,
                )
            else:
                audio_hidden_states = block(audio_hidden_states, audio_encoder_hidden_states, **kwargs)

        values = self.audio_scale_shift_table[None, None] + embedded_timestep[:, :, None]
        shift, scale = values[:, :, 0], values[:, :, 1]
        audio_hidden_states = self.audio_norm_out(audio_hidden_states) * (1 + scale) + shift
        return self.audio_proj_out(audio_hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load only names represented by the audio-only module tree."""
        params = dict(self.named_parameters())
        try:
            tp_size = get_tensor_model_parallel_world_size()
        except AssertionError:
            # Unit loading and CPU inspection can happen before distributed
            # parallel-state initialization.
            tp_size = 1
        tp_rank = get_tensor_model_parallel_rank() if tp_size > 1 else 0
        loaded: set[str] = set()

        def maybe_shard(weight: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
            if tp_size <= 1 or weight.shape == param.shape:
                return weight
            if weight.ndim == 1 and weight.numel() == param.numel() * tp_size:
                return weight.chunk(tp_size, dim=0)[tp_rank]
            if weight.ndim == 2:
                if weight.shape[0] == param.shape[0] * tp_size:
                    return weight.chunk(tp_size, dim=0)[tp_rank]
                if weight.shape[1] == param.shape[1] * tp_size:
                    return weight.chunk(tp_size, dim=1)[tp_rank]
            return weight

        for name, weight in weights:
            for packed_name, source_name, shard_id in self.stacked_params_mapping:
                if source_name not in name:
                    continue
                target = name.replace(source_name, packed_name)
                if target not in params:
                    continue
                param = params[target]
                param.weight_loader(param, weight, shard_id)
                loaded.add(target)
                break
            else:
                if name not in params:
                    continue
                param = params[name]
                loader = getattr(param, "weight_loader", None)
                if loader is not None:
                    loader(param, weight)
                else:
                    default_weight_loader(param, maybe_shard(weight, param))
                loaded.add(name)
        return loaded
