# SPDX-License-Identifier: Apache-2.0
"""Causal three-way Cosmos3 transformer used by Cosmos-Dreams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from vllm.distributed import get_tensor_model_parallel_world_size

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3 import (
    Cosmos3CrossAttention,
    Cosmos3GenDecoderLayer,
    Cosmos3VFMTransformer,
    _apply_rotary_pos_emb,
    _tf_config_get,
)
from vllm_omni.diffusion.models.cosmos_dreams.config import CosmosDreamsManifest
from vllm_omni.diffusion.models.cosmos_dreams.geometry import CosmosDreamsGeometry
from vllm_omni.diffusion.models.cosmos_dreams.utils import (
    build_interleaved_mrope_position_ids,
    interleave_action_vision_tokens,
    split_interleaved_action_vision_tokens,
    zero_null_action_values,
)
from vllm_omni.experimental.ar_diffusion.kv_cache.paged_attention import (
    ARDiffusionPagedLayerInputs,
    paged_write_attn,
)
from vllm_omni.platforms import current_omni_platform


@dataclass
class CosmosDreamsTransformerOutput:
    video: torch.Tensor
    current_kv: list[tuple[torch.Tensor, torch.Tensor]]


class CosmosDreamsJointAttention(Cosmos3CrossAttention):
    """One softmax over text, committed history, and the current chunk."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        text_k: torch.Tensor,
        text_v: torch.Tensor,
        real_text_kv_len: int,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        dense_history: tuple[torch.Tensor, torch.Tensor] | None = None,
        paged_context: ARDiffusionPagedLayerInputs | None = None,
        num_frames: int,
        tokens_per_frame: int,
        action_tokens_per_frame: int,
        null_action_frame_indexes: tuple[int, ...] = (),
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if hidden_states.shape[0] != 1:
            raise ValueError(f"Cosmos-Dreams causal attention supports batch_size=1, got {hidden_states.shape[0]}")
        if real_text_kv_len <= 0 or real_text_kv_len > text_k.shape[1]:
            raise ValueError(
                "Cosmos-Dreams real text KV length must be in the stored range, "
                f"got real={real_text_kv_len}, stored={text_k.shape[1]}"
            )
        if text_k.shape != text_v.shape:
            raise ValueError(f"Cosmos-Dreams text K/V shapes differ: {tuple(text_k.shape)} != {tuple(text_v.shape)}")
        if dense_history is not None and paged_context is not None:
            raise ValueError("Cosmos-Dreams attention accepts either dense_history or paged_context, not both")

        batch, seq_len, _ = hidden_states.shape
        q = self.to_q(hidden_states).view(batch, seq_len, self.num_heads_local, self.head_dim)
        k = self.to_k(hidden_states).view(batch, seq_len, self.num_kv_heads_local, self.head_dim)
        v = self.to_v(hidden_states).view(batch, seq_len, self.num_kv_heads_local, self.head_dim)
        if self.qk_norm:
            q = F.rms_norm(q, (self.head_dim,), self.norm_q.weight, eps=self.norm_q.variance_epsilon)
            k = F.rms_norm(k, (self.head_dim,), self.norm_k.weight, eps=self.norm_k.variance_epsilon)
        q, k = _apply_rotary_pos_emb(q, k, freqs_cos, freqs_sin)
        v = zero_null_action_values(
            v,
            num_frames=num_frames,
            tokens_per_frame=tokens_per_frame,
            action_tokens_per_frame=action_tokens_per_frame,
            null_frame_indexes=null_action_frame_indexes,
        )

        if paged_context is not None:
            output = paged_write_attn(
                paged_context,
                q[0],
                k[0],
                v[0],
                text_k[0, :real_text_kv_len],
                text_v[0, :real_text_kv_len],
                self.head_dim**-0.5,
            ).unsqueeze(0)
        else:
            key_parts = [text_k[:, :real_text_kv_len]]
            value_parts = [text_v[:, :real_text_kv_len]]
            if dense_history is not None:
                history_k, history_v = dense_history
                if history_k.shape[1]:
                    key_parts.append(history_k)
                    value_parts.append(history_v)
            key_parts.append(k)
            value_parts.append(v)
            output = self.attn(q, torch.cat(key_parts, dim=1), torch.cat(value_parts, dim=1))

        return self.to_out(output.reshape(batch, seq_len, -1)), k, v


class CosmosDreamsGenDecoderLayer(Cosmos3GenDecoderLayer):
    """Cosmos3 GEN layer with causal joint attention and unchanged weight names."""

    def __init__(
        self,
        *,
        layer_idx: int | None = None,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rms_norm_eps: float,
        quant_config=None,
        mlp_cls,
        qk_norm: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__(
            layer_idx=layer_idx,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            rms_norm_eps=rms_norm_eps,
            quant_config=quant_config,
            mlp_cls=mlp_cls,
            qk_norm=qk_norm,
            prefix=prefix,
        )
        self.cross_attention = CosmosDreamsJointAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            rms_norm_eps=rms_norm_eps,
            quant_config=quant_config,
            qk_norm=qk_norm,
            prefix=f"{prefix}.cross_attention",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        **attention_kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = hidden_states
        attention_input = self.input_layernorm(hidden_states)
        attention_output, current_k, current_v = self.cross_attention(
            attention_input,
            **attention_kwargs,
        )
        hidden_states = residual + attention_output
        residual = hidden_states
        hidden_states = residual + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, current_k, current_v


class CosmosDreamsTransformer(Cosmos3VFMTransformer):
    """Cosmos3 MoT generator with persistent causal GEN K/V history."""

    _gen_layer_cls = CosmosDreamsGenDecoderLayer
    _repeated_blocks = ["CosmosDreamsGenDecoderLayer"]

    def _language_model_kwargs(self) -> dict[str, Any]:
        return {"use_und_k_norm_for_gen": bool(self.use_und_k_norm_for_gen)}

    def validate_loaded_weights(self, loaded: set[str]) -> None:
        required = {f"transformer.{name}" for name, _ in self.named_parameters()}
        missing = sorted(required - loaded)
        if missing:
            preview = ", ".join(missing[:12])
            suffix = "" if len(missing) <= 12 else f" (and {len(missing) - 12} more)"
            raise ValueError(f"Cosmos-Dreams checkpoint is missing required transformer weights: {preview}{suffix}")

    @staticmethod
    def _validate_supported_config(model_config: Any) -> None:
        expected_values = {
            "qk_norm_for_diffusion": True,
            "qk_norm_for_text": True,
            "position_embedding_type": "unified_3d_mrope",
            "unified_3d_mrope_reset_spatial_ids": True,
            "joint_attn_implementation": "three_way",
            "video_temporal_causal": True,
        }
        for key, expected in expected_values.items():
            actual = _tf_config_get(model_config, key, expected)
            if actual != expected:
                raise ValueError(
                    f"Unsupported Cosmos-Dreams transformer config: {key}={actual!r}; expected {expected!r}."
                )

    def __init__(
        self,
        od_config: OmniDiffusionConfig,
        *,
        temporal_compression_factor: int | None = None,
        sound_gen: bool = False,
        sound_dim: int | None = None,
        sound_latent_fps: float | None = None,
    ) -> None:
        self.manifest = CosmosDreamsManifest.from_od_config(od_config)
        self.manifest.require_exported_artifact()
        if sound_gen:
            raise ValueError("Cosmos-Dreams v1 does not support joint sound generation")
        super().__init__(
            od_config,
            temporal_compression_factor=temporal_compression_factor,
            sound_gen=sound_gen,
            sound_dim=sound_dim,
            sound_latent_fps=sound_latent_fps,
        )
        if not self.action_gen:
            raise ValueError("Cosmos-Dreams checkpoints must enable action_gen")
        if self.action_dim != self.manifest.max_action_dim:
            raise ValueError(
                "Cosmos-Dreams action dimension differs between transformer and manifest: "
                f"{self.action_dim} != {self.manifest.max_action_dim}"
            )
        if self.temporal_compression_factor != self.manifest.temporal_compression_factor:
            raise ValueError(
                "Cosmos-Dreams temporal compression differs between VAE and manifest: "
                f"{self.temporal_compression_factor} != {self.manifest.temporal_compression_factor}"
            )
        self.temporal_modality_margin = self.manifest.temporal_modality_margin
        self.base_fps = self.manifest.base_fps
        self.enable_fps_modulation = self.manifest.enable_fps_modulation

    @property
    def num_kv_heads_local(self) -> int:
        return self.num_key_value_heads // get_tensor_model_parallel_world_size()

    @staticmethod
    def pad_text_kv(
        layer_kv: list[tuple[torch.Tensor, torch.Tensor]],
        *,
        max_len: int,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        padded: list[tuple[torch.Tensor, torch.Tensor]] = []
        for key, value in layer_kv:
            if key.shape != value.shape or key.shape[0] != 1:
                raise ValueError(
                    f"Cosmos-Dreams text K/V must be matching batch-one tensors, got {key.shape} and {value.shape}"
                )
            if key.shape[1] > max_len:
                raise ValueError(f"Cosmos-Dreams prompt has {key.shape[1]} KV tokens but text_cache_max_len={max_len}")
            pad_len = max_len - key.shape[1]
            if pad_len:
                key = torch.cat([key, key.new_zeros(1, pad_len, *key.shape[2:])], dim=1)
                value = torch.cat([value, value.new_zeros(1, pad_len, *value.shape[2:])], dim=1)
            padded.append((key, value))
        return padded

    def _current_rope(
        self,
        hidden: torch.Tensor,
        *,
        frame_start: int,
        num_frames: int,
        grid_h: int,
        grid_w: int,
        real_text_kv_len: int,
        fps: float,
        null_action_frame_indexes: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        absolute_null_frames = tuple(frame_start + frame for frame in null_action_frame_indexes)
        position_ids = (
            build_interleaved_mrope_position_ids(
                frame_start=frame_start,
                num_frames=num_frames,
                grid_h=grid_h,
                grid_w=grid_w,
                text_temporal_offset=real_text_kv_len,
                temporal_modality_margin=self.temporal_modality_margin,
                fps=fps,
                base_fps=self.base_fps,
                temporal_compression_factor=self.temporal_compression_factor,
                enable_fps_modulation=self.enable_fps_modulation,
                action_tokens_per_frame=self.manifest.action_tokens_per_frame,
                null_action_frames=absolute_null_frames,
            )
            .unsqueeze(1)
            .to(hidden.device)
        )
        cos, sin = self.language_model.rotary_emb(hidden, position_ids=position_ids)
        return cos.unsqueeze(2), sin.unsqueeze(2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        *,
        geometry: CosmosDreamsGeometry,
        text_kv: list[tuple[torch.Tensor, torch.Tensor]],
        real_text_kv_len: int,
        frame_start: int,
        fps: float,
        action_latents: torch.Tensor | None,
        action_domain_ids: torch.Tensor,
        paged_kv: list[Any] | None = None,
        dense_history: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        condition_vision: bool = False,
        null_action_frame_indexes: tuple[int, ...] = (),
    ) -> CosmosDreamsTransformerOutput:
        """Denoise or clean-commit one current chunk.

        ``paged_kv`` is non-committing during denoise and committing only for a
        one-frame clean refresh. Dense history is the permanent numerical oracle.
        """
        if hidden_states.ndim != 5 or hidden_states.shape[0] != 1:
            raise ValueError(
                f"Cosmos-Dreams hidden_states must have shape [1,C,T,H,W], got {tuple(hidden_states.shape)}"
            )
        if timestep.shape not in {(1,), (1, 1)}:
            raise ValueError(f"Cosmos-Dreams timestep must have shape [1] or [1,1], got {tuple(timestep.shape)}")
        if len(text_kv) != self.num_hidden_layers:
            raise ValueError(f"Cosmos-Dreams expected {self.num_hidden_layers} text KV layers, got {len(text_kv)}")
        if paged_kv is not None and dense_history is not None:
            raise ValueError("Cosmos-Dreams forward accepts either paged_kv or dense_history, not both")
        if paged_kv is not None and len(paged_kv) != self.num_hidden_layers:
            raise ValueError(f"Cosmos-Dreams expected {self.num_hidden_layers} paged contexts, got {len(paged_kv)}")
        if dense_history is not None and len(dense_history) != self.num_hidden_layers:
            raise ValueError(
                f"Cosmos-Dreams expected {self.num_hidden_layers} dense history layers, got {len(dense_history)}"
            )

        _, _, num_frames, latent_h, latent_w = hidden_states.shape
        grid_h, grid_w, _, _ = self._pad_to_patch_size(latent_h, latent_w)
        if (latent_h, latent_w) != (geometry.latent_height, geometry.latent_width):
            raise RuntimeError(
                "Cosmos-Dreams transformer latent shape does not match resolved geometry: "
                f"{(latent_h, latent_w)} != {(geometry.latent_height, geometry.latent_width)}"
            )
        if (grid_h, grid_w) != geometry.patch_grid:
            raise RuntimeError(
                "Cosmos-Dreams transformer patch grid does not match resolved geometry: "
                f"{(grid_h, grid_w)} != {geometry.patch_grid}"
            )
        vision_tokens = self.patchify(hidden_states, num_frames, latent_h, latent_w)
        vision_tokens = self.proj_in(vision_tokens).view(
            1,
            num_frames,
            grid_h * grid_w,
            self.hidden_size,
        )
        if not condition_vision:
            with torch.autocast(current_omni_platform.device_type, enabled=False):
                time_embed = self.time_embedder((timestep.reshape(1) * self.timestep_scale).float())
            vision_tokens = vision_tokens + time_embed.to(vision_tokens.dtype).view(1, 1, 1, -1)

        action_count = self.manifest.action_tokens_per_frame
        actual_tokens_per_frame = action_count + grid_h * grid_w
        if action_latents is None:
            action_latents = hidden_states.new_zeros(1, num_frames * action_count, self.action_dim)
        elif action_latents.ndim == 2:
            action_latents = action_latents.unsqueeze(0)
        expected_action_shape = (1, num_frames * action_count, self.action_dim)
        if tuple(action_latents.shape) != expected_action_shape:
            raise ValueError(
                f"Cosmos-Dreams actions must have shape {expected_action_shape}, got {tuple(action_latents.shape)}"
            )
        action_hidden = self.action_proj_in(action_latents, action_domain_ids)
        action_hidden = action_hidden + self.action_modality_embed.to(action_hidden.dtype)
        action_hidden = action_hidden.view(1, num_frames, action_count, self.hidden_size)
        hidden = interleave_action_vision_tokens(action_hidden, vision_tokens)
        freqs_cos, freqs_sin = self._current_rope(
            hidden,
            frame_start=frame_start,
            num_frames=num_frames,
            grid_h=grid_h,
            grid_w=grid_w,
            real_text_kv_len=real_text_kv_len,
            fps=fps,
            null_action_frame_indexes=null_action_frame_indexes,
        )

        if paged_kv is not None:
            forward_context = paged_kv[0].forward_ctx
            if forward_context.seq_len != hidden.shape[1]:
                raise RuntimeError(
                    "Cosmos-Dreams paged context token count does not match "
                    f"this chunk: {forward_context.seq_len} != {hidden.shape[1]}"
                )
            forward_context.prepare(
                device=hidden.device,
                action_len=real_text_kv_len,
                query_len=hidden.shape[1],
            )
            paged_kv = [layer_context.to_layer_inputs() for layer_context in paged_kv]

        # Only the dense path reads current_kv; in paged mode the K/V are
        # already in the pool, so keeping 40 layers of them alive per denoise
        # step would defeat the point of paging.
        collect_current_kv = paged_kv is None
        current_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
        with self._offload_context("generator"):
            for layer_idx, layer in enumerate(self.gen_layers):
                text_k, text_v = text_kv[layer_idx]
                layer_output = layer(
                    hidden,
                    text_k=text_k,
                    text_v=text_v,
                    real_text_kv_len=real_text_kv_len,
                    freqs_cos=freqs_cos,
                    freqs_sin=freqs_sin,
                    dense_history=None if dense_history is None else dense_history[layer_idx],
                    paged_context=None if paged_kv is None else paged_kv[layer_idx],
                    num_frames=num_frames,
                    tokens_per_frame=actual_tokens_per_frame,
                    action_tokens_per_frame=action_count,
                    null_action_frame_indexes=null_action_frame_indexes,
                )
                hidden, current_k, current_v = layer_output
                if collect_current_kv:
                    current_kv.append((current_k, current_v))
            hidden = self.norm_moe_gen(hidden)
            _, vision_hidden = split_interleaved_action_vision_tokens(
                hidden,
                num_frames=num_frames,
                action_tokens_per_frame=action_count,
                vision_tokens_per_frame=grid_h * grid_w,
            )
            vision_hidden = vision_hidden.flatten(1, 2)
            video = self.unpatchify(self.proj_out(vision_hidden), num_frames, latent_h, latent_w)
        return CosmosDreamsTransformerOutput(video=video, current_kv=current_kv)
