# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage-local Wan DiT for WaveServe chunk serving.

Reuses ``wan2_2.WanTransformer3DModel`` (same diffusers Wan 2.1 weight layout /
``load_weights``) and runs real 5D latent steps via ``forward_latent_step``.
Layer split follows WaveServe ``G`` (not Omni ``S·G`` PP ranks) via explicit
``layer_pp_rank`` / ``layer_pp_world`` on the Wan builder.

``forward_latent_step`` is G-aware: stage-first patches, middle groups resume
from ``IntermediateTensors``, stage-last unpatches. Activation packs carry
``latent`` (and ``hidden_states`` between groups) along the PP chain.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors

from vllm_omni.experimental.ar_diffusion.kv_cache.paged_attention import paged_write_attn

logger = init_logger(__name__)

# Wan 2.1 1.3B T2V defaults (diffusers config field names).
_WAN21_1_3B_CONFIG: dict[str, Any] = {
    "patch_size": [1, 2, 2],
    "num_attention_heads": 12,
    "attention_head_dim": 128,
    "in_channels": 16,
    "out_channels": 16,
    "text_dim": 4096,
    "freq_dim": 256,
    "ffn_dim": 8960,
    "num_layers": 30,
    "cross_attn_norm": True,
    "eps": 1e-6,
    "rope_max_seq_len": 1024,
}


def stage_layer_range(num_layers: int, group: int, groups: int) -> tuple[int, int]:
    """Even split of ``num_layers`` across ``groups`` (stage-local G, not pp_world)."""
    if groups < 1:
        raise ValueError("groups must be positive")
    if not 0 <= group < groups:
        raise ValueError(f"group {group} out of range for {groups}")
    return (num_layers * group) // groups, (num_layers * (group + 1)) // groups


def _wan_self_attn_with_kv(
    attn: nn.Module,
    hidden_states: torch.Tensor,
    rotary_emb: tuple[torch.Tensor, torch.Tensor] | None,
    kv_ctx: Any | None,
) -> torch.Tensor:
    """WanSelfAttention math with optional NoisyKV / paged write+attend."""
    qkv, _ = attn.to_qkv(hidden_states)
    q_size = attn.num_heads * attn.head_dim
    kv_size = attn.num_kv_heads * attn.head_dim
    query, key, value = qkv.split([q_size, kv_size, kv_size], dim=-1)
    query = attn.norm_q(query)
    key = attn.norm_k(key)
    query = query.unflatten(2, (attn.num_heads, attn.head_dim))
    key = key.unflatten(2, (attn.num_kv_heads, attn.head_dim))
    value = value.unflatten(2, (attn.num_kv_heads, attn.head_dim))
    if rotary_emb is not None:
        freqs_cos, freqs_sin = rotary_emb
        query = attn.rotary_embedding(query, freqs_cos, freqs_sin)
        key = attn.rotary_embedding(key, freqs_cos, freqs_sin)
    scale = 1.0 / (attn.head_dim**0.5)
    if kv_ctx is not None:
        inputs = kv_ctx.to_layer_inputs() if hasattr(kv_ctx, "to_layer_inputs") else kv_ctx
        outs = [paged_write_attn(inputs, query[i], key[i], value[i], None, None, scale) for i in range(query.shape[0])]
        hidden_states = torch.stack(outs, dim=0).flatten(2, 3).type_as(query)
    else:
        hidden_states = attn.attn(query, key, value, None)
        hidden_states = hidden_states.flatten(2, 3).type_as(query)
    hidden_states = attn.to_out(hidden_states)
    hidden_states = attn.dropout(hidden_states)
    return hidden_states


class StageWanTransformer(nn.Module):
    """Wan DiT with stage-local G slices and paged Latest-KV on self-attention."""

    def __init__(
        self,
        *,
        num_layers: int = 30,
        dim: int = 1536,
        num_heads: int = 12,
        ffn_dim: int = 8960,
        in_channels: int = 16,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        layer_groups: int = 1,
        pp_rank: int = 0,
        transformer_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.layer_groups = max(1, int(layer_groups))
        self.pp_rank = int(pp_rank)
        self.group = 0 if self.layer_groups == 1 else self.pp_rank % self.layer_groups
        self.is_stage_first = self.group == 0
        self.is_stage_last = self.group == self.layer_groups - 1
        self.patch_size = tuple(patch_size)
        self.in_features = in_channels * math.prod(self.patch_size)
        self._wan: nn.Module | None = None
        self._init_wan(
            transformer_config=transformer_config,
            fallback_geo={
                "num_layers": num_layers,
                "dim": dim,
                "num_heads": num_heads,
                "ffn_dim": ffn_dim,
                "in_channels": in_channels,
                "patch_size": self.patch_size,
            },
        )

    def _init_wan(self, *, transformer_config: dict[str, Any] | None, fallback_geo: dict[str, Any]) -> None:
        from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import create_transformer_from_config

        cfg = dict(_WAN21_1_3B_CONFIG)
        if transformer_config:
            cfg.update({k: v for k, v in transformer_config.items() if v is not None})
        else:
            # Geometry kwargs from pipeline when no HF config is available yet.
            patch = fallback_geo["patch_size"]
            heads = int(fallback_geo["num_heads"])
            dim = int(fallback_geo["dim"])
            cfg.update(
                {
                    "patch_size": list(patch),
                    "num_attention_heads": heads,
                    "attention_head_dim": dim // heads,
                    "in_channels": int(fallback_geo["in_channels"]),
                    "out_channels": int(fallback_geo["in_channels"]),
                    "ffn_dim": int(fallback_geo["ffn_dim"]),
                    "num_layers": int(fallback_geo["num_layers"]),
                }
            )

        wan = create_transformer_from_config(
            cfg,
            layer_pp_rank=self.group,
            layer_pp_world=self.layer_groups,
        )

        self._wan = wan
        self.num_layers = int(cfg["num_layers"])
        self.num_heads = int(cfg["num_attention_heads"])
        self.head_dim = int(cfg["attention_head_dim"])
        self.dim = self.num_heads * self.head_dim
        self.start_layer = int(wan.start_layer)
        self.end_layer = int(wan.end_layer)
        patch = tuple(cfg["patch_size"])
        self.patch_size = patch
        self.in_features = int(cfg["in_channels"]) * math.prod(patch)
        self.schedule_patch_embed = None
        logger.info(
            "StageWanTransformer: wan2_2 WanTransformer3DModel local_layers=[%d, %d) G=%d group=%d",
            self.start_layer,
            self.end_layer,
            self.layer_groups,
            self.group,
        )

    @property
    def local_num_layers(self) -> int:
        return max(0, self.end_layer - self.start_layer)

    @property
    def wan(self) -> nn.Module | None:
        return self._wan

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str] | None:
        """Load Diffusers / Omni Wan weights into the DiT.

        Returns ``None`` so the Omni loader skips strict name coverage (inner
        Wan ``load_weights`` names are relative to ``_wan``).
        """
        if self._wan is None:
            return None

        def _strip() -> Iterable[tuple[str, torch.Tensor]]:
            for name, tensor in weights:
                key = name
                for prefix in ("transformer.", "model.transformer.", "dit."):
                    if key.startswith(prefix):
                        key = key[len(prefix) :]
                        break
                if key.startswith("_wan."):
                    key = key[len("_wan.") :]
                yield key, tensor

        self._wan.load_weights(_strip())
        return None

    def seq_len_for_latent(self, latent_shape: tuple[int, ...]) -> int:
        """Token count after Wan patch embed for ``latent`` ``(B,C,T,H,W)``."""
        if len(latent_shape) != 5:
            raise ValueError(f"expected B,C,T,H,W; got {latent_shape}")
        _, _, t, h, w = latent_shape
        pt, ph, pw = self.patch_size
        if t % pt or h % ph or w % pw:
            raise ValueError(f"latent {latent_shape} not aligned to patch {self.patch_size}")
        return (t // pt) * (h // ph) * (w // pw)

    def forward_latent_step(
        self,
        latent: torch.Tensor,
        *,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        kv_contexts: list[Any] | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        """One denoise (or clean) pass over this rank's layer group.

        Always needs 5D ``latent`` (RoPE / grid shape). Stage-first patches;
        non-first resumes from ``intermediate_tensors["hidden_states"]``.
        Stage-last unpatches to a 5D velocity/noise pred; non-last returns
        ``IntermediateTensors`` for the next group on the PP chain.

        Self-attn is hooked to NoisyKV when ``kv_contexts`` is provided.
        ``G=1`` (first and last) is the full DiT path unchanged.
        """
        if self._wan is None:
            raise RuntimeError("forward_latent_step requires the Wan DiT")
        if latent.ndim != 5:
            raise ValueError(f"latent must be B,C,T,H,W; got shape {tuple(latent.shape)}")
        wan = self._wan
        batch_size, _c, num_frames, height, width = latent.shape
        p_t, p_h, p_w = self.patch_size
        post_t, post_h, post_w = num_frames // p_t, height // p_h, width // p_w

        # RoPE from 5D latent on every group (same as wan2_2 PP).
        freqs_cos, freqs_sin = wan.rope(latent)
        rotary_emb = (
            freqs_cos[..., 0::2].to(latent.dtype),
            freqs_sin[..., 1::2].to(latent.dtype),
        )

        if self.is_stage_first:
            if intermediate_tensors is not None:
                raise ValueError("stage-first layer group must not receive intermediate_tensors")
            hidden = wan.patch_embedding(latent)
            hidden = hidden.flatten(2).transpose(1, 2)
        else:
            if intermediate_tensors is None:
                raise RuntimeError('non-first layer group requires intermediate_tensors["hidden_states"]')
            hidden = intermediate_tensors["hidden_states"]

        if timestep.ndim == 0:
            timestep = timestep.expand(batch_size)
        elif timestep.ndim == 1 and timestep.shape[0] == 1 and batch_size > 1:
            timestep = timestep.expand(batch_size)

        # Conditioning on every group: each owns local blocks that need temb.
        temb, timestep_proj, enc, _enc_img = wan.condition_embedder(
            timestep, encoder_hidden_states, None, timestep_seq_len=None
        )
        timestep_proj = wan.timestep_proj_prepare(timestep_proj, None)

        local_count = self.local_num_layers
        for idx in range(self.start_layer, self.end_layer):
            block = wan.blocks[idx]
            ctx = None
            if kv_contexts is not None:
                ctx = kv_contexts[idx - self.start_layer] if len(kv_contexts) == local_count else kv_contexts[idx]
            attn1 = block.attn1
            orig_forward = attn1.forward

            def _hooked(
                hs: torch.Tensor,
                rotary_emb_arg: tuple[torch.Tensor, torch.Tensor] | None = None,
                attn_metadata: Any = None,
                *,
                _ctx: Any | None = ctx,
                _orig=orig_forward,
                _attn=attn1,
            ) -> torch.Tensor:
                del attn_metadata
                if _ctx is None:
                    return _orig(hs, rotary_emb_arg, None)
                return _wan_self_attn_with_kv(_attn, hs, rotary_emb_arg, _ctx)

            attn1.forward = _hooked  # type: ignore[method-assign]
            try:
                hidden = block(hidden, enc, timestep_proj, rotary_emb, None, None, False)
            finally:
                attn1.forward = orig_forward  # type: ignore[method-assign]

        if not self.is_stage_last:
            return IntermediateTensors({"hidden_states": hidden})

        shift, scale = wan.output_scale_shift_prepare(temb)
        shift = shift.to(hidden.device)
        scale = scale.to(hidden.device)
        if shift.ndim == 2:
            shift = shift.unsqueeze(1)
            scale = scale.unsqueeze(1)
        hidden = wan.norm_out(hidden, scale, shift).type_as(hidden)
        hidden = wan.proj_out(hidden)
        hidden = hidden.reshape(batch_size, post_t, post_h, post_w, p_t, p_h, p_w, -1)
        hidden = hidden.permute(0, 7, 1, 4, 2, 5, 3, 6)
        return hidden.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    def forward(
        self,
        hidden_states: torch.Tensor | IntermediateTensors | None = None,
        kv_contexts: list[Any] | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        del hidden_states, kv_contexts, intermediate_tensors
        raise RuntimeError(
            "StageWanTransformer must use forward_latent_step(5D latent); abstract token forward was removed"
        )
