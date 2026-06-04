# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.
# Copyright (c) Ant Group. All rights reserved.
# Ported from:
# https://github.com/inclusionAI/Ming/blob/e58533db227031990c5a6864dcf5f08fb53ed0d2/talker_module/dit.py
#
# Ported from:
# https://github.com/inclusionAI/Ming/blob/e58533db227031990c5a6864dcf5f08fb53ed0d2/talker_module/modules.py
# Ported from:
# https://github.com/inclusionAI/Ming/blob/e58533db227031990c5a6864dcf5f08fb53ed0d2/talker_module/cfm.py

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# Partial of the following source code
# is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
import logging
from contextlib import nullcontext
from dataclasses import dataclass
from functools import cached_property
from queue import Queue
from threading import Lock
from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase, Qwen2Config, Qwen2Model, StaticCache
from transformers.models.qwen2.modeling_qwen2 import (
    ALL_ATTENTION_FUNCTIONS,
    Qwen2Attention,
    eager_attention_forward,
)
from transformers.models.qwen2.modeling_qwen2 import (
    apply_rotary_pos_emb as qwen2_apply_rotary_pos_emb,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from x_transformers.x_transformers import RotaryEmbedding, apply_rotary_pos_emb, rotate_half

from vllm_omni.model_executor.layers.timestep_embedding import DiTTimestepEmbedding

from .audio_vae import AudioVAE

logger = init_logger(__name__)


def _record_function(name: str):
    return nullcontext()


def _apply_rotary_pos_emb_from_trig(
    t: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    scale: torch.Tensor | float = 1,
) -> torch.Tensor:
    rot_dim, seq_len, orig_dtype = cos.shape[-1], t.shape[-2], t.dtype

    cos = cos[:, -seq_len:, :]
    sin = sin[:, -seq_len:, :]
    scale = scale[:, -seq_len:, :] if torch.is_tensor(scale) else scale

    if t.ndim == 4 and cos.ndim == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        if torch.is_tensor(scale):
            scale = scale.unsqueeze(1)

    t, t_unrotated = t[..., :rot_dim], t[..., rot_dim:]
    if torch.is_tensor(scale) or scale != 1:
        t = (t * cos * scale) + (rotate_half(t) * sin * scale)
    else:
        t = (t * cos) + (rotate_half(t) * sin)
    if t_unrotated.shape[-1] > 0:
        t = torch.cat((t, t_unrotated), dim=-1)

    return t.type(orig_dtype)


def _apply_rotary_pos_emb_from_trig_seq_first(
    t: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    scale: torch.Tensor | float = 1,
) -> torch.Tensor:
    rot_dim, seq_len, orig_dtype = cos.shape[-1], t.shape[1], t.dtype

    cos = cos[:, -seq_len:, :].unsqueeze(2)
    sin = sin[:, -seq_len:, :].unsqueeze(2)
    if torch.is_tensor(scale):
        scale = scale[:, -seq_len:, :].unsqueeze(2)

    t, t_unrotated = t[..., :rot_dim], t[..., rot_dim:]
    if torch.is_tensor(scale) or scale != 1:
        t = (t * cos * scale) + (rotate_half(t) * sin * scale)
    else:
        t = (t * cos) + (rotate_half(t) * sin)
    if t_unrotated.shape[-1] > 0:
        t = torch.cat((t, t_unrotated), dim=-1)

    return t.type(orig_dtype)


########################################################################
# DiT Modules
# Ported from:
# https://github.com/inclusionAI/Ming/blob/e58533db227031990c5a6864dcf5f08fb53ed0d2/talker_module/modules.py
# Ported from:
# https://github.com/inclusionAI/Ming/blob/e58533db227031990c5a6864dcf5f08fb53ed0d2/talker_module/dit.py
########################################################################


@dataclass(slots=True)
class MingTalkerSlot:
    req_id: str | None = None


class MingTalkerSlotTable:
    def __init__(self, max_slots: int) -> None:
        if max_slots <= 0:
            raise ValueError(f"max_slots must be positive, got {max_slots}")
        self.slots = [MingTalkerSlot() for _ in range(max_slots)]
        self.req_to_slot: dict[str, int] = {}
        self.free_slots = list(range(max_slots - 1, -1, -1))

    def allocate(self, req_id: str) -> int:
        existing = self.req_to_slot.get(req_id)
        if existing is not None:
            return existing
        if not self.free_slots:
            raise RuntimeError("Ming Talker slot table exhausted")
        slot = self.free_slots.pop()
        self.req_to_slot[req_id] = slot
        self.slots[slot] = MingTalkerSlot(req_id)
        return slot

    def free(self, req_id: str) -> None:
        slot = self.req_to_slot.pop(req_id, None)
        if slot is None:
            return
        self.slots[slot] = MingTalkerSlot()
        self.free_slots.append(slot)

    def active_slots(self) -> list[int]:
        return [idx for idx, slot in enumerate(self.slots) if slot.req_id is not None]

    def active_request_ids(self) -> list[str]:
        return [self.slots[idx].req_id for idx in self.active_slots()]


@dataclass(slots=True)
class MingTalkerBatchPolicy:
    max_batch_size: int = 8
    max_wait_ms: float = 20.0
    bucket_sizes: tuple[int, ...] = (1, 2, 4, 8, 16)

    def choose_bucket(self, queued: int) -> int:
        if queued <= 0:
            return 0
        capped = min(queued, self.max_batch_size)
        for bucket in self.bucket_sizes:
            if capped <= bucket:
                return min(bucket, self.max_batch_size)
        return self.max_batch_size

    def should_dispatch(self, queued: int, oldest_wait_ms: float) -> bool:
        if queued <= 0:
            return False
        return queued >= self.max_batch_size or oldest_wait_ms >= self.max_wait_ms


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            x = x.to(self.weight.dtype)
        x = F.rms_norm(x, normalized_shape=(x.shape[-1],), weight=self.weight, eps=self.eps)
        return x


class FeedForward(nn.Module):
    def __init__(
        self, dim: int, dim_out: int | None = None, mult: float = 4, dropout: float = 0.0, approximate: str = "none"
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        activation = nn.GELU(approximate=approximate)
        project_in = nn.Sequential(nn.Linear(dim, inner_dim), activation)
        self.ff = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with _record_function("ming.feed_forward.in_proj_gelu"):
            x = self.ff[0](x)
        with _record_function("ming.feed_forward.dropout"):
            x = self.ff[1](x)
        with _record_function("ming.feed_forward.out_proj"):
            return self.ff[2](x)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        qk_norm: str | None = None,
        pe_attn_head: int | None = None,
        attn_mask_enabled: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.inner_dim = dim_head * heads
        self.dropout = dropout

        self.to_q = nn.Linear(dim, self.inner_dim)
        self.to_k = nn.Linear(dim, self.inner_dim)
        self.to_v = nn.Linear(dim, self.inner_dim)
        self.to_qkv: nn.Linear | None = None
        if qk_norm is None:
            self.q_norm = None
            self.k_norm = None
        elif qk_norm == "rms_norm":
            self.q_norm = RMSNorm(dim_head)
            self.k_norm = RMSNorm(dim_head)
        else:
            raise ValueError(f"Unimplemented qk_norm: {qk_norm}")

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(self.inner_dim, dim))
        self.to_out.append(nn.Dropout(dropout))

        self.pe_attn_head = pe_attn_head
        self.attn_mask_enabled = attn_mask_enabled

    def pack_qkv(self) -> None:
        if self.to_qkv is not None:
            return
        if self.to_q.bias is None or self.to_k.bias is None or self.to_v.bias is None:
            raise ValueError("Ming fused QKV packing expects q/k/v bias tensors")
        qkv = nn.Linear(
            self.dim,
            self.inner_dim * 3,
            bias=True,
            device=self.to_q.weight.device,
            dtype=self.to_q.weight.dtype,
        )
        with torch.no_grad():
            qkv.weight.copy_(torch.cat([self.to_q.weight, self.to_k.weight, self.to_v.weight], dim=0))
            qkv.bias.copy_(torch.cat([self.to_q.bias, self.to_k.bias, self.to_v.bias], dim=0))
        qkv.requires_grad_(False)
        self.to_qkv = qkv

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        rope: tuple[torch.Tensor, ...] | None = None,
    ) -> torch.Tensor:
        batch_size = x.shape[0]

        with _record_function("ming.attn.qkv_proj"):
            if self.to_qkv is None:
                query = self.to_q(x)
                key = self.to_k(x)
                value = self.to_v(x)
            else:
                query, key, value = self.to_qkv(x).chunk(3, dim=-1)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads
        query = query.view(batch_size, -1, self.heads, head_dim)
        key = key.view(batch_size, -1, self.heads, head_dim)
        value = value.view(batch_size, -1, self.heads, head_dim)

        if rope is not None and len(rope) == 4:
            with _record_function("ming.attn.qk_norm"):
                if self.q_norm is not None:
                    query = self.q_norm(query)
                if self.k_norm is not None:
                    key = self.k_norm(key)

            with _record_function("ming.attn.rope"):
                cos, sin, q_xpos_scale, k_xpos_scale = rope
                if self.pe_attn_head is not None:
                    on = self.pe_attn_head
                    query[:, :, :on, :] = _apply_rotary_pos_emb_from_trig_seq_first(
                        query[:, :, :on, :], cos, sin, q_xpos_scale
                    )
                    key[:, :, :on, :] = _apply_rotary_pos_emb_from_trig_seq_first(
                        key[:, :, :on, :], cos, sin, k_xpos_scale
                    )
                else:
                    query = _apply_rotary_pos_emb_from_trig_seq_first(query, cos, sin, q_xpos_scale)
                    key = _apply_rotary_pos_emb_from_trig_seq_first(key, cos, sin, k_xpos_scale)

            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
        else:
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)

            with _record_function("ming.attn.qk_norm"):
                if self.q_norm is not None:
                    query = self.q_norm(query)
                if self.k_norm is not None:
                    key = self.k_norm(key)

            with _record_function("ming.attn.rope"):
                if rope is not None:
                    if len(rope) == 4:
                        cos, sin, q_xpos_scale, k_xpos_scale = rope

                        def apply_rope(t: torch.Tensor, scale: torch.Tensor | float) -> torch.Tensor:
                            return _apply_rotary_pos_emb_from_trig(t, cos, sin, scale)
                    else:
                        freqs, xpos_scale = rope
                        q_xpos_scale, k_xpos_scale = (
                            (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)
                        )

                        def apply_rope(t: torch.Tensor, scale: torch.Tensor | float) -> torch.Tensor:
                            return apply_rotary_pos_emb(t, freqs, scale)

                    if self.pe_attn_head is not None:
                        on = self.pe_attn_head
                        query[:, :on, :, :] = apply_rope(query[:, :on, :, :], q_xpos_scale)
                        key[:, :on, :, :] = apply_rope(key[:, :on, :, :], k_xpos_scale)
                    else:
                        query = apply_rope(query, q_xpos_scale)
                        key = apply_rope(key, k_xpos_scale)

        if self.attn_mask_enabled and mask is not None:
            valid_sample_indices = mask.any(dim=1)
            final_output = torch.zeros_like(query).to(query.device)

            attn_mask = mask[valid_sample_indices]
            query = query[valid_sample_indices]
            key = key[valid_sample_indices]
            value = value[valid_sample_indices]
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)
            attn_mask = attn_mask.expand(valid_sample_indices.sum().item(), self.heads, query.shape[-2], key.shape[-2])
        else:
            attn_mask = None

        with _record_function("ming.attn.sdpa"):
            x = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        if self.attn_mask_enabled and mask is not None:
            final_output[valid_sample_indices] = x
            x = final_output

        x = x.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
        x = x.to(query.dtype)

        with _record_function("ming.attn.out_proj"):
            x = self.to_out[0](x)
            x = self.to_out[1](x)

        if mask is not None:
            mask = mask.unsqueeze(-1)
            x = x.masked_fill(~mask, 0.0)

        return x


class DiTBlock(nn.Module):
    """A DiT block with pre-norm and residual connections."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        qk_norm: str | None = None,
        pe_attn_head: int | None = None,
        attn_mask_enabled: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = Attention(
            dim=hidden_size,
            heads=num_heads,
            dim_head=hidden_size // num_heads,
            dropout=dropout,
            qk_norm=qk_norm,
            pe_attn_head=pe_attn_head,
            attn_mask_enabled=attn_mask_enabled,
        )
        self.norm2 = RMSNorm(hidden_size)
        self.mlp = FeedForward(dim=hidden_size, mult=mlp_ratio, dropout=dropout, approximate="tanh")

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None,
        rope: tuple[torch.Tensor, ...] | None,
    ) -> torch.Tensor:
        with _record_function("ming.dit_block.norm1"):
            normed = self.norm1(x)
        with _record_function("ming.dit_block.attn"):
            attn_out = self.attn(normed, mask=mask, rope=rope)
            x = x + attn_out if torch.is_grad_enabled() else x.add_(attn_out)
        with _record_function("ming.dit_block.norm2"):
            normed = self.norm2(x)
        with _record_function("ming.dit_block.mlp"):
            mlp_out = self.mlp(normed)
            x = x + mlp_out if torch.is_grad_enabled() else x.add_(mlp_out)
        return x


class FinalLayer(nn.Module):
    """The final layer of DiT."""

    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with _record_function("ming.final_layer.norm"):
            x = self.norm_final(x)
        with _record_function("ming.final_layer.linear"):
            x = self.linear(x)
        return x


class CondEmbedder(nn.Module):
    """Embeds LLM hidden states with optional CFG dropout."""

    def __init__(self, input_feature_size: int, hidden_size: int):
        super().__init__()
        self.cond_embedder = nn.Linear(input_feature_size, hidden_size)

    def forward(self, llm_cond: torch.Tensor) -> torch.Tensor:
        return self.cond_embedder(llm_cond)


class DiT(nn.Module):
    """Diffusion model with a Transformer backbone for audio latent generation."""

    def __init__(
        self,
        in_channels: int = 64,
        hidden_size: int = 1024,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        llm_cond_dim: int = 896,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_heads = num_heads

        self.t_embedder = DiTTimestepEmbedding(hidden_size)
        self.x_embedder = nn.Linear(in_channels, hidden_size)
        self.c_embedder = CondEmbedder(llm_cond_dim, hidden_size)
        if "spk_dim" in kwargs:
            self.spk_embedder = nn.Linear(kwargs["spk_dim"], hidden_size)
        else:
            self.spk_embedder = None
        self.hidden_size = hidden_size
        self.use_precomputed_rope_trig = True

        self.rotary_embed = RotaryEmbedding(hidden_size // num_heads)

        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, **kwargs) for _ in range(depth)]
        )
        self.final_layer = FinalLayer(hidden_size, self.out_channels)

    def _maybe_precompute_rope_trig(self, rope: tuple[torch.Tensor, torch.Tensor | None]) -> tuple[torch.Tensor, ...]:
        if not self.use_precomputed_rope_trig:
            return rope
        freqs, xpos_scale = rope
        q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)
        return freqs.cos(), freqs.sin(), q_xpos_scale, k_xpos_scale

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        latent_history: torch.Tensor,
        spk_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = torch.cat([latent_history, x], dim=1)
        x = self.x_embedder(x)
        t = self.t_embedder(t).unsqueeze(1)
        c = self.c_embedder(c)
        y = t + c
        if spk_emb is None:
            assert self.spk_embedder is None
            x = torch.cat([y, x], dim=1)
        else:
            x = torch.cat([self.spk_embedder(spk_emb), y, x], dim=1)
        rope = self._maybe_precompute_rope_trig(self.rotary_embed.forward_from_seq_len(x.shape[1]))

        for block in self.blocks:
            x = block(x, None, rope)
        x = self.final_layer(x)
        return x

    def forward_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        latent_history: torch.Tensor,
        spk_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward with classifier-free guidance (doubles batch for CFG)."""
        x = torch.cat([x, x], dim=0)
        latent_history = torch.cat([latent_history, latent_history], dim=0)
        fake_latent = torch.zeros_like(c)
        c = torch.cat([c, fake_latent], dim=0)
        if t.ndim == 0:
            t = t.repeat(x.shape[0])
        if spk_emb is not None:
            spk_emb = torch.cat([spk_emb, spk_emb], dim=0)
        model_out = self.forward(x, t, c, latent_history, spk_emb)
        return model_out[:, -x.shape[1] :, :]

    def forward_with_prepared_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        latent_history: torch.Tensor,
        spk_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward with CFG when step-invariant CFG inputs are pre-expanded."""
        x = torch.cat([x, x], dim=0)
        if t.ndim == 0:
            t = t.repeat(x.shape[0])
        model_out = self.forward(x, t, c, latent_history, spk_emb)
        return model_out[:, -x.shape[1] :, :]

    def forward_with_preembedded_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c_emb: torch.Tensor,
        latent_history_emb: torch.Tensor,
        rope: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Forward with CFG and step-invariant embeddings precomputed."""
        x = torch.cat([x, x], dim=0)
        patch_len = x.shape[1]
        x = self.x_embedder(x)
        if t.ndim == 0:
            t = t.repeat(x.shape[0])
        t = self.t_embedder(t).unsqueeze(1)
        x = torch.cat([t + c_emb, latent_history_emb, x], dim=1)

        for block in self.blocks:
            x = block(x, None, rope)
        x = self.final_layer(x[:, -patch_len:, :])
        return x

    def forward_with_preembedded_cfg_temb(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        c_emb: torch.Tensor,
        latent_history_emb: torch.Tensor,
        rope: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Forward with CFG and precomputed timestep/condition embeddings."""
        x = self.x_embedder(x)
        patch_len = x.shape[1]
        x = torch.cat([x, x], dim=0)
        x = torch.cat([t_emb + c_emb, latent_history_emb, x], dim=1)

        for block in self.blocks:
            x = block(x, None, rope)
        x = self.final_layer(x[:, -patch_len:, :])
        return x


#########################################################################################
# CFM
# Ported from:
# https://github.com/inclusionAI/Ming/blob/e58533db227031990c5a6864dcf5f08fb53ed0d2/talker_module/cfm.py
#########################################################################################


def get_epss_timesteps(n, device, dtype):
    dt = 1 / 32
    predefined_timesteps = {
        5: [0, 2, 4, 8, 16, 32],
        6: [0, 2, 4, 6, 8, 16, 32],
        7: [0, 2, 4, 6, 8, 16, 24, 32],
        10: [0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32],
        12: [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32],
        16: [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32],
    }
    t = predefined_timesteps.get(n, [])
    if not t:
        return torch.linspace(0, 1, n + 1, device=device, dtype=dtype)
    return dt * torch.tensor(t, device=device, dtype=dtype)


class CFM(nn.Module):
    """Conditional Flow Matching module for audio latent generation."""

    def __init__(self, model: nn.Module, steps: int = 10, sway_sampling_coef: float | None = -1.0):
        """
        Args:
            model: DiT used for the velocity prediction.
            steps: number of integration steps per sample call.
            sway_sampling_coef: coefficient used to skew the integration
                grid towards low-noise timesteps. Defaults to -1.0 which
                packs more steps near t=0, where prediction error is highest.
                Set to `None` to use the linear grid as-is.
        """
        super().__init__()
        self.model = model
        self.steps = steps
        self.sway_sampling_coef = sway_sampling_coef
        self.use_prepared_cfg = False
        self.use_preembedded_cfg = True
        self.use_precomputed_temb = True

    def prepare_timesteps(self, t: torch.Tensor) -> torch.Tensor:
        if self.sway_sampling_coef is None:
            return t
        return t + self.sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

    @torch.no_grad()
    def sample(
        self,
        llm_cond: torch.Tensor,
        lat_cond: torch.Tensor,
        y0: torch.Tensor,
        t: torch.Tensor,
        sde_args: torch.Tensor,
        sde_rnd: torch.Tensor | None,
        *,
        timesteps_are_swayed: bool = False,
    ):
        """Sample audio latent via ODE/SDE integration with CFG.

        Args:
            llm_cond: LLM hidden state (B, 1, hidden_size)
            lat_cond: latent history (B, his_patch_size, latent_dim)
            y0: initial noise (B, patch_size, latent_dim)
            t: timesteps from get_epss_timesteps
            sde_args: [cfg_strength, sigma, temperature]
            sde_rnd: random noise for SDE steps (steps, B, patch_size, latent_dim)
        """

        if not timesteps_are_swayed:
            t = self.prepare_timesteps(t)

        if self.use_preembedded_cfg:
            lat_cond_emb = self.model.x_embedder(lat_cond)
            cfg_lat_cond_emb = torch.cat([lat_cond_emb, lat_cond_emb], dim=0)
            llm_cond_emb = self.model.c_embedder(llm_cond)
            null_cond_bias = self.model.c_embedder.cond_embedder.bias
            if null_cond_bias is None:
                null_llm_cond_emb = self.model.c_embedder(torch.zeros_like(llm_cond))
            else:
                null_llm_cond_emb = null_cond_bias.view(1, 1, -1).expand_as(llm_cond_emb)
            cfg_llm_cond_emb = torch.cat([llm_cond_emb, null_llm_cond_emb], dim=0)
            rope = self.model._maybe_precompute_rope_trig(
                self.model.rotary_embed.forward_from_seq_len(1 + cfg_lat_cond_emb.shape[1] + y0.shape[1])
            )
            if self.use_precomputed_temb:
                t_emb = self.model.t_embedder(t[:-1]).unsqueeze(1)

            def fn(fn_t, x):
                if self.use_precomputed_temb:
                    step_t_emb = t_emb[step : step + 1]
                    return self._guided_prediction_from_cfg(
                        self.model.forward_with_preembedded_cfg_temb(
                            x,
                            step_t_emb,
                            cfg_llm_cond_emb,
                            cfg_lat_cond_emb,
                            rope,
                        ),
                        sde_args[0],
                    )
                pred_cfg = self.model.forward_with_preembedded_cfg(
                    x,
                    fn_t,
                    cfg_llm_cond_emb,
                    cfg_lat_cond_emb,
                    rope,
                )
                return self._guided_prediction_from_cfg(pred_cfg, sde_args[0])

        elif self.use_prepared_cfg:
            cfg_lat_cond = torch.cat([lat_cond, lat_cond], dim=0)
            cfg_llm_cond = torch.cat([llm_cond, torch.zeros_like(llm_cond)], dim=0)

            def fn(fn_t, x):
                pred_cfg = self.model.forward_with_prepared_cfg(x, fn_t, cfg_llm_cond, cfg_lat_cond, None)
                pred, null_pred = torch.chunk(pred_cfg, 2, dim=0)
                return pred + (pred - null_pred) * sde_args[0]

        else:

            def fn(fn_t, x):
                pred_cfg = self.model.forward_with_cfg(x, fn_t, llm_cond, lat_cond, None)
                pred, null_pred = torch.chunk(pred_cfg, 2, dim=0)
                return pred + (pred - null_pred) * sde_args[0]

        for step in range(self.steps):
            dt = t[step + 1] - t[step]
            y0 = y0 + fn(t[step], y0) * dt
            if sde_rnd is not None:
                y0 = y0 + sde_args[1] * (sde_args[2] ** 0.5) * (dt.abs() ** 0.5) * sde_rnd[step]

        return y0

    @staticmethod
    def _guided_prediction_from_cfg(pred_cfg: torch.Tensor, cfg_strength: torch.Tensor) -> torch.Tensor:
        pred, null_pred = torch.chunk(pred_cfg, 2, dim=0)
        return pred + (pred - null_pred) * cfg_strength


class CFMGraphExecutor:
    """CUDA graph-accelerated executor for CFM + Aggregator + StopHead pipeline."""

    def __init__(
        self,
        config,
        cfm,
        aggregator,
        stop_head: nn.Linear,
        *,
        deterministic_sde_noise: bool = False,
        return_stop_logits: bool = False,
    ):
        self.config = config
        self.cfm = cfm
        self.aggregator = aggregator
        self.stop_head = stop_head
        self.deterministic_sde_noise = deterministic_sde_noise
        self.return_stop_logits = return_stop_logits
        self.initialized = False

        self.last_hidden_state_placeholder = None
        self.his_lat_placeholder = None
        self.randn_like_placeholder = None
        self.t_placeholder = None
        self.sde_args_placeholder = None
        self.sde_rnd_placeholder = None
        self.gen_lat_placeholder = None
        self.inputs_embeds_placeholder = None
        self.stop_out_placeholder = None
        self.graph = None

    def execute(
        self,
        input_tensor: torch.Tensor,
        his_lat: torch.Tensor,
        cfg_strength: float = 2.0,
        sigma: float = 0.25,
        temperature: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.deterministic_sde_noise and temperature != 0.0:
            raise ValueError("deterministic CFM graph executor requires temperature=0.0")
        bat_size, his_patch_size, z_dim = his_lat.shape
        randn_tensor = torch.randn(
            (bat_size, self.config.patch_size, z_dim), device=input_tensor.device, dtype=input_tensor.dtype
        )
        sde_shape = (self.config.steps, *randn_tensor.shape)
        sde_rnd = None
        if self.deterministic_sde_noise:
            # Preserve the RNG stream used by the original temperature=0 path.
            # The sampled noise is mathematically multiplied by zero, so it is
            # consumed for determinism but intentionally kept out of the graph.
            _unused_sde_rnd = torch.randn(sde_shape, device=input_tensor.device, dtype=input_tensor.dtype)
            del _unused_sde_rnd
        else:
            sde_rnd = torch.randn(sde_shape, device=input_tensor.device, dtype=input_tensor.dtype)

        if not self.initialized:
            self._initialize_graph(input_tensor, his_lat, randn_tensor, sde_rnd)

        self.last_hidden_state_placeholder.copy_(input_tensor)
        self.his_lat_placeholder.copy_(his_lat)
        self.randn_like_placeholder.copy_(randn_tensor)
        self.sde_args_placeholder[0] = cfg_strength
        self.sde_args_placeholder[1] = sigma
        self.sde_args_placeholder[2] = temperature
        if self.sde_rnd_placeholder is not None:
            assert sde_rnd is not None
            self.sde_rnd_placeholder.copy_(sde_rnd)

        self.graph.replay()

        gen_lat = torch.empty_like(self.gen_lat_placeholder)
        gen_lat.copy_(self.gen_lat_placeholder)

        inputs_embeds = torch.empty_like(self.inputs_embeds_placeholder)
        inputs_embeds.copy_(self.inputs_embeds_placeholder)

        stop_out = torch.empty_like(self.stop_out_placeholder)
        stop_out.copy_(self.stop_out_placeholder)

        return gen_lat, inputs_embeds, stop_out

    def _initialize_graph(
        self,
        input_tensor: torch.Tensor,
        his_lat: torch.Tensor,
        randn_tensor: torch.Tensor,
        sde_rnd: torch.Tensor | None,
    ) -> None:
        self.last_hidden_state_placeholder = torch.empty_like(input_tensor)
        self.his_lat_placeholder = torch.empty_like(his_lat)
        self.randn_like_placeholder = torch.empty_like(randn_tensor)
        self.t_placeholder = self.cfm.prepare_timesteps(
            get_epss_timesteps(self.config.steps, device=input_tensor.device, dtype=input_tensor.dtype)
        )
        self.sde_args_placeholder = torch.empty(3, device=input_tensor.device, dtype=input_tensor.dtype)
        self.sde_rnd_placeholder = torch.empty_like(sde_rnd) if sde_rnd is not None else None

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, pool=current_platform.get_global_graph_pool()):
            self.gen_lat_placeholder = self.cfm.sample(
                self.last_hidden_state_placeholder,
                self.his_lat_placeholder,
                self.randn_like_placeholder,
                self.t_placeholder,
                self.sde_args_placeholder,
                self.sde_rnd_placeholder,
                timesteps_are_swayed=True,
            )
            self.inputs_embeds_placeholder = self.aggregator(self.gen_lat_placeholder)
            self.stop_out_placeholder = self.stop_head(self.last_hidden_state_placeholder[:, -1, :])
            if not self.return_stop_logits:
                self.stop_out_placeholder = self.stop_out_placeholder.softmax(dim=-1)

        self.initialized = True


class CFMGraphExecutorPool:
    """Thread-safe pool of CFMGraphExecutors for concurrent inference."""

    def __init__(
        self,
        config,
        cfm,
        aggregator,
        stop_head: nn.Linear,
        pool_size: int = 1,
        *,
        deterministic_sde_noise: bool = False,
        return_stop_logits: bool = False,
    ):
        self.config = config
        self.cfm = cfm
        self.aggregator = aggregator
        self.stop_head = stop_head
        self.pool_size = pool_size
        self.deterministic_sde_noise = deterministic_sde_noise
        self.return_stop_logits = return_stop_logits
        self.pool = Queue(maxsize=pool_size)
        self.lock = Lock()

        for _ in range(pool_size):
            executor = CFMGraphExecutor(
                config,
                cfm,
                aggregator,
                stop_head,
                deterministic_sde_noise=deterministic_sde_noise,
                return_stop_logits=return_stop_logits,
            )
            self.pool.put(executor)

    def acquire(self) -> CFMGraphExecutor:
        return self.pool.get()

    def release(self, executor: CFMGraphExecutor) -> None:
        self.pool.put(executor)

    def execute(
        self,
        input_tensor: torch.Tensor,
        his_lat: torch.Tensor,
        cfg_strength: float = 2.0,
        sigma: float = 0.25,
        temperature: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        executor = self.acquire()
        try:
            return executor.execute(
                input_tensor, his_lat, cfg_strength=cfg_strength, sigma=sigma, temperature=temperature
            )
        finally:
            self.release(executor)


class VAEStreamDecodeGraphExecutor:
    """CUDA graph executor for one stateless AudioVAE stream decode shape."""

    def __init__(self, audio_vae: AudioVAE) -> None:
        self.audio_vae = audio_vae
        self.initialized = False
        self.latent_placeholder: torch.Tensor | None = None
        self.waveform_placeholder: torch.Tensor | None = None
        self.graph: torch.cuda.CUDAGraph | None = None

    def execute(self, latent: torch.Tensor) -> torch.Tensor:
        if not self.initialized:
            self._initialize_graph(latent)

        assert self.latent_placeholder is not None
        assert self.waveform_placeholder is not None
        assert self.graph is not None
        self.latent_placeholder.copy_(latent)
        self.graph.replay()
        waveform = torch.empty_like(self.waveform_placeholder)
        waveform.copy_(self.waveform_placeholder)
        return waveform

    def _initialize_graph(self, latent: torch.Tensor) -> None:
        device = latent.device
        self.latent_placeholder = torch.empty_like(latent)
        self.graph = torch.cuda.CUDAGraph()
        self.latent_placeholder.copy_(latent)

        graph_stream = torch.cuda.Stream(device=device)
        graph_stream.wait_stream(torch.cuda.current_stream(device))
        with torch.no_grad(), torch.cuda.stream(graph_stream):
            for _ in range(3):
                self.audio_vae.decode(
                    self.latent_placeholder,
                    use_cache=False,
                    stream_state=(None, None, None),
                    last_chunk=True,
                )
        torch.cuda.current_stream(device).wait_stream(graph_stream)
        torch.accelerator.synchronize(device)

        with torch.no_grad(), torch.cuda.graph(self.graph):
            self.waveform_placeholder, _, _ = self.audio_vae.decode(
                self.latent_placeholder,
                use_cache=False,
                stream_state=(None, None, None),
                last_chunk=True,
            )
        torch.accelerator.synchronize(device)
        self.initialized = True


########################################################################
# Audio Postprocess
# Adapted from:
# https://github.com/inclusionAI/Ming/blob/e58533db227031990c5a6864dcf5f08fb53ed0d2/modeling_bailing_talker.py
########################################################################


@torch.no_grad()
def resample(waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """Resample a waveform via linear interpolation (no torchaudio dep).

    Args:
        waveform: Tensor shaped ``(..., num_samples)``.
        orig_sr: Source sample rate (Hz); must be > 0.
        target_sr: Target sample rate (Hz); must be > 0.

    Raises:
        ValueError: If sample rates are non-positive, the waveform is empty,
            or the resampled length would round to zero.
    """
    if orig_sr <= 0:
        raise ValueError(f"orig_sr must be positive, got {orig_sr}")
    if target_sr <= 0:
        raise ValueError(f"target_sr must be positive, got {target_sr}")
    if waveform.numel() == 0 or waveform.shape[-1] == 0:
        raise ValueError("waveform must contain at least one sample")
    if orig_sr == target_sr:
        return waveform

    ratio = target_sr / orig_sr
    new_len = int(waveform.shape[-1] * ratio)
    if new_len <= 0:
        raise ValueError(
            f"resampled waveform would be empty for input length {waveform.shape[-1]}, "
            f"orig_sr={orig_sr}, target_sr={target_sr}"
        )
    return torch.nn.functional.interpolate(
        waveform.unsqueeze(0),
        size=new_len,
        mode="linear",
        align_corners=False,
    ).squeeze(0)


def trim_trailing_silence(
    waveform: torch.Tensor,
    sample_rate: int,
    sil_th: float = 1e-3,
    tail_silence_s: float = 0.3,
) -> torch.Tensor:
    """Trim low-energy tail while keeping a short trailing silence.

    Works on 2-D ``(channels, samples)`` or 3-D ``(batch, channels, samples)``
    tensors. Any other shape is returned unchanged.
    """
    if waveform.numel() == 0:
        return waveform

    original_dim = waveform.dim()
    if original_dim == 3:
        speech = waveform[:, 0, :]
    elif original_dim == 2:
        speech = waveform
    else:
        return waveform

    frame_step = int(sample_rate * 0.1)
    frame_size = int(sample_rate * 0.1)
    if speech.shape[-1] < frame_size:
        keep = min(speech.shape[-1], int(tail_silence_s * sample_rate))
        trimmed = speech[..., :keep]
    else:
        num_frame = (speech.shape[-1] - frame_size) // frame_step + 1
        cur_len = (num_frame - 1) * frame_step + frame_size
        speech = speech[..., :cur_len]
        spe_frames = speech.unfold(-1, frame_size, frame_step)
        scores = spe_frames.abs().mean(dim=-1)
        scores = scores.mean(dim=list(range(scores.dim() - 1)))
        idx = scores.shape[0] - 1
        while idx >= 0 and scores[idx] <= sil_th:
            idx -= 1
        if idx < 0:
            keep = min(speech.shape[-1], int(tail_silence_s * sample_rate))
            trimmed = speech[..., :keep]
        else:
            non_sil_len = idx * frame_step + frame_size + int(tail_silence_s * sample_rate)
            non_sil_len = min(non_sil_len, speech.shape[-1])
            trimmed = speech[..., :non_sil_len]

    if original_dim == 3:
        return trimmed.unsqueeze(1)
    return trimmed


def silence_holder(
    speech: torch.Tensor,
    sample_rate: int,
    sil_cache: dict | None = None,
    last_chunk: bool = True,
    sil_th: float = 1e-3,
    last_sil: float = 0.3,
) -> tuple[torch.Tensor, dict]:
    """Ming-style streaming silence holder.

    Used during streaming VAE decode to defer emission of silent regions
    until a non-silent frame arrives (or the stream ends). ``sil_cache``
    is carried across chunks and updated in place.
    """
    if speech.numel() == 0:
        return speech, sil_cache or {"holder": [], "buffer": []}

    frame_step = int(sample_rate * 0.1)
    frame_size = int(sample_rate * 0.1)
    if sil_cache is None:
        sil_cache = {"holder": [], "buffer": []}

    if sil_cache["buffer"]:
        speech = torch.cat([*sil_cache["buffer"], speech], dim=-1)
        sil_cache["buffer"] = []

    if speech.shape[-1] < frame_size:
        sil_cache["buffer"].append(speech)
        if last_chunk:
            speech = torch.cat(sil_cache["holder"] + sil_cache["buffer"], dim=-1)
            return speech[..., : int(last_sil * sample_rate)], sil_cache
        return torch.zeros((*speech.shape[:-1], 0), device=speech.device, dtype=speech.dtype), sil_cache

    num_frame = (speech.shape[-1] - frame_size) // frame_step + 1
    cur_len = (num_frame - 1) * frame_step + frame_size
    if speech.shape[-1] > cur_len:
        sil_cache["buffer"].append(speech[..., cur_len:])
        speech = speech[..., :cur_len]

    spe_frames = speech.unfold(-1, frame_size, frame_step)
    scores = spe_frames.abs().mean(dim=-1)
    scores = scores.mean(dim=list(range(scores.dim() - 1)))
    idx = scores.shape[0] - 1
    while idx >= 0 and scores[idx] <= sil_th:
        idx -= 1

    if idx < 0:
        sil_cache["holder"].append(speech)
        if last_chunk:
            speech = torch.cat(sil_cache["holder"] + sil_cache["buffer"], dim=-1)
            return speech[..., : int(last_sil * sample_rate)], sil_cache
        return torch.zeros((*speech.shape[:-1], 0), device=speech.device, dtype=speech.dtype), sil_cache

    non_sil_len = idx * frame_step + frame_size
    if last_chunk:
        non_sil_len += int(last_sil * sample_rate)
    non_sil_len = min(non_sil_len, speech.shape[-1])
    speech_out = torch.cat([*sil_cache["holder"], speech[..., :non_sil_len]], dim=-1)
    sil_cache["holder"] = []
    if non_sil_len < speech.shape[-1]:
        sil_cache["holder"].append(speech[..., non_sil_len:])
    return speech_out, sil_cache


########################################################################
# Audio Postprocess
# Ported from:
# https://github.com/inclusionAI/Ming/blob/e58533db227031990c5a6864dcf5f08fb53ed0d2/talker_module/aggregator.py
########################################################################


class Aggregator(nn.Module):
    """Maps generated audio latent patches back to LLM embedding space."""

    def __init__(
        self,
        in_channels: int = 64,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        llm_input_dim: int = 896,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_heads = num_heads

        self.word_embedder = nn.Embedding(1, hidden_size)
        self.x_embedder = nn.Linear(in_channels, hidden_size)
        self.hidden_size = hidden_size

        self.rotary_embed = RotaryEmbedding(hidden_size // num_heads)
        self.use_precomputed_rope_trig = True

        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, **kwargs) for _ in range(depth)]
        )
        self.final_layer = FinalLayer(hidden_size, llm_input_dim)

    def _maybe_precompute_rope_trig(self, rope: tuple[torch.Tensor, torch.Tensor | None]) -> tuple[torch.Tensor, ...]:
        if not self.use_precomputed_rope_trig:
            return rope
        freqs, xpos_scale = rope
        q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)
        return freqs.cos(), freqs.sin(), q_xpos_scale, k_xpos_scale

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.x_embedder(x)
        cls_embed = self.word_embedder(torch.zeros((x.shape[0], 1), dtype=torch.long, device=x.device))
        x = torch.cat([cls_embed, x], dim=1)

        rope = self._maybe_precompute_rope_trig(self.rotary_embed.forward_from_seq_len(x.shape[1]))
        if mask is not None:
            mask_pad = mask.clone().detach()[:, :1]
            mask = torch.cat([mask_pad, mask], dim=-1)
        for block in self.blocks:
            x = block(x, mask, rope)
        x = self.final_layer(x[:, :1, :])
        return x


def pack_attention_qkv_projections(module: nn.Module) -> int:
    packed = 0
    for child in module.modules():
        if isinstance(child, Attention):
            child.pack_qkv()
            packed += 1
    return packed


def _qwen2_attention_forward_with_packed_qkv(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: StaticCache | None = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs,
):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    q_out = self.q_proj.out_features
    k_out = self.k_proj.out_features
    v_out = self.v_proj.out_features
    query_states, key_states, value_states = self.qkv_proj(hidden_states).split((q_out, k_out, v_out), dim=-1)
    query_states = query_states.view(hidden_shape).transpose(1, 2)
    key_states = key_states.view(hidden_shape).transpose(1, 2)
    value_states = value_states.view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = qwen2_apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation,
        eager_attention_forward,
    )
    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def pack_qwen2_attention_qkv_projections(module: nn.Module) -> int:
    packed = 0
    for child in module.modules():
        if not isinstance(child, Qwen2Attention):
            continue
        if hasattr(child, "qkv_proj"):
            continue
        if child.q_proj.bias is None or child.k_proj.bias is None or child.v_proj.bias is None:
            raise ValueError("Ming Qwen2 fused QKV packing expects q/k/v bias tensors")
        qkv = nn.Linear(
            child.config.hidden_size,
            child.q_proj.out_features + child.k_proj.out_features + child.v_proj.out_features,
            bias=True,
            device=child.q_proj.weight.device,
            dtype=child.q_proj.weight.dtype,
        )
        with torch.no_grad():
            qkv.weight.copy_(torch.cat([child.q_proj.weight, child.k_proj.weight, child.v_proj.weight], dim=0))
            qkv.bias.copy_(torch.cat([child.q_proj.bias, child.k_proj.bias, child.v_proj.bias], dim=0))
        qkv.requires_grad_(False)
        child.qkv_proj = qkv
        child.forward = MethodType(_qwen2_attention_forward_with_packed_qkv, child)
        packed += 1
    return packed


def _snapshot_static_cache(
    past_key_values: StaticCache,
) -> list[tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]]:
    snapshot = []
    for layer in getattr(past_key_values, "layers", []):
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        cumulative_length = getattr(layer, "cumulative_length", None)
        snapshot.append(
            (
                keys.detach().clone() if keys is not None else None,
                values.detach().clone() if values is not None else None,
                cumulative_length.detach().clone() if cumulative_length is not None else None,
            )
        )
    return snapshot


def _restore_static_cache(
    past_key_values: StaticCache,
    snapshot: list[tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]],
) -> None:
    for layer, (keys, values, cumulative_length) in zip(getattr(past_key_values, "layers", []), snapshot):
        if keys is not None:
            layer.keys.copy_(keys)
        if values is not None:
            layer.values.copy_(values)
        if cumulative_length is not None:
            layer.cumulative_length.copy_(cumulative_length)


class MingLLMDecodeGraphExecutor:
    def __init__(
        self,
        model: Qwen2Model,
        past_key_values: StaticCache,
        sample_inputs_embeds: torch.Tensor,
        cache_position_start: int,
    ) -> None:
        if sample_inputs_embeds.device.type != "cuda":
            raise ValueError("Ming LLM decode graph requires CUDA tensors")
        self._model = model
        self._past_key_values = past_key_values
        self._input_ph = torch.empty_like(sample_inputs_embeds)
        self._cache_pos_ph = torch.empty(
            (sample_inputs_embeds.shape[1],),
            device=sample_inputs_embeds.device,
            dtype=torch.long,
        )
        self._graph = torch.cuda.CUDAGraph()
        self._output: torch.Tensor | None = None
        self._shape = tuple(sample_inputs_embeds.shape)
        self._capture(sample_inputs_embeds, cache_position_start)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    def _set_inputs(self, inputs_embeds: torch.Tensor, cache_position_start: int) -> None:
        self._input_ph.copy_(inputs_embeds)
        self._cache_pos_ph.copy_(
            torch.arange(
                cache_position_start,
                cache_position_start + inputs_embeds.shape[1],
                device=inputs_embeds.device,
                dtype=torch.long,
            )
        )

    def _run_model(self):
        return self._model(
            past_key_values=self._past_key_values,
            inputs_embeds=self._input_ph,
            use_cache=True,
            cache_position=self._cache_pos_ph,
        )

    def _capture(self, sample_inputs_embeds: torch.Tensor, cache_position_start: int) -> None:
        device = sample_inputs_embeds.device
        cache_snapshot = _snapshot_static_cache(self._past_key_values)
        rng_state = torch.cuda.get_rng_state(device)
        self._set_inputs(sample_inputs_embeds, cache_position_start)

        graph_stream = torch.cuda.Stream(device=device)
        graph_stream.wait_stream(torch.cuda.current_stream(device))
        with torch.no_grad(), torch.cuda.stream(graph_stream):
            for _ in range(3):
                self._run_model()
        torch.cuda.current_stream(device).wait_stream(graph_stream)
        torch.accelerator.synchronize(device)
        _restore_static_cache(self._past_key_values, cache_snapshot)
        torch.cuda.set_rng_state(rng_state, device)

        with torch.no_grad(), torch.cuda.graph(self._graph):
            graph_outputs = self._run_model()
            self._output = graph_outputs.last_hidden_state[:, -1:, :]
        torch.accelerator.synchronize(device)
        _restore_static_cache(self._past_key_values, cache_snapshot)
        torch.cuda.set_rng_state(rng_state, device)

    def replay(self, inputs_embeds: torch.Tensor, cache_position_start: int) -> torch.Tensor:
        if tuple(inputs_embeds.shape) != self._shape:
            raise ValueError(f"LLM decode graph shape mismatch: {tuple(inputs_embeds.shape)} != {self._shape}")
        self._set_inputs(inputs_embeds, cache_position_start)
        self._graph.replay()
        if self._output is None:
            raise RuntimeError("Ming LLM decode graph output was not captured")
        return self._output


########################################################################
# Prompt Builder
# Adapted from:
# https://github.com/inclusionAI/Ming/blob/e58533db227031990c5a6864dcf5f08fb53ed0d2/modeling_bailing_talker.py
########################################################################

_MUSIC_TAGS = ("Genre: ", "Mood: ", "Instrument: ", "Theme: ", "Duration: ")


def _looks_like_music_prompt(text: str) -> bool:
    return all(tag in text for tag in _MUSIC_TAGS)


def build_tts_input(
    *,
    tokenizer: PreTrainedTokenizerBase,
    embed_tokens: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    text: str,
    prompt: str,
    spk_emb: list[torch.Tensor] | None = None,
    instruction: str | None = None,
    prompt_text: str | None = None,
    prompt_wav_emb: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (inputs_embeds, input_ids) for one TTS segment.

    Args:
        tokenizer: HF tokenizer
        embed_tokens: The LLM's input-embedding module
        device: Device to place the returned tensors on.
        dtype: dtype for the returned `inputs_embeds`.
        text: Text to synthesize.
        prompt: System-level instruction prompt prepended to the user turn.
        spk_emb: Optional list of speaker embeddings already projected into
            LLM hidden dim; each is injected at a `<|vision_start|>` slot.
        instruction: Optional free-form instruction
        prompt_text: Reference text for zero-shot voice cloning.
        prompt_wav_emb: Reference-wav embeddings to inject.
    """
    spk_emb_prompt: list[int] = []
    if spk_emb is not None:
        for i in range(len(spk_emb)):
            spk_emb_prompt.extend(
                tokenizer.encode(f"  speaker_{i + 1}:")
                + tokenizer.encode("<|vision_start|>")
                + tokenizer.encode("<|vision_pad|>")
                + tokenizer.encode("<|vision_end|>\n")
            )

    instruction_prompt: list[int] = []
    if instruction is not None:
        instruction_prompt = tokenizer.encode(instruction) + tokenizer.encode("<|im_end|>")

    prompt_text_token: list[int] = []
    prompt_latent_token: list[int] = []
    if prompt_wav_emb is not None and prompt_text is not None:
        prompt_text_token = tokenizer.encode(prompt_text)
        prompt_latent_token = tokenizer.encode("<audioPatch>") * prompt_wav_emb.size(1)

    prompt2 = [] if _looks_like_music_prompt(text) else tokenizer.encode(" Text input:\n")

    input_part = (
        tokenizer.encode("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n")
        + tokenizer.encode("<|im_start|>user\n")
        + tokenizer.encode(prompt)
        + spk_emb_prompt
        + prompt2
        + prompt_text_token
        + tokenizer.encode(text)
        + tokenizer.encode("<|im_end|>\n")
        + tokenizer.encode("<|im_start|>assistant\n")
        + instruction_prompt
        + tokenizer.encode("<audio>")
        + prompt_latent_token
    )

    input_ids = torch.tensor(input_part, dtype=torch.long, device=device).unsqueeze(0)
    inputs_embeds = embed_tokens(input_ids).to(device=device, dtype=dtype)

    # inject speaker embeddings
    if spk_emb is not None:
        spk_token_id = tokenizer.encode("<|vision_start|>")
        assert len(spk_token_id) == 1, "<|vision_start|> must tokenize to a single id"
        spk_indices = torch.where(input_ids[0] == spk_token_id[0])[0]
        assert len(spk_indices) > 0, "expected at least one <|vision_start|> slot"
        for i, se in enumerate(spk_emb):
            inputs_embeds[0, spk_indices[i] + 1] = se

    # inject prompt-wav embeddings after <audio>
    if prompt_wav_emb is not None and prompt_text is not None:
        audio_token_id = tokenizer.encode("<audio>")
        assert len(audio_token_id) == 1, "<audio> must tokenize to a single id"
        audio_indices = torch.where(input_ids[0] == audio_token_id[0])[0]
        assert len(audio_indices) > 0, "expected at least one <audio> slot"
        start = audio_indices[0] + 1
        inputs_embeds[0, start : start + prompt_wav_emb.size(1), :] = prompt_wav_emb[0]

    return inputs_embeds, input_ids


########################################################################
# Audio Generator
########################################################################


class MingAudioGenerator:
    """Generator driving prefill -> AR decode -> VAE decode
    for a single TTS request. The generator is stateless across requests.
    """

    def __init__(
        self,
        config,
        llm_config: Qwen2Config,
        model: Qwen2Model,
        cfm: CFM,
        aggregator: Aggregator,
        stop_head: torch.nn.Module,
        audio_vae: AudioVAE | None,
        patch_size: int,
        his_patch_size: int,
        latent_dim: int,
        cfg_strength: float,
        use_cuda_graphs: bool,
    ) -> None:
        self._config = config
        self._llm_config = llm_config
        self._model = model
        self._cfm = cfm
        self._aggregator = aggregator
        self._stop_head = stop_head
        self._audio_vae = audio_vae

        self.patch_size = patch_size
        self.his_patch_size = his_patch_size
        self.latent_dim = latent_dim
        self.cfg_strength = cfg_strength

        self._use_cuda_graphs = use_cuda_graphs

        # For FA2, let it see a full-length seq Q
        # trailing latent frames prepended on each decode call
        self._vae_decode_pad_frames = 32
        self._pack_qkv_enabled = True
        self._qkv_packed = False
        self._llm_decode_graph_enabled = True
        self._llm_decode_graph_batch_compaction_enabled = False
        self._vae_stream_graph_enabled = True
        self._stop_logit_decision_enabled = True
        self._llm_decode_graphs: dict[int, MingLLMDecodeGraphExecutor] = {}
        self._reusable_kv_caches: dict[tuple[int, str, torch.dtype], StaticCache] = {}
        self._reusable_kv_cache_lock = Lock()
        self._vae_stream_decode_graphs: dict[tuple[int, torch.device, torch.dtype], VAEStreamDecodeGraphExecutor] = {}

    def _maybe_pack_qkv_projections(self) -> None:
        if self._pack_qkv_enabled and not self._qkv_packed:
            packed_talker = pack_attention_qkv_projections(self._cfm) + pack_attention_qkv_projections(self._aggregator)
            packed_llm = pack_qwen2_attention_qkv_projections(self._model)
            logger.info(
                "Packed Ming Talker fused QKV projections for %d CFM/Aggregator and %d LLM attention modules",
                packed_talker,
                packed_llm,
            )
            self._qkv_packed = True

    @cached_property
    def _sampler_pools(self) -> dict[tuple[int, bool, bool], CFMGraphExecutorPool]:
        return {}

    def _get_sampler_pool(
        self, batch_size: int, device: torch.device, deterministic_sde_noise: bool
    ) -> CFMGraphExecutorPool | None:
        if not self._use_cuda_graphs or device.type != "cuda":
            return None
        key = (batch_size, deterministic_sde_noise, self._stop_logit_decision_enabled)
        pool = self._sampler_pools.get(key)
        if pool is None:
            pool = CFMGraphExecutorPool(
                self._config,
                self._cfm,
                self._aggregator,
                self._stop_head,
                pool_size=1,
                deterministic_sde_noise=deterministic_sde_noise,
                return_stop_logits=self._stop_logit_decision_enabled,
            )
            self._sampler_pools[key] = pool
        return pool

    def duration_capped_steps(self, text_len: int, requested_max_steps: int) -> int:
        """Apply the original Ming duration heuristic as a cap on decode steps."""
        if self._audio_vae is None:
            return requested_max_steps

        sample_rate = float(self._audio_vae.config.sample_rate)
        vae_patch_size = float(getattr(self._audio_vae.config, "patch_size", 4))
        hop_size = float(getattr(self._audio_vae.decoder, "hop_length", 320))
        seconds_per_step = (self.patch_size * vae_patch_size * hop_size) / sample_rate
        if seconds_per_step <= 0:
            return requested_max_steps

        max_duration_s = max(2.0, float(text_len) * (5818.0 / 16000.0))
        max_steps_by_duration = max(1, int(max_duration_s / seconds_per_step))
        return min(requested_max_steps, max_steps_by_duration)

    @torch.no_grad()
    def generate_latents(
        self,
        inputs_embeds: torch.Tensor,
        *,
        prompt_wav_lat: torch.Tensor | None = None,
        min_new_token: int = 10,
        max_steps: int = 1000,
        cfg: float | None = None,
        sigma: float = 0.25,
        temperature: float = 0.0,
        use_static_cache: bool = True,
    ) -> list[torch.Tensor]:
        """Autoregressive LLM + CFM sampling loop"""
        if cfg is None:
            cfg = self.cfg_strength
        device = next(self._model.parameters()).device
        dtype = next(self._model.parameters()).dtype
        self._maybe_pack_qkv_projections()

        his_lat = self._init_his_lat(prompt_wav_lat, device, dtype)
        past_key_values, max_cache_len = self._init_kv_cache(use_static_cache, device, dtype)
        prefill_len = inputs_embeds.shape[1]
        all_latents: list[torch.Tensor] = []

        for step in range(min(max_steps, max_cache_len - prefill_len)):
            last_hs = self.llm_step(
                inputs_embeds,
                step=step,
                past_key_values=past_key_values,
                use_static_cache=use_static_cache,
            )
            gen_lat, inputs_embeds, stop_out = self.cfm_sample_step(
                last_hs, his_lat, cfg=cfg, sigma=sigma, temperature=temperature
            )
            his_lat = self._update_his_lat(his_lat, gen_lat)
            all_latents.append(gen_lat)

            stop_now = bool(self._stop_decision(stop_out)[0].cpu().item())

            if logger.isEnabledFor(logging.DEBUG):
                stop_prob = self._stop_probability(stop_out)[0].cpu().item()
                if step % 50 == 0 or step < 5:
                    logger.debug(
                        "step=%d stop_prob=%.4f hs_norm=%.4f lat_norm=%.4f emb_norm=%.4f",
                        step,
                        stop_prob,
                        last_hs.float().norm().item(),
                        gen_lat.float().norm().item(),
                        inputs_embeds.float().norm().item(),
                    )

            if step > min_new_token and stop_now:
                logger.debug("Stopping at step %d", step)
                break

        return all_latents

    def cfm_sample_step(
        self,
        last_hidden_state: torch.Tensor,
        his_lat: torch.Tensor,
        *,
        cfg: float | None = None,
        sigma: float = 0.25,
        temperature: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one CFM sampling step.

        This is the CFM one-shot sampling step with CUDA-graph fast path.
        """
        if cfg is None:
            cfg = self.cfg_strength

        self._maybe_pack_qkv_projections()
        deterministic_sde_noise = temperature == 0.0
        sampler_pool = self._get_sampler_pool(his_lat.shape[0], last_hidden_state.device, deterministic_sde_noise)
        if sampler_pool is not None:
            return sampler_pool.execute(last_hidden_state, his_lat, cfg, sigma, temperature)

        bat_size, _, z_dim = his_lat.shape
        randn_tensor = torch.randn(
            (bat_size, self.patch_size, z_dim),
            device=last_hidden_state.device,
            dtype=last_hidden_state.dtype,
        )
        t = self._cfm.prepare_timesteps(
            get_epss_timesteps(self._config.steps, device=last_hidden_state.device, dtype=last_hidden_state.dtype)
        )
        sde_shape = (self._config.steps, *randn_tensor.shape)
        sde_rnd = None
        if deterministic_sde_noise:
            # Keep the same RNG progression as the original temperature=0 path
            # while avoiding zero-contribution SDE work in CFM.sample.
            _unused_sde_rnd = torch.randn(
                sde_shape,
                device=last_hidden_state.device,
                dtype=last_hidden_state.dtype,
            )
            del _unused_sde_rnd
        else:
            sde_rnd = torch.randn(
                sde_shape,
                device=last_hidden_state.device,
                dtype=last_hidden_state.dtype,
            )
        sde_args = torch.tensor(
            [cfg, sigma, temperature],
            device=last_hidden_state.device,
            dtype=last_hidden_state.dtype,
        )

        gen_lat = self._cfm.sample(
            last_hidden_state,
            his_lat,
            randn_tensor,
            t,
            sde_args,
            sde_rnd,
            timesteps_are_swayed=True,
        )
        inputs_embeds = self._aggregator(gen_lat)
        stop_out = self._stop_head(last_hidden_state[:, -1, :])
        if not self._stop_logit_decision_enabled:
            stop_out = stop_out.softmax(dim=-1)

        return gen_lat, inputs_embeds, stop_out

    @torch.no_grad()
    def generate_latents_batch(
        self,
        inputs_embeds: torch.Tensor,
        *,
        prompt_wav_lat: torch.Tensor | None = None,
        min_new_token: int = 10,
        max_steps: int = 1000,
        cfg: float | None = None,
        sigma: float = 0.25,
        temperature: float = 0.0,
        use_static_cache: bool = True,
    ) -> list[list[torch.Tensor]]:
        if cfg is None:
            cfg = self.cfg_strength
        device = next(self._model.parameters()).device
        dtype = next(self._model.parameters()).dtype
        batch_size = inputs_embeds.shape[0]
        self._maybe_pack_qkv_projections()

        if prompt_wav_lat is not None:
            raise NotImplementedError("batched prompt_wav_lat is not supported yet")

        all_latents: list[list[torch.Tensor]] = [[] for _ in range(batch_size)]
        active_indices = torch.arange(batch_size, device=device)
        current_inputs_embeds = inputs_embeds
        his_lat = torch.zeros(batch_size, self.his_patch_size, self.latent_dim, device=device, dtype=dtype)
        use_active_compaction = True
        allow_llm_decode_graph = not (
            use_active_compaction and batch_size > 1 and not self._llm_decode_graph_batch_compaction_enabled
        )

        past_key_values, max_cache_len = self._init_batched_kv_cache(
            batch_size,
            use_static_cache,
            device,
            dtype,
            allow_llm_decode_graph=allow_llm_decode_graph,
        )
        prefill_len = inputs_embeds.shape[1]

        for step in range(min(max_steps, max_cache_len - prefill_len)):
            last_hs = self.llm_step(
                current_inputs_embeds,
                step=step,
                past_key_values=past_key_values,
                use_static_cache=use_static_cache,
                allow_llm_decode_graph=allow_llm_decode_graph,
            )
            gen_lat, next_inputs_embeds, stop_out = self.cfm_sample_step(
                last_hs, his_lat, cfg=cfg, sigma=sigma, temperature=temperature
            )
            for row, original_idx in enumerate(active_indices.tolist()):
                all_latents[original_idx].append(gen_lat[row : row + 1])

            should_stop = self._stop_decision(stop_out)
            if step <= min_new_token:
                should_stop = torch.zeros_like(should_stop, dtype=torch.bool)
            if bool(torch.all(should_stop)):
                break

            if use_active_compaction and bool(torch.any(should_stop)):
                keep = ~should_stop
                active_indices = active_indices[keep]
                if active_indices.numel() == 0:
                    break
                current_inputs_embeds = next_inputs_embeds[keep]
                his_lat = self._update_his_lat(his_lat, gen_lat)[keep]
                if past_key_values is not None:
                    self._select_cache_batch(past_key_values, torch.nonzero(keep, as_tuple=False).flatten())
            else:
                current_inputs_embeds = next_inputs_embeds
                his_lat = self._update_his_lat(his_lat, gen_lat)

        return all_latents

    def _stop_decision(self, stop_out: torch.Tensor) -> torch.Tensor:
        if self._stop_logit_decision_enabled:
            return stop_out[:, 1] > stop_out[:, 0]
        return stop_out[:, 1] > 0.5

    def _stop_probability(self, stop_out: torch.Tensor) -> torch.Tensor:
        if self._stop_logit_decision_enabled:
            return torch.sigmoid((stop_out[:, 1] - stop_out[:, 0]).float())
        return stop_out[:, 1]

    @staticmethod
    def _select_cache_batch(past_key_values: StaticCache, keep_indices: torch.Tensor) -> None:
        try:
            past_key_values.batch_select_indices(keep_indices)
            return
        except AttributeError:
            pass

        for layer in getattr(past_key_values, "layers", []):
            keys = getattr(layer, "keys", None)
            values = getattr(layer, "values", None)
            if keys is not None:
                layer.keys = keys.index_select(0, keep_indices).contiguous()
            if values is not None:
                layer.values = values.index_select(0, keep_indices).contiguous()

    def _init_batched_kv_cache(
        self,
        batch_size: int,
        use_static_cache: bool,
        device: torch.device,
        dtype: torch.dtype,
        *,
        allow_llm_decode_graph: bool = True,
    ) -> tuple[StaticCache | None, int]:
        max_cache_len = 2048
        if not use_static_cache:
            return None, max_cache_len
        if allow_llm_decode_graph and self._llm_decode_graph_enabled and device.type == "cuda":
            cache = self._get_reusable_kv_cache(batch_size, device, dtype, max_cache_len)
            self._reset_static_cache(cache)
            return cache, max_cache_len
        cache = StaticCache(
            config=self._llm_config,
            max_batch_size=batch_size,
            max_cache_len=max_cache_len,
            device=device,
            dtype=dtype,
        )
        return cache, max_cache_len

    def decode_to_waveform(self, latents: list[torch.Tensor], stream_decode: bool = True) -> torch.Tensor:
        """Decode accumulated latents to waveform via AudioVAE."""
        if self._audio_vae is None:
            raise RuntimeError("AudioVAE not loaded. Cannot decode audio latents to waveform.")

        if stream_decode:
            return self._stream_decode(latents)

        all_lat = torch.cat(latents, dim=1)
        waveform, _, _ = self._audio_vae.decode(
            all_lat, use_cache=False, stream_state=(None, None, None), last_chunk=True
        )
        return waveform

    def llm_step(
        self,
        inputs_embeds: torch.Tensor,
        *,
        step: int,
        past_key_values: StaticCache | None,
        use_static_cache: bool,
        allow_llm_decode_graph: bool = True,
    ) -> torch.Tensor:
        if step == 0 or not use_static_cache:
            outputs = self._model(
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=True,
            )
        else:
            past_seen_tokens = past_key_values.get_seq_length()
            if isinstance(past_seen_tokens, torch.Tensor):
                past_seen_tokens = int(past_seen_tokens.max().item())
            if allow_llm_decode_graph:
                graph_hidden = self._llm_decode_graph_step(inputs_embeds, past_key_values, past_seen_tokens)
                if graph_hidden is not None:
                    return graph_hidden
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
            outputs = self._model(
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=True,
                cache_position=cache_position,
            )
        return outputs.last_hidden_state[:, -1:, :]

    def _llm_decode_graph_step(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: StaticCache,
        cache_position_start: int,
    ) -> torch.Tensor | None:
        if not self._llm_decode_graph_enabled or inputs_embeds.device.type != "cuda":
            return None

        key = id(past_key_values)
        graph = self._llm_decode_graphs.get(key)
        if graph is not None and graph.shape != tuple(inputs_embeds.shape):
            self._llm_decode_graphs.pop(key, None)
            return None
        if graph is None:
            logger.info(
                "Capturing Ming Talker LLM decode CUDA graph for shape=%s cache_position_start=%d",
                tuple(inputs_embeds.shape),
                cache_position_start,
            )
            graph = MingLLMDecodeGraphExecutor(
                self._model,
                past_key_values,
                inputs_embeds,
                cache_position_start,
            )
            self._llm_decode_graphs[key] = graph
        return graph.replay(inputs_embeds, cache_position_start)

    def _init_his_lat(
        self, prompt_wav_lat: torch.Tensor | None, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        his_lat = torch.zeros(1, self.his_patch_size, self.latent_dim, device=device, dtype=dtype)
        if prompt_wav_lat is not None:
            start_index = self.his_patch_size - prompt_wav_lat.size(1)
            if start_index < 0:
                his_lat[:] = prompt_wav_lat[:, -start_index:, :]
            else:
                his_lat[:, start_index:, :] = prompt_wav_lat
        return his_lat

    def _init_kv_cache(
        self, use_static_cache: bool, device: torch.device, dtype: torch.dtype
    ) -> tuple[StaticCache | None, int]:
        max_cache_len = 2048
        if not use_static_cache:
            return None, max_cache_len
        if self._llm_decode_graph_enabled and device.type == "cuda":
            cache = self._get_reusable_kv_cache(1, device, dtype, max_cache_len)
            self._reset_static_cache(cache)
            return cache, max_cache_len
        cache = StaticCache(
            config=self._llm_config,
            max_batch_size=1,
            max_cache_len=max_cache_len,
            device=device,
            dtype=dtype,
        )
        return cache, max_cache_len

    def _get_reusable_kv_cache(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        max_cache_len: int,
    ) -> StaticCache:
        key = (batch_size, str(device), dtype)
        with self._reusable_kv_cache_lock:
            cache = self._reusable_kv_caches.get(key)
            if cache is None or self._static_cache_batch_size(cache) != batch_size:
                cache = StaticCache(
                    config=self._llm_config,
                    max_batch_size=batch_size,
                    max_cache_len=max_cache_len,
                    device=device,
                    dtype=dtype,
                )
                self._reusable_kv_caches[key] = cache
            return cache

    @staticmethod
    def _static_cache_batch_size(cache: StaticCache) -> int | None:
        for layer in getattr(cache, "layers", []):
            keys = getattr(layer, "keys", None)
            if keys is not None:
                return int(keys.shape[0])
        return None

    @staticmethod
    def _reset_static_cache(cache: StaticCache) -> None:
        if hasattr(cache, "reset"):
            cache.reset()
            return
        for layer in getattr(cache, "layers", []):
            keys = getattr(layer, "keys", None)
            values = getattr(layer, "values", None)
            cumulative_length = getattr(layer, "cumulative_length", None)
            if keys is not None:
                keys.zero_()
            if values is not None:
                values.zero_()
            if cumulative_length is not None:
                cumulative_length.zero_()

    def _update_his_lat(self, his_lat: torch.Tensor, gen_lat: torch.Tensor) -> torch.Tensor:
        if self.his_patch_size == self.patch_size:
            return gen_lat
        if self.his_patch_size > self.patch_size:
            return torch.cat([his_lat[:, self.patch_size - self.his_patch_size :], gen_lat], dim=1)
        raise NotImplementedError(f"his_patch_size ({self.his_patch_size}) < patch_size ({self.patch_size})")

    # VAE streaming decode
    def _vae_stream_decode_step(self, vae_input: torch.Tensor) -> torch.Tensor:
        if not self._vae_stream_graph_enabled or vae_input.device.type != "cuda":
            speech, _, _ = self._audio_vae.decode(
                vae_input,
                use_cache=False,
                stream_state=(None, None, None),
                last_chunk=True,
            )
            return speech

        key = (vae_input.shape[1], vae_input.device, vae_input.dtype)
        executor = self._vae_stream_decode_graphs.get(key)
        if executor is None:
            executor = VAEStreamDecodeGraphExecutor(self._audio_vae)
            self._vae_stream_decode_graphs[key] = executor
        return executor.execute(vae_input)

    def _stream_decode(self, latents: list[torch.Tensor]) -> torch.Tensor:
        sr = int(self._audio_vae.config.sample_rate)
        decode_pad: torch.Tensor | None = None
        sil_cache: dict | None = None
        wav_chunks: list[torch.Tensor] = []

        for i, lat in enumerate(latents):
            last_chunk = i == (len(latents) - 1)

            if decode_pad is not None:
                vae_input = torch.cat([decode_pad, lat], dim=1)
                pad_frames = decode_pad.shape[1]
            else:
                vae_input = lat
                pad_frames = 0

            # Stateless, no KV cache accum intentionally.
            speech = self._vae_stream_decode_step(vae_input)

            total_frames = vae_input.shape[1]
            dcs = speech.shape[-1] // total_frames

            # keep only the new audio.
            speech_chunk = speech[:, :, pad_frames * dcs :][0].detach().float()
            speech_chunk, sil_cache = silence_holder(
                speech_chunk,
                sr,
                sil_cache=sil_cache,
                last_chunk=last_chunk,
            )
            if speech_chunk.numel() > 0:
                wav_chunks.append(speech_chunk)

            # Advance the sliding buffer
            decode_pad = vae_input[:, -self._vae_decode_pad_frames :, :].detach()

        if not wav_chunks:
            device = next(self._model.parameters()).device
            dtype = next(self._model.parameters()).dtype
            return torch.zeros((1, 1, 0), device=device, dtype=dtype)
        return torch.cat(wav_chunks, dim=-1).unsqueeze(0)

    # Post-decode helper
    def trim_trailing_silence(self, waveform: torch.Tensor) -> torch.Tensor:
        if self._audio_vae is None:
            return waveform
        return trim_trailing_silence(waveform, int(self._audio_vae.config.sample_rate))
