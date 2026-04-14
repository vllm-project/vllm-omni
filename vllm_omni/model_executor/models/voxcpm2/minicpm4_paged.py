# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniCPM4 with PagedAttention + fp32 RoPE/RMSNorm for VoxCPM2.

Uses vllm Attention for KV cache, keeps fp32 precision ops from
minicpm4_hf_compat.py to match native VoxCPM2 numerics.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from vllm.config import CacheConfig, VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import make_empty_intermediate_tensors_factory
from vllm.sequence import IntermediateTensors

from .minicpm4_hf_compat import (
    _apply_rotary_pos_emb,
    _MiniCPMLongRoPE,
    _MiniCPMMLP,
)

logger = init_logger(__name__)


def _resolve_lm_cfg(config: Any) -> Any:
    """Extract lm_config from VoxCPM2Config, converting dict to namespace if needed."""
    lm_cfg = getattr(config, "lm_config", config)
    if isinstance(lm_cfg, dict):

        class _Cfg:
            pass

        c = _Cfg()
        for k, v in lm_cfg.items():
            setattr(c, k, v)
        return c
    return lm_cfg


# ===================================================================
#  Attention with vllm PagedAttention backend
# ===================================================================


class _PagedMiniCPM4Attention(nn.Module):
    """PagedAttention + fp32 RoPE with separate q/k/v projections."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        kv_channels: int | None,
        layer_idx: int,
        cache_config: CacheConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = kv_channels if kv_channels else hidden_size // num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.q_proj = nn.Linear(hidden_size, self.q_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.kv_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.kv_size, bias=False)
        self.o_proj = nn.Linear(self.q_size, hidden_size, bias=False)
        self._fused_qkv_weight: torch.Tensor | None = None

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            scale=self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        rope_emb: _MiniCPMLongRoPE | None = None,
    ) -> torch.Tensor:
        """Forward: fused QKV → fp32 RoPE → PagedAttention → o_proj."""
        if self._fused_qkv_weight is None:
            self._fused_qkv_weight = torch.cat(
                [
                    self.q_proj.weight,
                    self.k_proj.weight,
                    self.v_proj.weight,
                ],
                dim=0,
            ).detach()
        qkv = nn.functional.linear(hidden_states, self._fused_qkv_weight)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        if rope_emb is not None:
            cos, sin = rope_emb(positions)
            bsz = q.shape[0]
            q_r = q.view(bsz, self.num_heads, self.head_dim)
            k_r = k.view(bsz, self.num_kv_heads, self.head_dim)
            q_r = q_r.unsqueeze(0).transpose(1, 2)  # [1, heads, n_tokens, dim]
            k_r = k_r.unsqueeze(0).transpose(1, 2)  # [1, kv_heads, n_tokens, dim]
            q_r, k_r = _apply_rotary_pos_emb(q_r, k_r, cos, sin)
            q = q_r.transpose(1, 2).squeeze(0).reshape(bsz, -1)  # [n_tokens, q_size]
            k = k_r.transpose(1, 2).squeeze(0).reshape(bsz, -1)  # [n_tokens, kv_size]

        attn_output = self.attn(q, k, v)

        output = self.o_proj(attn_output)
        return output


# ===================================================================
#  Decoder Layer
# ===================================================================


class _PagedMiniCPM4DecoderLayer(nn.Module):
    """Decoder layer: PagedAttention + fp32 RMSNorm + muP scale_depth."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        kv_channels: int | None,
        rms_norm_eps: float,
        layer_idx: int,
        num_hidden_layers: int,
        use_mup: bool,
        scale_depth: float,
        cache_config: CacheConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.self_attn = _PagedMiniCPM4Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            kv_channels=kv_channels,
            layer_idx=layer_idx,
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = _MiniCPMMLP(hidden_size, intermediate_size)
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

        self.use_mup = use_mup
        self.scale_depth = scale_depth
        self.num_hidden_layers = num_hidden_layers

    def _residual_scale(self) -> float:
        if self.use_mup:
            return self.scale_depth / math.sqrt(self.num_hidden_layers)
        return 1.0

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        rope_emb: _MiniCPMLongRoPE | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Pre-norm + attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states, rope_emb)

        scale = self._residual_scale()
        if scale != 1.0:
            hidden_states = residual + hidden_states * scale
        else:
            hidden_states = residual + hidden_states

        # Pre-norm + FFN
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        if scale != 1.0:
            hidden_states = residual + hidden_states * scale
        else:
            hidden_states = residual + hidden_states

        return hidden_states, None


# ===================================================================
#  Full Model
# ===================================================================


class MiniCPM4PagedForVoxCPM2(nn.Module):
    """PagedAttention base_lm (28 layers) for VoxCPM2 scaffold."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        self.config = config

        lm_cfg = _resolve_lm_cfg(config)

        hidden_size = lm_cfg.hidden_size
        num_hidden_layers = lm_cfg.num_hidden_layers
        kv_channels = getattr(lm_cfg, "kv_channels", None)

        self.vocab_size = lm_cfg.vocab_size
        self.embed_tokens = nn.Embedding(self.vocab_size, hidden_size)

        rope_scaling = getattr(lm_cfg, "rope_scaling", None)
        if isinstance(rope_scaling, dict):
            rope_scaling_dict = rope_scaling
        elif hasattr(rope_scaling, "__dict__"):
            rope_scaling_dict = {
                "short_factor": rope_scaling.short_factor,
                "long_factor": rope_scaling.long_factor,
                "original_max_position_embeddings": rope_scaling.original_max_position_embeddings,
            }
        else:
            rope_scaling_dict = {}

        no_rope = getattr(lm_cfg, "no_rope", False)
        if not no_rope:
            self.rope_emb = _MiniCPMLongRoPE(
                hidden_size=hidden_size,
                num_attention_heads=lm_cfg.num_attention_heads,
                kv_channels=kv_channels,
                rope_theta=getattr(lm_cfg, "rope_theta", 10000.0),
                max_position_embeddings=getattr(lm_cfg, "max_position_embeddings", 32768),
                rope_scaling=rope_scaling_dict,
            )
        else:
            self.rope_emb = None

        self.layers = nn.ModuleList(
            [
                _PagedMiniCPM4DecoderLayer(
                    hidden_size=hidden_size,
                    intermediate_size=lm_cfg.intermediate_size,
                    num_attention_heads=lm_cfg.num_attention_heads,
                    num_key_value_heads=lm_cfg.num_key_value_heads,
                    kv_channels=kv_channels,
                    rms_norm_eps=lm_cfg.rms_norm_eps,
                    layer_idx=i,
                    num_hidden_layers=num_hidden_layers,
                    use_mup=getattr(lm_cfg, "use_mup", False),
                    scale_depth=getattr(lm_cfg, "scale_depth", 1.0),
                    cache_config=cache_config,
                    prefix=f"{prefix}.layers.{i}",
                )
                for i in range(num_hidden_layers)
            ]
        )

        self.norm = RMSNorm(hidden_size, eps=lm_cfg.rms_norm_eps)

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], hidden_size
        )

        use_mup = getattr(lm_cfg, "use_mup", False)
        self._scale_emb = getattr(lm_cfg, "scale_emb", 1.0) if use_mup else 1.0
        self._compiled_layers: set[int] = set()

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        return self.embed_tokens(input_ids) * self._scale_emb

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | IntermediateTensors:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_input_ids(input_ids)

        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
                self.rope_emb,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states

    def compile_selective(self) -> list[str]:
        """Compile MLP + o_proj; keep RMSNorm/RoPE eager for precision."""
        compiled: list[str] = []
        for i, layer in enumerate(self.layers):
            if i in self._compiled_layers:
                continue
            try:
                layer.mlp = torch.compile(
                    layer.mlp,
                    mode="default",
                    fullgraph=True,
                )
                layer.self_attn.o_proj = torch.compile(
                    layer.self_attn.o_proj,
                    mode="default",
                    fullgraph=True,
                )
                layer.self_attn._fused_qkv_weight = None
                self._compiled_layers.add(i)
                if i == 0:
                    compiled.append(f"layers.*.mlp (×{len(self.layers)})")
                    compiled.append(f"layers.*.self_attn.o_proj (×{len(self.layers)})")
            except Exception as e:
                logger.warning("compile_selective: layer %d failed: %s", i, e)
                break
        return compiled

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights from native checkpoint (base_lm. prefix pre-stripped)."""
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded: set[str] = set()

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            param = params_dict.get(name)
            if param is None:
                continue
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded.add(name)

        return loaded


# ===================================================================
#  Residual LM with PagedAttention (no RoPE, 8 layers)
# ===================================================================


class MiniCPM4PagedResidualLM(nn.Module):
    """PagedAttention residual LM (8 layers, no RoPE) for VoxCPM2."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        self.config = config

        lm_cfg = _resolve_lm_cfg(config)

        hidden_size = lm_cfg.hidden_size
        num_hidden_layers = getattr(config, "residual_lm_num_layers", 8)
        kv_channels = getattr(lm_cfg, "kv_channels", None)

        self.rope_emb = None

        self.layers = nn.ModuleList(
            [
                _PagedMiniCPM4DecoderLayer(
                    hidden_size=hidden_size,
                    intermediate_size=lm_cfg.intermediate_size,
                    num_attention_heads=lm_cfg.num_attention_heads,
                    num_key_value_heads=lm_cfg.num_key_value_heads,
                    kv_channels=kv_channels,
                    rms_norm_eps=lm_cfg.rms_norm_eps,
                    layer_idx=i,
                    num_hidden_layers=num_hidden_layers,
                    use_mup=getattr(lm_cfg, "use_mup", False),
                    scale_depth=getattr(lm_cfg, "scale_depth", 1.0),
                    cache_config=cache_config,
                    prefix=f"{prefix}.layers.{i}",
                )
                for i in range(num_hidden_layers)
            ]
        )

        self.norm = RMSNorm(hidden_size, eps=lm_cfg.rms_norm_eps)
        self._compiled_layers: set[int] = set()

    def forward(
        self,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = inputs_embeds
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
                self.rope_emb,
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states

    def compile_selective(self) -> list[str]:
        """Compile MLP + o_proj (same as base_lm)."""
        compiled: list[str] = []
        for i, layer in enumerate(self.layers):
            if i in self._compiled_layers:
                continue
            try:
                layer.mlp = torch.compile(layer.mlp, mode="default", fullgraph=True)
                layer.self_attn.o_proj = torch.compile(layer.self_attn.o_proj, mode="default", fullgraph=True)
                layer.self_attn._fused_qkv_weight = None
                self._compiled_layers.add(i)
                if i == 0:
                    compiled.append(f"layers.*.mlp (×{len(self.layers)})")
                    compiled.append(f"layers.*.self_attn.o_proj (×{len(self.layers)})")
            except Exception as e:
                logger.warning("compile_selective: residual layer %d failed: %s", i, e)
        return compiled

    def load_weights_from_native(self, native_residual_lm: nn.Module) -> int:
        """Load weights from native residual_lm. Returns param count."""
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded = 0
        for name, param in native_residual_lm.named_parameters():
            if "rotary_emb" in name:
                continue
            target = params_dict.get(name)
            if target is None:
                continue
            weight_loader = getattr(target, "weight_loader", default_weight_loader)
            weight_loader(target, param.data)
            loaded += 1
        return loaded
