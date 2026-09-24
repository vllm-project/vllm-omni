# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cosmos3 Edge transformer variant with a Nemotron dense UND backbone."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

from vllm_omni.diffusion.attention.layer import Attention as FrameworkAttention

from .transformer_cosmos3 import (
    Cosmos3VFMTransformer,
    Qwen3VLTextRotaryEmbedding,
    RMSNorm,
    _apply_rotary_pos_emb,
    _nested_get,
    _tf_config_get,
)

COSMOS3_EDGE_BACKBONE_TYPE = "cosmos3_edge_nemotron_dense"


def _tf_config_nested_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        value = _nested_get(config, key)
    elif hasattr(config, "to_dict"):
        value = _nested_get(config.to_dict(), key)
    elif hasattr(config, "params"):
        value = _nested_get(config.params, key)
    else:
        value = getattr(config, key, None)
    return default if value is None else value


class Cosmos3Relu2MLP(nn.Module):
    """Nemotron dense MLP: down_proj(relu(up_proj(x)) ** 2)."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.up_proj = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=False,
            gather_output=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.up_proj(x))
        return self.down_proj(hidden * hidden)


class Cosmos3EdgeCausalAttention(nn.Module):
    """Edge UND causal attention.

    Reasoner self-attention uses raw UND Q/K with RoPE because
    ``qk_norm_for_text=false``.  The GEN-facing UND key cache is a separate
    normalized-then-RoPE'd view of the raw K tensor when GEN QK norm is enabled;
    no-GEN-QK-norm checkpoints reuse the raw RoPE'd UND K.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rms_norm_eps: float,
        use_und_k_norm_for_gen: bool,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = head_dim

        tp_size = get_tensor_model_parallel_world_size()
        self.num_heads_local = self.num_heads // tp_size
        self.num_kv_heads_local = self.num_kv_heads // tp_size

        self.to_q = ColumnParallelLinear(
            hidden_size,
            self.num_heads * self.head_dim,
            bias=False,
            gather_output=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.to_q",
        )
        self.to_k = ColumnParallelLinear(
            hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.to_k",
        )
        self.to_v = ColumnParallelLinear(
            hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.to_v",
        )
        self.to_out = RowParallelLinear(
            self.num_heads * self.head_dim,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.to_out",
        )

        self.k_norm_und_for_gen = RMSNorm(self.head_dim, eps=rms_norm_eps) if use_und_k_norm_for_gen else None
        self.attn = FrameworkAttention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            causal=True,
            softmax_scale=1.0 / (self.head_dim**0.5),
            num_kv_heads=self.num_kv_heads,
            skip_sequence_parallel=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, S, _ = hidden_states.shape

        q = self.to_q(hidden_states).view(B, S, self.num_heads_local, self.head_dim)
        k = self.to_k(hidden_states).view(B, S, self.num_kv_heads_local, self.head_dim)
        v = self.to_v(hidden_states).view(B, S, self.num_kv_heads_local, self.head_dim)

        q_reasoner, k_reasoner = _apply_rotary_pos_emb(q, k, freqs_cos, freqs_sin)

        if self.k_norm_und_for_gen is not None:
            k_gen = F.rms_norm(
                k,
                (self.head_dim,),
                self.k_norm_und_for_gen.weight,
                eps=self.k_norm_und_for_gen.variance_epsilon,
            )
            _, k_gen = _apply_rotary_pos_emb(q, k_gen, freqs_cos, freqs_sin)
        else:
            k_gen = k_reasoner

        out = self.attn(q_reasoner, k_reasoner, v).reshape(B, S, -1)
        return self.to_out(out), k_gen, v


class Cosmos3EdgeUndDecoderLayer(nn.Module):
    """Edge UND decoder layer: causal self-attention + ReLU2 MLP."""

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rms_norm_eps: float,
        use_und_k_norm_for_gen: bool,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.self_attn = Cosmos3EdgeCausalAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            rms_norm_eps=rms_norm_eps,
            use_und_k_norm_for_gen=use_und_k_norm_for_gen,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = Cosmos3Relu2MLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        cos, sin = freqs
        attn_out, k, v = self.self_attn(hidden_states, cos, sin)
        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)

        return hidden_states, k, v


class Cosmos3EdgeLanguageModel(nn.Module):
    """Nemotron dense UND tower that returns GEN-facing K plus raw V per layer."""

    _layerwise_offload_blocks_attrs = ["layers"]

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        vocab_size: int,
        rms_norm_eps: float,
        rope_theta: float,
        mrope_section: list[int],
        use_und_k_norm_for_gen: bool,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(
            head_dim=head_dim,
            rope_theta=rope_theta,
            mrope_section=mrope_section,
        )
        self.layers = nn.ModuleList(
            [
                Cosmos3EdgeUndDecoderLayer(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_attention_heads=num_attention_heads,
                    num_key_value_heads=num_key_value_heads,
                    head_dim=head_dim,
                    rms_norm_eps=rms_norm_eps,
                    use_und_k_norm_for_gen=use_und_k_norm_for_gen,
                    quant_config=quant_config,
                    prefix=f"{prefix}.layers.{i}",
                )
                for i in range(num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        text_ids: torch.Tensor,
        freqs: tuple[torch.Tensor, torch.Tensor],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        hidden = self.embed_tokens(text_ids)

        cached_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer in self.layers:
            hidden, k, v = layer(hidden, freqs)
            cached_kv.append((k, v))

        return cached_kv


class Cosmos3EdgeVFMTransformer(Cosmos3VFMTransformer):
    """Cosmos3 Edge variant with Nemotron dense UND and shared GEN diffusion."""

    _language_model_cls = Cosmos3EdgeLanguageModel
    _gen_mlp_cls = Cosmos3Relu2MLP

    @staticmethod
    def _validate_supported_config(model_config: Any) -> None:
        expected_values = {
            "qk_norm_for_text": False,
            "position_embedding_type": "unified_3d_mrope",
            "unified_3d_mrope_reset_spatial_ids": True,
            "joint_attn_implementation": "two_way",
        }
        for key, expected in expected_values.items():
            actual = _tf_config_get(model_config, key, expected)
            if actual != expected:
                raise ValueError(
                    f"Unsupported Cosmos3 Edge transformer config: {key}={actual!r}; expected {expected!r}."
                )

        backbone_type = _tf_config_get(model_config, "backbone_type", None)
        if backbone_type is None:
            raise ValueError(
                f"Cosmos3 Edge transformer config must declare backbone_type={COSMOS3_EDGE_BACKBONE_TYPE!r}."
            )
        if backbone_type != COSMOS3_EDGE_BACKBONE_TYPE:
            raise ValueError(
                "Unsupported Cosmos3 Edge transformer config: "
                f"backbone_type={backbone_type!r}; expected {COSMOS3_EDGE_BACKBONE_TYPE!r}."
            )

        required_values = {
            "latent_channel": 48,
            "latent_patch_size": 2,
            "temporal_compression_factor": 4,
        }
        for key, expected in required_values.items():
            actual = _tf_config_get(model_config, key, None)
            if actual is None:
                raise ValueError(f"Cosmos3 Edge transformer config must declare {key}={expected}.")
            if int(actual) != expected:
                raise ValueError(
                    f"Unsupported Cosmos3 Edge transformer config: {key}={actual!r}; expected {expected!r}."
                )

    @classmethod
    def _resolve_rms_norm_eps(cls, model_config: Any) -> float:
        return float(
            _tf_config_get(
                model_config,
                "rms_norm_eps",
                _tf_config_get(model_config, "layer_norm_epsilon", 1e-5),
            )
        )

    @classmethod
    def _resolve_rope_theta(cls, model_config: Any) -> float:
        return float(_tf_config_get(model_config, "rope_theta", 100_000_000))

    @classmethod
    def _resolve_mrope_section(cls, model_config: Any) -> list[int]:
        mrope_section = _tf_config_get(model_config, "mrope_section", None)
        if mrope_section is None:
            mrope_section = _tf_config_nested_get(model_config, "mrope_section", [24, 20, 20])
        return list(mrope_section)

    def _language_model_kwargs(self) -> dict[str, Any]:
        return {"use_und_k_norm_for_gen": self.use_und_k_norm_for_gen}

    def validate_loaded_weights(self, loaded: set[str]) -> None:
        missing: list[str] = []
        for layer_idx in range(self.num_hidden_layers):
            required_markers = (
                f"language_model.layers.{layer_idx}.mlp.up_proj.",
                f"language_model.layers.{layer_idx}.mlp.down_proj.",
                f"gen_layers.{layer_idx}.mlp.up_proj.",
                f"gen_layers.{layer_idx}.mlp.down_proj.",
            )
            if self.use_und_k_norm_for_gen:
                required_markers = (
                    *required_markers,
                    f"language_model.layers.{layer_idx}.self_attn.k_norm_und_for_gen.",
                )
            missing.extend(
                marker.rstrip(".") for marker in required_markers if not any(marker in name for name in loaded)
            )
        if missing:
            preview = ", ".join(missing[:10])
            suffix = " ..." if len(missing) > 10 else ""
            raise ValueError(
                "Cosmos3 Edge transformer checkpoint is missing required weights for "
                f"{preview}{suffix}. Use a converted Edge checkpoint with ReLU2 UND/GEN MLP weights."
            )
