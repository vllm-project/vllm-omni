# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention

ANIMA_TEXT_CONDITIONER_CONFIG = {
    "source_dim": 1024,
    "target_dim": 1024,
    "model_dim": 1024,
    "num_layers": 6,
    "num_attention_heads": 16,
    "mlp_ratio": 4.0,
    "target_vocab_size": 32128,
    "use_self_attention": True,
    "use_layer_norm": False,
    "min_sequence_length": 512,
}


def _rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    hidden_states_1 = hidden_states[..., : hidden_states.shape[-1] // 2]
    hidden_states_2 = hidden_states[..., hidden_states.shape[-1] // 2 :]
    return torch.cat((-hidden_states_2, hidden_states_1), dim=-1)


def _apply_rotary_pos_emb(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> torch.Tensor:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (hidden_states * cos) + (_rotate_half(hidden_states) * sin)


class AnimaRotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, rope_theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).to(dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        inv_freq_expanded = inv_freq_expanded.to(hidden_states.device)
        position_ids_expanded = position_ids[:, None, :].float()

        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype=hidden_states.dtype), sin.to(dtype=hidden_states.dtype)


class AnimaTextConditionerAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        prefix: str = "",
    ) -> None:
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim

        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.q_proj = nn.Linear(query_dim, inner_dim, bias=False)
        self.q_norm = nn.RMSNorm(attention_head_dim, eps=1e-6)
        self.k_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.k_norm = nn.RMSNorm(attention_head_dim, eps=1e-6)
        self.v_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.o_proj = nn.Linear(inner_dim, query_dim, bias=False)
        self.attn = Attention(
            num_heads=num_attention_heads,
            head_size=attention_head_dim,
            softmax_scale=1.0 / (attention_head_dim**0.5),
            causal=False,
            num_kv_heads=num_attention_heads,
            prefix=prefix,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        encoder_position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        encoder_hidden_states = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        input_shape = hidden_states.shape[:-1]
        encoder_input_shape = encoder_hidden_states.shape[:-1]

        query = self.q_proj(hidden_states)
        key = self.k_proj(encoder_hidden_states)
        value = self.v_proj(encoder_hidden_states)

        query = query.view(*input_shape, self.num_attention_heads, self.attention_head_dim)
        key = key.view(*encoder_input_shape, self.num_attention_heads, self.attention_head_dim)
        value = value.view(*encoder_input_shape, self.num_attention_heads, self.attention_head_dim)

        query = self.q_norm(query)
        key = self.k_norm(key)

        if position_embeddings is not None:
            if encoder_position_embeddings is None:
                raise ValueError("encoder_position_embeddings is required with rotary embeddings.")
            cos, sin = position_embeddings
            query = _apply_rotary_pos_emb(query, cos, sin, unsqueeze_dim=2)
            cos, sin = encoder_position_embeddings
            key = _apply_rotary_pos_emb(key, cos, sin, unsqueeze_dim=2)

        attn_metadata = AttentionMetadata(attn_mask=attention_mask) if attention_mask is not None else None
        hidden_states = self.attn(query, key, value, attn_metadata=attn_metadata)
        hidden_states = hidden_states.flatten(2, 3).contiguous()
        return self.o_proj(hidden_states)


class AnimaTextConditionerBlock(nn.Module):
    def __init__(
        self,
        source_dim: int,
        model_dim: int,
        num_attention_heads: int = 16,
        mlp_ratio: float = 4.0,
        use_self_attention: bool = True,
        use_layer_norm: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.use_self_attention = use_self_attention
        norm_cls = nn.LayerNorm if use_layer_norm else nn.RMSNorm
        norm_kwargs = {} if use_layer_norm else {"eps": 1e-6}

        if use_self_attention:
            self.norm_self_attn = norm_cls(model_dim, **norm_kwargs)
            self.self_attn = AnimaTextConditionerAttention(
                query_dim=model_dim,
                context_dim=model_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=model_dim // num_attention_heads,
                prefix=f"{prefix}.self_attn" if prefix else "self_attn",
            )

        self.norm_cross_attn = norm_cls(model_dim, **norm_kwargs)
        self.cross_attn = AnimaTextConditionerAttention(
            query_dim=model_dim,
            context_dim=source_dim,
            num_attention_heads=num_attention_heads,
            attention_head_dim=model_dim // num_attention_heads,
            prefix=f"{prefix}.cross_attn" if prefix else "cross_attn",
        )
        self.norm_mlp = norm_cls(model_dim, **norm_kwargs)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, int(model_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(model_dim * mlp_ratio), model_dim),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        target_attention_mask: torch.Tensor | None = None,
        source_attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        source_position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if self.use_self_attention:
            norm_hidden_states = self.norm_self_attn(hidden_states)
            attn_hidden_states = self.self_attn(
                norm_hidden_states,
                attention_mask=target_attention_mask,
                position_embeddings=position_embeddings,
                encoder_position_embeddings=position_embeddings,
            )
            hidden_states = hidden_states + attn_hidden_states

        norm_hidden_states = self.norm_cross_attn(hidden_states)
        attn_hidden_states = self.cross_attn(
            norm_hidden_states,
            attention_mask=source_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            position_embeddings=position_embeddings,
            encoder_position_embeddings=source_position_embeddings,
        )
        hidden_states = hidden_states + attn_hidden_states
        hidden_states = hidden_states + self.mlp(self.norm_mlp(hidden_states))
        return hidden_states


def is_conditioner_block_module(name: str, module: nn.Module) -> bool:
    return "blocks" in name and name.split(".")[-1].isdigit()


class AnimaTextConditioner(nn.Module):
    _repeated_blocks = ["AnimaTextConditionerBlock"]
    _hsdp_shard_conditions = [is_conditioner_block_module]

    def __init__(
        self,
        source_dim: int = 1024,
        target_dim: int = 1024,
        model_dim: int = 1024,
        num_layers: int = 6,
        num_attention_heads: int = 16,
        mlp_ratio: float = 4.0,
        target_vocab_size: int = 32128,
        use_self_attention: bool = True,
        use_layer_norm: bool = False,
        min_sequence_length: int = 512,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            source_dim=source_dim,
            target_dim=target_dim,
            model_dim=model_dim,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            mlp_ratio=mlp_ratio,
            target_vocab_size=target_vocab_size,
            use_self_attention=use_self_attention,
            use_layer_norm=use_layer_norm,
            min_sequence_length=min_sequence_length,
            extra_config=kwargs,
        )
        self.embed = nn.Embedding(target_vocab_size, target_dim)
        self.in_proj = nn.Linear(target_dim, model_dim) if model_dim != target_dim else nn.Identity()
        self.rotary_emb = AnimaRotaryEmbedding(model_dim // num_attention_heads)
        self.blocks = nn.ModuleList(
            [
                AnimaTextConditionerBlock(
                    source_dim=source_dim,
                    model_dim=model_dim,
                    num_attention_heads=num_attention_heads,
                    mlp_ratio=mlp_ratio,
                    use_self_attention=use_self_attention,
                    use_layer_norm=use_layer_norm,
                    prefix=f"blocks.{i}",
                )
                for i in range(num_layers)
            ]
        )
        self.out_proj = nn.Linear(model_dim, target_dim)
        self.norm = nn.RMSNorm(target_dim, eps=1e-6)
        self.gradient_checkpointing = False

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @staticmethod
    def _prepare_attention_mask(attention_mask: torch.Tensor | None) -> torch.Tensor | None:
        if attention_mask is None:
            return None
        return attention_mask.to(torch.bool)

    def forward(
        self,
        source_hidden_states: torch.Tensor,
        target_input_ids: torch.Tensor,
        target_attention_mask: torch.Tensor | None = None,
        source_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        target_attention_mask = self._prepare_attention_mask(target_attention_mask)
        source_attention_mask = self._prepare_attention_mask(source_attention_mask)

        hidden_states = self.embed(target_input_ids).to(dtype=source_hidden_states.dtype)
        hidden_states = self.in_proj(hidden_states)

        position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
        source_position_ids = torch.arange(source_hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        source_position_embeddings = self.rotary_emb(hidden_states, source_position_ids)

        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                source_hidden_states,
                target_attention_mask=target_attention_mask,
                source_attention_mask=source_attention_mask,
                position_embeddings=position_embeddings,
                source_position_embeddings=source_position_embeddings,
            )

        hidden_states = self.norm(self.out_proj(hidden_states))

        if target_attention_mask is not None:
            hidden_states = hidden_states * target_attention_mask.to(hidden_states).unsqueeze(-1)

        if hidden_states.shape[1] < self.config.min_sequence_length:
            hidden_states = F.pad(hidden_states, (0, 0, 0, self.config.min_sequence_length - hidden_states.shape[1]))

        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        loaded = set()
        for name, tensor in weights:
            if name not in params:
                continue
            param = params[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, tensor)
            loaded.add(name)
        return loaded
