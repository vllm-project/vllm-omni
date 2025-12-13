# Copyright (c) 2024 The Qwen Team and The HuggingFace Inc. team.
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# This file is adapted from Bagel's `modeling/bagel/qwen2_navit.py`.
# It is vendored here so vllm-omni does not depend on the external Bagel repo.
#
# NOTE: This implementation relies on `flash_attn` for varlen attention. If your
# environment does not have flash-attn, BagelPipeline will not run.

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import torch
from torch import nn
from transformers.utils import ModelOutput

try:
    from flash_attn import flash_attn_varlen_func  # type: ignore
except Exception as exc:  # pragma: no cover
    flash_attn_varlen_func = None
    _flash_attn_import_error = exc

from transformers.models.qwen2.configuration_qwen2 import Qwen2Config as _Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2MLP,
    Qwen2PreTrainedModel,
    Qwen2RMSNorm,
    Qwen2RotaryEmbedding,
    apply_rotary_pos_emb,
)


class Qwen2Config(_Qwen2Config):
    model_type = "qwen2"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        *args,
        qk_norm: bool = True,
        layer_module: str = "Qwen2DecoderLayer",
        freeze_und: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.qk_norm = qk_norm
        self.layer_module = layer_module
        self.freeze_und = freeze_und


class NaiveCache:
    """Simple cache container used by Bagel for varlen KV."""

    def __init__(self, num_layers: int):
        self.key_cache = {k: None for k in range(num_layers)}
        self.value_cache = {k: None for k in range(num_layers)}

    @property
    def num_layers(self) -> int:
        return len(self.key_cache)

    @property
    def seq_lens(self) -> int:
        if self.key_cache[0] is not None:
            return self.key_cache[0].shape[0]
        return 0


@dataclass
class BaseNavitOutputWithPast(ModelOutput):
    packed_query_sequence: torch.FloatTensor = None
    past_key_values: NaiveCache | None = None


class PackedAttention(Qwen2Attention):
    """Varlen attention for packed sequences, using flash-attn."""

    def __init__(self, config: Qwen2Config, layer_idx: int | None = None):
        super().__init__(config, layer_idx=layer_idx)
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.layer_idx = layer_idx or 0

        if getattr(config, "qk_norm", False):
            self.q_norm = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        packed_query_indexes: torch.Tensor,
        past_key_values: NaiveCache | None = None,
        key_values_lens: torch.Tensor | None = None,
        packed_key_value_indexes: torch.Tensor | None = None,
        update_past_key_values: bool = True,
        is_causal: bool = True,
    ) -> tuple[torch.Tensor, NaiveCache | None]:
        if flash_attn_varlen_func is None:  # pragma: no cover
            raise RuntimeError(
                "BagelPackedAttention requires flash-attn. Import error: "
                + repr(getattr(globals(), "_flash_attn_import_error", None))
            )

        # Project q/k/v
        q = self.q_proj(packed_query_sequence).view(-1, self.num_heads, self.head_dim)
        k = self.k_proj(packed_query_sequence).view(-1, self.num_key_value_heads, self.head_dim)
        v = self.v_proj(packed_query_sequence).view(-1, self.num_key_value_heads, self.head_dim)

        cos, sin = packed_query_position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        q = self.q_norm(q)
        k = self.k_norm(k)

        # Merge with past (only if past KV actually exists).
        #
        # NOTE: For the very first prefill, Bagel passes an allocated NaiveCache
        # but with empty past (key_values_lens == 0 and/or cache entries None).
        # In that case we must NOT try to assign `None` into merged tensors.
        can_merge_past = (
            past_key_values is not None
            and key_values_lens is not None
            and packed_key_value_indexes is not None
            and packed_key_value_indexes.numel() > 0
            and past_key_values.key_cache[self.layer_idx] is not None
            and past_key_values.value_cache[self.layer_idx] is not None
        )
        if can_merge_past:
            past_k = past_key_values.key_cache[self.layer_idx]
            past_v = past_key_values.value_cache[self.layer_idx]
            seqlens = int(query_lens.sum().item() + key_values_lens.sum().item())
            merged_k = k.new_zeros(size=[seqlens, self.num_key_value_heads, self.head_dim])
            merged_v = v.new_zeros(size=[seqlens, self.num_key_value_heads, self.head_dim])
            merged_k[packed_query_indexes] = k
            merged_v[packed_query_indexes] = v
            merged_k[packed_key_value_indexes] = past_k
            merged_v[packed_key_value_indexes] = past_v
            key_values_lens = key_values_lens + query_lens
        else:
            merged_k = k
            merged_v = v
            key_values_lens = query_lens

        cu_seqlens_q = torch.nn.functional.pad(torch.cumsum(query_lens, dim=0), (1, 0)).to(torch.int32)
        cu_seqlens_k = torch.nn.functional.pad(torch.cumsum(key_values_lens, dim=0), (1, 0)).to(torch.int32)

        out = flash_attn_varlen_func(
            q=q,
            k=merged_k,
            v=merged_v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=int(query_lens.max().item()),
            max_seqlen_k=int(key_values_lens.max().item()),
            causal=is_causal,
        )
        out = out.reshape(-1, self.hidden_size)
        out = self.o_proj(out)

        if update_past_key_values and past_key_values is not None:
            past_key_values.key_cache[self.layer_idx] = merged_k
            past_key_values.value_cache[self.layer_idx] = merged_v

        return out, past_key_values


class PackedAttentionMoT(Qwen2Attention):
    """MoT attention: separate projections for VAE tokens in `mode="gen"`."""

    def __init__(self, config: Qwen2Config, layer_idx: int | None = None):
        super().__init__(config, layer_idx=layer_idx)
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.layer_idx = layer_idx or 0

        if getattr(config, "qk_norm", False):
            self.q_norm = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.q_norm_moe_gen = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm_moe_gen = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
            self.q_norm_moe_gen = nn.Identity()
            self.k_norm_moe_gen = nn.Identity()

        self.q_proj_moe_gen = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj_moe_gen = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj_moe_gen = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj_moe_gen = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        packed_query_indexes: torch.Tensor,
        past_key_values: NaiveCache | None = None,
        key_values_lens: torch.Tensor | None = None,
        packed_key_value_indexes: torch.Tensor | None = None,
        update_past_key_values: bool = True,
        is_causal: bool = True,
        mode: str = "und",
        packed_vae_token_indexes: torch.Tensor | None = None,
        packed_text_indexes: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, NaiveCache | None]:
        if flash_attn_varlen_func is None:  # pragma: no cover
            raise RuntimeError(
                "BagelPackedAttentionMoT requires flash-attn. Import error: "
                + repr(getattr(globals(), "_flash_attn_import_error", None))
            )

        # q/k/v projections with optional gen-branch on VAE tokens
        if mode == "und":
            q = self.q_proj(packed_query_sequence).view(-1, self.num_heads, self.head_dim)
            k = self.k_proj(packed_query_sequence).view(-1, self.num_key_value_heads, self.head_dim)
            v = self.v_proj(packed_query_sequence).view(-1, self.num_key_value_heads, self.head_dim)
        else:
            assert packed_vae_token_indexes is not None and packed_text_indexes is not None
            q = torch.empty(
                (packed_query_sequence.shape[0], self.num_heads, self.head_dim),
                device=packed_query_sequence.device,
                dtype=packed_query_sequence.dtype,
            )
            k = torch.empty(
                (packed_query_sequence.shape[0], self.num_key_value_heads, self.head_dim),
                device=packed_query_sequence.device,
                dtype=packed_query_sequence.dtype,
            )
            v = torch.empty(
                (packed_query_sequence.shape[0], self.num_key_value_heads, self.head_dim),
                device=packed_query_sequence.device,
                dtype=packed_query_sequence.dtype,
            )
            q[packed_text_indexes] = self.q_proj(packed_query_sequence[packed_text_indexes]).view(
                -1, self.num_heads, self.head_dim
            )
            k[packed_text_indexes] = self.k_proj(packed_query_sequence[packed_text_indexes]).view(
                -1, self.num_key_value_heads, self.head_dim
            )
            v[packed_text_indexes] = self.v_proj(packed_query_sequence[packed_text_indexes]).view(
                -1, self.num_key_value_heads, self.head_dim
            )
            q[packed_vae_token_indexes] = self.q_proj_moe_gen(packed_query_sequence[packed_vae_token_indexes]).view(
                -1, self.num_heads, self.head_dim
            )
            k[packed_vae_token_indexes] = self.k_proj_moe_gen(packed_query_sequence[packed_vae_token_indexes]).view(
                -1, self.num_key_value_heads, self.head_dim
            )
            v[packed_vae_token_indexes] = self.v_proj_moe_gen(packed_query_sequence[packed_vae_token_indexes]).view(
                -1, self.num_key_value_heads, self.head_dim
            )

        cos, sin = packed_query_position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if mode == "und":
            q = self.q_norm(q)
            k = self.k_norm(k)
        else:
            q_ = torch.zeros_like(q)
            k_ = torch.zeros_like(k)
            q_[packed_text_indexes] = self.q_norm(q[packed_text_indexes])
            k_[packed_text_indexes] = self.k_norm(k[packed_text_indexes])
            q_[packed_vae_token_indexes] = self.q_norm_moe_gen(q[packed_vae_token_indexes])
            k_[packed_vae_token_indexes] = self.k_norm_moe_gen(k[packed_vae_token_indexes])
            q, k = q_, k_

        # Merge with past (only if past KV actually exists).
        can_merge_past = (
            past_key_values is not None
            and key_values_lens is not None
            and packed_key_value_indexes is not None
            and packed_key_value_indexes.numel() > 0
            and past_key_values.key_cache[self.layer_idx] is not None
            and past_key_values.value_cache[self.layer_idx] is not None
        )
        if can_merge_past:
            past_k = past_key_values.key_cache[self.layer_idx]
            past_v = past_key_values.value_cache[self.layer_idx]
            seqlens = int(query_lens.sum().item() + key_values_lens.sum().item())
            merged_k = k.new_zeros(size=[seqlens, self.num_key_value_heads, self.head_dim])
            merged_v = v.new_zeros(size=[seqlens, self.num_key_value_heads, self.head_dim])
            merged_k[packed_query_indexes] = k
            merged_v[packed_query_indexes] = v
            merged_k[packed_key_value_indexes] = past_k
            merged_v[packed_key_value_indexes] = past_v
            key_values_lens = key_values_lens + query_lens
        else:
            merged_k = k
            merged_v = v
            key_values_lens = query_lens

        cu_seqlens_q = torch.nn.functional.pad(torch.cumsum(query_lens, dim=0), (1, 0)).to(torch.int32)
        cu_seqlens_k = torch.nn.functional.pad(torch.cumsum(key_values_lens, dim=0), (1, 0)).to(torch.int32)

        out = flash_attn_varlen_func(
            q=q,
            k=merged_k,
            v=merged_v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=int(query_lens.max().item()),
            max_seqlen_k=int(key_values_lens.max().item()),
            causal=is_causal,
        )
        out = out.reshape(-1, self.hidden_size)
        if mode == "und":
            out = self.o_proj(out)
        else:
            assert packed_vae_token_indexes is not None and packed_text_indexes is not None
            out_ = torch.zeros_like(out)
            out_[packed_text_indexes] = self.o_proj(out[packed_text_indexes])
            out_[packed_vae_token_indexes] = self.o_proj_moe_gen(out[packed_vae_token_indexes])
            out = out_

        if update_past_key_values and past_key_values is not None:
            past_key_values.key_cache[self.layer_idx] = merged_k
            past_key_values.value_cache[self.layer_idx] = merged_v

        return out, past_key_values


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: int | None = None):
        super().__init__()
        self.self_attn = PackedAttention(config, layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        packed_query_indexes: torch.Tensor,
        past_key_values: NaiveCache | None = None,
        key_values_lens: torch.Tensor | None = None,
        packed_key_value_indexes: torch.Tensor | None = None,
        update_past_key_values: bool = True,
        is_causal: bool = True,
        **_: object,
    ):
        residual = packed_query_sequence
        packed_query_sequence = self.input_layernorm(packed_query_sequence)
        packed_query_sequence, past_key_values = self.self_attn.forward_inference(
            packed_query_sequence=packed_query_sequence,
            query_lens=query_lens,
            packed_query_position_embeddings=packed_query_position_embeddings,
            packed_query_indexes=packed_query_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=update_past_key_values,
            is_causal=is_causal,
        )
        packed_query_sequence = residual + packed_query_sequence
        residual = packed_query_sequence
        packed_query_sequence = self.post_attention_layernorm(packed_query_sequence)
        packed_query_sequence = self.mlp(packed_query_sequence)
        packed_query_sequence = residual + packed_query_sequence
        return packed_query_sequence, past_key_values


class Qwen2MoEDecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: int | None = None):
        super().__init__()
        self.self_attn = PackedAttention(config, layer_idx)
        self.mlp = Qwen2MLP(config)
        self.mlp_moe_gen = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        packed_query_indexes: torch.Tensor,
        past_key_values: NaiveCache | None = None,
        key_values_lens: torch.Tensor | None = None,
        packed_key_value_indexes: torch.Tensor | None = None,
        update_past_key_values: bool = True,
        is_causal: bool = True,
        mode: str = "und",
        packed_vae_token_indexes=None,
        packed_text_indexes=None,
    ):
        residual = packed_query_sequence
        packed_query_sequence = self.input_layernorm(packed_query_sequence)
        packed_query_sequence, past_key_values = self.self_attn.forward_inference(
            packed_query_sequence=packed_query_sequence,
            query_lens=query_lens,
            packed_query_position_embeddings=packed_query_position_embeddings,
            packed_query_indexes=packed_query_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=update_past_key_values,
            is_causal=is_causal,
        )
        packed_query_sequence = residual + packed_query_sequence

        residual = packed_query_sequence
        packed_query_sequence = self.post_attention_layernorm(packed_query_sequence)
        if mode == "und":
            packed_query_sequence = self.mlp(packed_query_sequence)
        else:
            packed_query_sequence_ = torch.zeros_like(packed_query_sequence).to(torch.bfloat16)
            packed_query_sequence_[packed_text_indexes] = self.mlp(packed_query_sequence[packed_text_indexes])
            packed_query_sequence_[packed_vae_token_indexes] = self.mlp_moe_gen(
                packed_query_sequence[packed_vae_token_indexes]
            )
            packed_query_sequence = packed_query_sequence_
        packed_query_sequence = residual + packed_query_sequence
        return packed_query_sequence, past_key_values


class Qwen2MoTDecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen2Config,
        layer_idx: int | None = None,
        attn_module: type[Qwen2Attention] | None = PackedAttentionMoT,
    ):
        super().__init__()
        self.self_attn = attn_module(config, layer_idx)
        self.mlp = Qwen2MLP(config)
        self.mlp_moe_gen = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm_moe_gen = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm_moe_gen = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        packed_query_indexes: torch.Tensor,
        past_key_values: NaiveCache | None = None,
        key_values_lens: torch.Tensor | None = None,
        packed_key_value_indexes: torch.Tensor | None = None,
        update_past_key_values: bool = True,
        is_causal: bool = True,
        mode: str = "und",
        packed_vae_token_indexes=None,
        packed_text_indexes=None,
        **_: object,
    ):
        residual = packed_query_sequence
        if mode == "und":
            packed_query_sequence = self.input_layernorm(packed_query_sequence)
        else:
            packed_query_sequence_ = torch.zeros_like(packed_query_sequence)
            packed_query_sequence_[packed_text_indexes] = self.input_layernorm(
                packed_query_sequence[packed_text_indexes]
            )
            packed_query_sequence_[packed_vae_token_indexes] = self.input_layernorm_moe_gen(
                packed_query_sequence[packed_vae_token_indexes]
            )
            packed_query_sequence = packed_query_sequence_

        packed_query_sequence, past_key_values = self.self_attn.forward_inference(
            packed_query_sequence=packed_query_sequence,
            query_lens=query_lens,
            packed_query_position_embeddings=packed_query_position_embeddings,
            packed_query_indexes=packed_query_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=update_past_key_values,
            is_causal=is_causal,
            mode=mode,
            packed_vae_token_indexes=packed_vae_token_indexes,
            packed_text_indexes=packed_text_indexes,
        )
        packed_query_sequence = residual + packed_query_sequence

        residual = packed_query_sequence
        if mode == "und":
            packed_query_sequence = self.post_attention_layernorm(packed_query_sequence)
            packed_query_sequence = self.mlp(packed_query_sequence)
        else:
            text_h = self.post_attention_layernorm(packed_query_sequence[packed_text_indexes]).to(torch.bfloat16)
            vae_h = self.post_attention_layernorm_moe_gen(packed_query_sequence[packed_vae_token_indexes]).to(
                torch.bfloat16
            )
            packed_query_sequence_ = torch.zeros_like(packed_query_sequence).to(torch.bfloat16)
            packed_query_sequence_[packed_text_indexes] = self.mlp(text_h)
            packed_query_sequence_[packed_vae_token_indexes] = self.mlp_moe_gen(vae_h)
            packed_query_sequence = packed_query_sequence_
        packed_query_sequence = residual + packed_query_sequence
        return packed_query_sequence, past_key_values


Decoder_layer_dict = {
    "Qwen2DecoderLayer": Qwen2DecoderLayer,
    "Qwen2MoEDecoderLayer": Qwen2MoEDecoderLayer,
    "Qwen2MoTDecoderLayer": partial(Qwen2MoTDecoderLayer, attn_module=PackedAttentionMoT),
}


class Qwen2Model(Qwen2PreTrainedModel):
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.use_moe = "Mo" in config.layer_module

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        layer_module = Decoder_layer_dict[config.layer_module]
        self.layers = nn.ModuleList([layer_module(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])

        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if self.use_moe:
            self.norm_moe_gen = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)
        self.post_init()

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_ids: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: NaiveCache | None = None,
        key_values_lens: torch.Tensor | None = None,
        packed_key_value_indexes: torch.Tensor | None = None,
        update_past_key_values: bool = True,
        is_causal: bool = True,
        mode: str = "und",
        packed_vae_token_indexes=None,
        packed_text_indexes=None,
    ) -> BaseNavitOutputWithPast:
        cos, sin = self.rotary_emb(packed_query_sequence, packed_query_position_ids.unsqueeze(0))
        packed_query_position_embeddings = (cos.squeeze(0), sin.squeeze(0))

        extra_inputs = {}
        if self.use_moe:
            extra_inputs.update(mode=mode)
            if mode == "gen":
                assert packed_vae_token_indexes is not None and packed_text_indexes is not None
                extra_inputs.update(
                    packed_vae_token_indexes=packed_vae_token_indexes, packed_text_indexes=packed_text_indexes
                )

        for decoder_layer in self.layers:
            packed_query_sequence, past_key_values = decoder_layer.forward_inference(
                packed_query_sequence=packed_query_sequence,
                query_lens=query_lens,
                packed_query_position_embeddings=packed_query_position_embeddings,
                packed_query_indexes=packed_query_indexes,
                past_key_values=past_key_values,
                key_values_lens=key_values_lens,
                packed_key_value_indexes=packed_key_value_indexes,
                update_past_key_values=update_past_key_values,
                is_causal=is_causal,
                **extra_inputs,
            )

        if self.use_moe:
            if mode == "und":
                packed_query_sequence = self.norm(packed_query_sequence)
            else:
                packed_query_sequence_ = torch.zeros_like(packed_query_sequence)
                packed_query_sequence_[packed_text_indexes] = self.norm(packed_query_sequence[packed_text_indexes])
                packed_query_sequence_[packed_vae_token_indexes] = self.norm_moe_gen(
                    packed_query_sequence[packed_vae_token_indexes]
                )
                packed_query_sequence = packed_query_sequence_
        else:
            packed_query_sequence = self.norm(packed_query_sequence)

        return BaseNavitOutputWithPast(packed_query_sequence=packed_query_sequence, past_key_values=past_key_values)


class Qwen2ForCausalLM(Qwen2PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def init_moe(self):
        # Copy-init *_moe_gen params from base params if present.
        for name, param in self.named_parameters():
            if "moe_gen" in name:
                original_name = name.replace("_moe_gen", "")
                if original_name in self.state_dict():
                    param.data.copy_(self.state_dict()[original_name].data)

    def forward_inference(self, *args, **kwargs) -> BaseNavitOutputWithPast:
        return self.model.forward_inference(*args, **kwargs)
