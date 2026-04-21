# SPDX-License-Identifier: Apache-2.0
"""Fun-Audio-Chat continuous audio encoder.

Line-by-line port of the reference
    github.com/FunAudioLLM/Fun-Audio-Chat
    funaudiochat/modeling_funaudiochat.py L77-499
(FunAudioChatAudioAttention, FunAudioChatAudioEncoderLayer,
SinusoidsPositionEmbedding, FunAudioChatAudioEncoder,
FunAudioChatDiscreteEncoder).

Weight names match the reference 1:1 so an AutoWeightsLoader
mapping `continuous_audio_tower.*` / `audio_tower.*` to these
classes loads cleanly from the checkpoint.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput

from vllm_omni.transformers_utils.configs.fun_audio_chat import (
    FunAudioChatAudioEncoderConfig,
)

__all__ = [
    "FunAudioChatAudioEncoder",
    "FunAudioChatDiscreteEncoder",
]


def _eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,  # [1, H, T, D]
    key: torch.Tensor,    # [1, H, T, D]
    value: torch.Tensor,  # [1, H, T, D]
    attention_mask: torch.Tensor | None,
    dropout: float,
    scaling: float,
    **_: object,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Matches transformers' eager_attention_forward signature.
    attn = torch.matmul(query, key.transpose(-1, -2)) * scaling
    if attention_mask is not None:
        attn = attn + attention_mask
    attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(query.dtype)
    if dropout > 0:
        attn = F.dropout(attn, p=dropout, training=module.training)
    out = torch.matmul(attn, value)  # [1, H, T, D]
    out = out.transpose(1, 2).contiguous()  # [1, T, H, D]
    return out, attn


class FunAudioChatAudioAttention(nn.Module):
    """Ref L100-171. Packed attention using `cu_seqlens` for variable-length chunks."""

    def __init__(self, config: FunAudioChatAudioEncoderConfig) -> None:
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.dropout = config.attention_dropout
        self.head_dim = self.embed_dim // self.num_heads
        self.num_key_value_groups = 1  # for eager attention compatibility
        self.config = config
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by num_heads "
                f"({self.num_heads})."
            )
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = 0.0
        self.is_decoder = False
        self.is_causal = False
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,  # [T, D]
        cu_seqlens: torch.Tensor,     # [n_chunks+1]
        attention_mask: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        T, _ = hidden_states.shape
        q = self.q_proj(hidden_states).reshape(T, self.num_heads, -1)
        k = self.k_proj(hidden_states).reshape(T, self.num_heads, -1)
        v = self.v_proj(hidden_states).reshape(T, self.num_heads, -1)
        # [T, H, D] -> [1, H, T, D]
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)
        attn_output, _ = _eager_attention_forward(
            self,
            q, k, v,
            attention_mask=attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            cu_seq_lens_q=cu_seqlens,
            cu_seq_lens_k=cu_seqlens,
            max_length_q=(cu_seqlens[1:] - cu_seqlens[:-1]).max(),
            max_length_k=(cu_seqlens[1:] - cu_seqlens[:-1]).max(),
            is_causal=False,
        )
        # attn_output is [1, T, H, D]; collapse to [T, D]
        attn_output = attn_output.reshape(T, -1).contiguous()
        return self.out_proj(attn_output)


class FunAudioChatAudioEncoderLayer(nn.Module):
    """Ref L174-225. Pre-LN block: LN -> self_attn (+residual) -> LN -> FFN (+residual)."""

    def __init__(self, config: FunAudioChatAudioEncoderConfig) -> None:
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = FunAudioChatAudioAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor]:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, cu_seqlens=cu_seqlens, attention_mask=attention_mask
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16:
            clamp = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp, max=clamp)
        return (hidden_states,)


class SinusoidsPositionEmbedding(nn.Module):
    """Ref L228-243. Fixed sinusoid pos embeddings, d_model must be even."""

    def __init__(self, length: int, channels: int, max_timescale: float = 10000) -> None:
        super().__init__()
        if channels % 2 != 0:
            raise ValueError("SinusoidsPositionEmbedding needs even channels")
        log_ts_incr = np.log(max_timescale) / (channels // 2 - 1)
        inv_ts = torch.exp(-log_ts_incr * torch.arange(channels // 2).float())
        scaled = torch.arange(length)[:, None] * inv_ts[None, :]
        self.register_buffer(
            "positional_embedding",
            torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=1),
            persistent=False,
        )

    def forward(self, seqlen: int) -> torch.Tensor:
        return self.positional_embedding[:seqlen, :]


class FunAudioChatAudioEncoder(nn.Module):
    """Continuous audio encoder. Ref L252-499.

    Weight layout (checkpoint):
        continuous_audio_tower.{conv1, conv2, ln_post, avg_pooler, proj,
                                positional_embedding.positional_embedding,
                                audio_bos_eos_token,
                                layers.{i}.{self_attn, self_attn_layer_norm,
                                          fc1, fc2, final_layer_norm}}
    """

    def __init__(self, config: FunAudioChatAudioEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.dropout = config.dropout
        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.n_window = config.n_window
        self.conv1 = nn.Conv1d(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)
        self.layers = nn.ModuleList(
            [FunAudioChatAudioEncoderLayer(config) for _ in range(config.encoder_layers)]
        )
        self.ln_post = nn.LayerNorm(config.d_model)
        self.avg_pooler = nn.AvgPool1d(2, stride=2)
        self.proj = nn.Linear(config.d_model, config.output_dim)
        self.positional_embedding = SinusoidsPositionEmbedding(
            self.max_source_positions, embed_dim
        )
        self.audio_bos_eos_token = nn.Embedding(2, config.output_dim)

    # Ref L290-304.
    def _prepare_attention_mask(
        self, inputs_tensor: torch.Tensor, cu_seqlens: torch.Tensor
    ) -> torch.Tensor:
        seq_length = inputs_tensor.shape[0]
        attention_mask = torch.full(
            [1, 1, seq_length, seq_length],
            torch.finfo(inputs_tensor.dtype).min,
            device=inputs_tensor.device,
            dtype=inputs_tensor.dtype,
        )
        for i in range(1, len(cu_seqlens)):
            attention_mask[
                ...,
                cu_seqlens[i - 1] : cu_seqlens[i],
                cu_seqlens[i - 1] : cu_seqlens[i],
            ] = 0
        return attention_mask

    # Ref L454-491.
    def padded_and_mask_function(
        self,
        tensor_list: list[torch.Tensor],
        tensor_len: torch.Tensor,
        padding_value: float = 0,
        padding_side: str = "right",  # only "right" is used by ref
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        max_len = int(tensor_len.max().item())
        dim = tensor_list[0].shape[0]
        device = tensor_list[0].device
        dtype = tensor_list[0].dtype
        padded_tensor = torch.full(
            (len(tensor_list), dim, max_len),
            fill_value=padding_value,
            dtype=dtype,
            device=device,
        )
        batch_mask = torch.zeros(
            (len(tensor_len), max_len), dtype=torch.long, device=device
        )
        for i, length in enumerate(tensor_len):
            length_i = int(length.item())
            batch_mask[i, :length_i] = 1
            padded_tensor[i, :, :length_i] = tensor_list[i]
        feature_lens_after_cnn = (tensor_len - 1) // 2 + 1
        max_len_after_cnn = int(feature_lens_after_cnn.max().item())
        batch_mask_after_cnn = torch.zeros(
            (len(tensor_len), max_len_after_cnn), dtype=torch.long, device=device
        )
        for i, length in enumerate(feature_lens_after_cnn):
            batch_mask_after_cnn[i, : int(length.item())] = 1
        return padded_tensor, batch_mask.unsqueeze(1), batch_mask_after_cnn.bool()

    # Ref L493-499.
    def _get_feat_extract_output_lengths(
        self, input_lengths: torch.LongTensor
    ) -> tuple[torch.LongTensor, torch.LongTensor]:
        """Lengths after conv stack and after the avg_pool1d(k=2,s=2)."""
        after_cnn = (input_lengths - 1) // 2 + 1
        after_pool = (after_cnn - 2) // 2 + 1
        return after_cnn, after_pool

    def forward(
        self,
        input_features: torch.Tensor,   # [num_mel_bins, sum(feature_lens)] (flattened)
        feature_lens: torch.LongTensor, # [B]
        aftercnn_lens: torch.LongTensor,# [B]
        speech_maxlen: int,
    ) -> BaseModelOutput:
        original_batch_size = feature_lens.size(0)
        device = input_features.device

        valid_mask = feature_lens > 0
        valid_indices = torch.where(valid_mask)[0]

        if valid_indices.numel() == 0:
            output_dim = self.proj.out_features
            return BaseModelOutput(
                last_hidden_state=torch.zeros(
                    (original_batch_size, speech_maxlen, output_dim),
                    device=device,
                    dtype=self.proj.weight.dtype,
                )
            )

        input_features_list = input_features.split(feature_lens.tolist(), dim=1)
        valid_input_features = torch.cat(
            [input_features_list[i] for i in valid_indices], dim=1
        )
        valid_feature_lens = feature_lens[valid_mask]
        valid_aftercnn_lens = aftercnn_lens[valid_mask]

        # Chunk into windows of n_window*2 mel frames (ref: training-time chunking).
        full_chunk_len = self.n_window * 2
        chunk_num = torch.ceil(valid_feature_lens.float() / full_chunk_len).long()
        chunk_lengths_list: list[int] = []
        for i, length in enumerate(valid_feature_lens):
            n_chunks = int(chunk_num[i].item())
            if n_chunks == 0:
                continue
            chunk_lengths_list.extend([full_chunk_len] * (n_chunks - 1))
            last_chunk_len = int(length.item()) % full_chunk_len
            if last_chunk_len == 0:
                last_chunk_len = full_chunk_len
            chunk_lengths_list.append(last_chunk_len)
        chunk_lengths = torch.tensor(chunk_lengths_list, dtype=torch.long, device=device)

        chunk_list = valid_input_features.split(chunk_lengths.tolist(), dim=1)
        padded_feature, padded_mask, padded_mask_after_cnn = self.padded_and_mask_function(
            chunk_list, chunk_lengths, padding_value=0, padding_side="right"
        )
        padded_feature = padded_feature.to(self.conv1.weight.dtype)
        padded_mask = padded_mask.to(padded_feature.dtype)
        padded_embed = F.gelu(self.conv1(padded_feature)) * padded_mask
        padded_embed = F.gelu(self.conv2(padded_embed)).transpose(1, 2)

        padded_embed = padded_embed + self.positional_embedding.positional_embedding[
            : padded_embed.shape[1], :
        ].unsqueeze(0).to(padded_embed.dtype)

        hidden_states = padded_embed[padded_mask_after_cnn]
        cu_seqlens = torch.cat(
            (
                torch.zeros(1, device=padded_mask_after_cnn.device, dtype=torch.int32),
                padded_mask_after_cnn.sum(1).cumsum(0),
            )
        ).to(torch.int32)
        attention_mask = self._prepare_attention_mask(hidden_states, cu_seqlens)

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states, cu_seqlens=cu_seqlens, attention_mask=attention_mask
            )[0]

        hidden_states_list = hidden_states.split(valid_aftercnn_lens.tolist(), dim=0)

        # Pool + ln + proj with ZeRO-3-friendly batched apply (ref L408-434).
        pooled_list: list[torch.Tensor] = []
        pooled_lengths: list[int] = []
        for each in hidden_states_list:
            seq_len = each.shape[0]
            if seq_len >= 2:
                pooled = F.avg_pool1d(
                    each.transpose(0, 1), kernel_size=2, stride=2
                ).transpose(0, 1)
            else:
                pooled = each
            pooled_list.append(pooled)
            pooled_lengths.append(pooled.shape[0])
        pooled_concat = torch.cat(pooled_list, dim=0)
        processed_concat = self.proj(self.ln_post(pooled_concat))
        processed_audio_list = list(processed_concat.split(pooled_lengths, dim=0))

        output_dim = processed_audio_list[0].shape[-1]
        output_hidden_states = torch.zeros(
            (original_batch_size, speech_maxlen, output_dim),
            dtype=processed_audio_list[0].dtype,
            device=device,
        )
        for valid_idx, processed in zip(valid_indices, processed_audio_list):
            seq_len = min(processed.shape[0], speech_maxlen)
            output_hidden_states[valid_idx, :seq_len] = processed[:seq_len]

        return BaseModelOutput(last_hidden_state=output_hidden_states)


class FunAudioChatDiscreteEncoder(nn.Module):
    """Discrete speech-token encoder (`audio_tower`). Ref L502-560.

    Weight layout:
        audio_tower.embed_tokens
        audio_tower.output_matching
        audio_tower.continual_output_matching
    """

    def __init__(self, config: FunAudioChatAudioEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.group_size = config.group_size
        self.hidden_size = config.output_dim
        self.continuous_features_mode = getattr(
            config, "continuous_features_mode", "add"
        )  # checkpoint default is "replace"
        self.embed_tokens = nn.Embedding(
            config.codebook_size, self.hidden_size, self.padding_idx
        )
        self.output_matching = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.continual_output_matching = nn.Linear(
            self.hidden_size, self.hidden_size, bias=False
        )

    def forward(
        self,
        audio_ids: torch.LongTensor,  # [B, T_padded_to_group_size]
        continuous_audio_features: torch.Tensor | None = None,
        feature_exist_mask: torch.Tensor | None = None,
        return_dict: bool = True,
    ) -> BaseModelOutput | tuple:
        inputs_embeds = self.embed_tokens(audio_ids)  # [B, T, H]
        # Group-pool by group_size: reshape, average.
        inputs_embeds_grouped = inputs_embeds.reshape(
            inputs_embeds.shape[0], -1, self.group_size, self.hidden_size
        )
        inputs_embeds_mean = inputs_embeds_grouped.mean(dim=2)  # [B, T/group, H]
        hidden_states = self.output_matching(inputs_embeds_mean)

        continuous_audio_hidden_states = None
        if continuous_audio_features is not None:
            cont = continuous_audio_features.reshape(
                continuous_audio_features.shape[0], -1, self.group_size, self.hidden_size
            ).mean(dim=2)
            continuous_audio_hidden_states = self.continual_output_matching(cont)
            if self.continuous_features_mode == "add":
                hidden_states[feature_exist_mask] += continuous_audio_hidden_states
            else:  # "replace" — checkpoint default
                hidden_states[feature_exist_mask] = continuous_audio_hidden_states

        # ref returns inputs_embeds (reshaped to H*group), hidden, continuous as hidden_states tuple
        inputs_embeds_flat = inputs_embeds.reshape(
            inputs_embeds.shape[0], -1, self.group_size * self.hidden_size
        )
        encoder_states = (inputs_embeds_flat, hidden_states, continuous_audio_hidden_states)
        if not return_dict:
            return tuple(v for v in (hidden_states, encoder_states, None) if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=None,
        )

    # Ref L554-560.
    def _get_feat_extract_output_lengths(
        self, input_lengths: torch.LongTensor
    ) -> tuple[torch.LongTensor, torch.LongTensor]:
        return input_lengths, (input_lengths + self.group_size - 1) // self.group_size
