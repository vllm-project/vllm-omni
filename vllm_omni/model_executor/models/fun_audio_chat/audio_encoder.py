# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audio encoder components for Fun-Audio-Chat.

This module contains the audio encoding components:
- FunAudioChatAudioEncoder: Continuous audio encoder (Whisper-like)
- FunAudioChatDiscreteEncoder: Discrete speech token encoder

Reference: https://huggingface.co/FunAudioLLM/Fun-Audio-Chat-8B
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from vllm.logger import init_logger

logger = init_logger(__name__)


class SinusoidsPositionEmbedding(nn.Module):
    """Sinusoidal position embeddings for audio sequences."""

    def __init__(self, length: int, channels: int, max_timescale: int = 10000):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError("SinusoidsPositionEmbedding needs even channels input")
        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2).float())
        scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        self.register_buffer(
            "positional_embedding",
            torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1),
            persistent=False,
        )

    def forward(self, seqlen: int) -> torch.Tensor:
        return self.positional_embedding[:seqlen, :]


class FunAudioChatAudioAttention(nn.Module):
    """Multi-headed attention for audio encoder.

    This implements attention similar to Whisper's audio encoder attention.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = attention_dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got embed_dim={embed_dim}, num_heads={num_heads})"
            )

        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [total_seq_len, embed_dim] - packed sequence
            cu_seqlens: Cumulative sequence lengths for variable-length batching
            attention_mask: Optional attention mask
        """
        seq_length, _ = hidden_states.size()

        # Project to Q, K, V
        query_states = self.q_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
        key_states = self.k_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
        value_states = self.v_proj(hidden_states).reshape(seq_length, self.num_heads, -1)

        # Reshape for attention: [batch, num_heads, seq_len, head_dim]
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        # Compute attention weights
        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2)) * self.scaling

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape output
        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output


class FunAudioChatAudioEncoderLayer(nn.Module):
    """Single encoder layer for audio processing.

    Structure:
    - Self-attention with pre-norm
    - FFN with pre-norm
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        activation_function: str = "gelu",
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.self_attn = FunAudioChatAudioAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = dropout
        self.activation_fn = ACT2FN[activation_function]
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [total_seq_len, embed_dim]
            cu_seqlens: Cumulative sequence lengths
            attention_mask: Optional attention mask
        """
        # Self-attention with pre-norm
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states

        # FFN with pre-norm
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        # Clamp for fp16 stability
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        return hidden_states


class FunAudioChatAudioEncoder(nn.Module):
    """Continuous audio encoder (Whisper-like architecture).

    Takes mel spectrogram features and produces embeddings for the language model.

    Architecture:
    - 2 Conv1d layers for initial processing
    - 32 transformer encoder layers
    - Average pooling for downsampling
    - Linear projection to output_dim

    Config parameters (from audio_config):
    - num_mel_bins: 128
    - d_model: 1280
    - encoder_layers: 32
    - encoder_attention_heads: 20
    - encoder_ffn_dim: 5120
    - output_dim: 3584 (matches LLM hidden_size)
    - n_window: 100 (chunking window)
    - max_source_positions: 1500
    """

    def __init__(self, config):
        super().__init__()

        # Extract config values
        self.num_mel_bins = getattr(config, "num_mel_bins", 128)
        self.d_model = getattr(config, "d_model", 1280)
        self.encoder_layers = getattr(config, "encoder_layers", 32)
        self.encoder_attention_heads = getattr(config, "encoder_attention_heads", 20)
        self.encoder_ffn_dim = getattr(config, "encoder_ffn_dim", 5120)
        self.output_dim = getattr(config, "output_dim", 3584)
        self.n_window = getattr(config, "n_window", 100)
        self.max_source_positions = getattr(config, "max_source_positions", 1500)
        self.dropout = getattr(config, "dropout", 0.0)
        self.attention_dropout = getattr(config, "attention_dropout", 0.0)
        self.activation_function = getattr(config, "activation_function", "gelu")
        self.scale_embedding = getattr(config, "scale_embedding", False)

        embed_dim = self.d_model
        self.embed_scale = math.sqrt(embed_dim) if self.scale_embedding else 1.0

        # Convolutional frontend
        self.conv1 = nn.Conv1d(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)

        # Positional embeddings
        self.positional_embedding = SinusoidsPositionEmbedding(self.max_source_positions, embed_dim)

        # Transformer encoder layers
        self.layers = nn.ModuleList(
            [
                FunAudioChatAudioEncoderLayer(
                    embed_dim=embed_dim,
                    num_heads=self.encoder_attention_heads,
                    ffn_dim=self.encoder_ffn_dim,
                    dropout=self.dropout,
                    attention_dropout=self.attention_dropout,
                    activation_function=self.activation_function,
                )
                for _ in range(self.encoder_layers)
            ]
        )

        # Output processing
        self.ln_post = nn.LayerNorm(embed_dim)
        self.avg_pooler = nn.AvgPool1d(2, stride=2)
        self.proj = nn.Linear(embed_dim, self.output_dim)

        # Special tokens for audio boundaries
        self.audio_bos_eos_token = nn.Embedding(2, self.output_dim)

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """Compute output lengths after conv and pooling."""
        # After conv2 (stride=2)
        after_cnn = (input_lengths - 1) // 2 + 1
        # After avg_pool (kernel=2, stride=2)
        output_lengths = (after_cnn - 2) // 2 + 1
        return after_cnn, output_lengths

    def _prepare_attention_mask(self, inputs_tensor: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        """Prepare attention mask for variable-length sequences."""
        seq_length = inputs_tensor.shape[0]
        attention_mask = torch.full(
            [1, 1, seq_length, seq_length],
            torch.finfo(inputs_tensor.dtype).min,
            device=inputs_tensor.device,
            dtype=inputs_tensor.dtype,
        )
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0
        return attention_mask

    def forward(
        self,
        input_features: torch.Tensor,
        feature_lens: torch.Tensor | None = None,
        aftercnn_lens: torch.Tensor | None = None,
        speech_maxlen: int | None = None,
    ) -> torch.Tensor:
        """
        Process mel spectrogram features.

        Args:
            input_features: Mel features [num_mel_bins, total_frames] (packed)
                           or [batch, num_mel_bins, frames] (batched)
            feature_lens: Length of each audio in the batch
            aftercnn_lens: Length after CNN processing
            speech_maxlen: Maximum output length

        Returns:
            Audio embeddings [batch, seq_len, output_dim]
        """
        device = input_features.device

        # Handle packed vs batched input
        if input_features.ndim == 2:
            # Packed input: [num_mel_bins, total_frames]
            # Need to unpack using feature_lens
            if feature_lens is None:
                # Assume single audio
                feature_lens = torch.tensor([input_features.shape[1]], device=device)

            batch_size = feature_lens.size(0)

            # Check for empty inputs
            valid_mask = feature_lens > 0
            if not valid_mask.any():
                output_dim = self.output_dim
                return torch.zeros(
                    (batch_size, speech_maxlen or 1, output_dim),
                    device=device,
                    dtype=self.proj.weight.dtype,
                )

            # Split features by length
            input_features_list = input_features.split(feature_lens.tolist(), dim=1)
        else:
            # Batched input: [batch, num_mel_bins, frames]
            batch_size = input_features.shape[0]
            if feature_lens is None:
                feature_lens = torch.full((batch_size,), input_features.shape[2], device=device)
            input_features_list = [input_features[i, :, : feature_lens[i]] for i in range(batch_size)]

        # Process through CNN
        processed_list = []
        processed_lens = []

        for feat in input_features_list:
            if feat.shape[1] == 0:
                continue
            # feat: [num_mel_bins, frames]
            feat = feat.unsqueeze(0)  # [1, num_mel_bins, frames]

            # Conv layers
            x = F.gelu(self.conv1(feat))  # [1, d_model, frames]
            x = F.gelu(self.conv2(x))  # [1, d_model, frames//2]

            x = x.transpose(1, 2)  # [1, seq_len, d_model]

            # Add positional embeddings
            seq_len = x.shape[1]
            pos_emb = self.positional_embedding.positional_embedding[:seq_len, :].to(x.dtype)
            x = x + pos_emb.unsqueeze(0)

            processed_list.append(x.squeeze(0))  # [seq_len, d_model]
            processed_lens.append(x.shape[1])

        if not processed_list:
            return torch.zeros(
                (batch_size, speech_maxlen or 1, self.output_dim),
                device=device,
                dtype=self.proj.weight.dtype,
            )

        # Pack sequences for efficient processing
        hidden_states = torch.cat(processed_list, dim=0)  # [total_seq_len, d_model]

        # Create cumulative sequence lengths
        cu_seqlens = torch.tensor(
            [0] + list(np.cumsum(processed_lens)),
            device=device,
            dtype=torch.int32,
        )

        # Prepare attention mask
        attention_mask = self._prepare_attention_mask(hidden_states, cu_seqlens)

        # Process through transformer layers
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                cu_seqlens=cu_seqlens,
                attention_mask=attention_mask,
            )

        # Split back to individual sequences
        hidden_states_list = hidden_states.split(processed_lens, dim=0)

        # Apply pooling, layer norm, and projection
        output_list = []
        for hs in hidden_states_list:
            # Average pooling
            if hs.shape[0] >= 2:
                hs = (
                    F.avg_pool1d(
                        hs.transpose(0, 1).unsqueeze(0),
                        kernel_size=2,
                        stride=2,
                    )
                    .squeeze(0)
                    .transpose(0, 1)
                )

            # Layer norm and projection
            hs = self.proj(self.ln_post(hs))
            output_list.append(hs)

        # Pad to uniform length
        max_len = max(o.shape[0] for o in output_list)
        if speech_maxlen is not None:
            max_len = speech_maxlen

        output = torch.zeros(
            (len(output_list), max_len, self.output_dim),
            device=device,
            dtype=output_list[0].dtype,
        )

        for i, o in enumerate(output_list):
            seq_len = min(o.shape[0], max_len)
            output[i, :seq_len] = o[:seq_len]

        # Handle case where we filtered out empty sequences
        if len(output_list) < batch_size:
            full_output = torch.zeros(
                (batch_size, max_len, self.output_dim),
                device=device,
                dtype=output.dtype,
            )
            valid_idx = 0
            for i in range(batch_size):
                if feature_lens[i] > 0:
                    full_output[i] = output[valid_idx]
                    valid_idx += 1
            output = full_output

        return output


class FunAudioChatDiscreteEncoder(nn.Module):
    """Discrete speech token encoder.

    Encodes discrete speech tokens (from speech tokenizer) into embeddings
    that can be combined with continuous audio features.

    Architecture:
    - Embedding layer for discrete tokens
    - Grouping (5 tokens → 1 embedding at 5Hz)
    - Linear projection

    Config parameters (from audio_config):
    - codebook_size: 6565
    - output_dim: 3584
    - group_size: 5
    - pad_token_id: padding token
    - continuous_features_mode: "add" or "replace"
    """

    def __init__(self, config):
        super().__init__()

        self.codebook_size = getattr(config, "codebook_size", 6565)
        self.output_dim = getattr(config, "output_dim", 3584)
        self.group_size = getattr(config, "group_size", 5)
        self.pad_token_id = getattr(config, "pad_token_id", 0)
        self.continuous_features_mode = getattr(config, "continuous_features_mode", "add")

        # Embedding for discrete tokens
        self.embed_tokens = nn.Embedding(
            self.codebook_size,
            self.output_dim,
            padding_idx=self.pad_token_id,
        )

        # Projection layers
        self.output_matching = nn.Linear(self.output_dim, self.output_dim, bias=False)
        self.continual_output_matching = nn.Linear(self.output_dim, self.output_dim, bias=False)

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """Compute output lengths after grouping."""
        output_lengths = (input_lengths + self.group_size - 1) // self.group_size
        return input_lengths, output_lengths

    def forward(
        self,
        audio_ids: torch.Tensor,
        continuous_audio_features: torch.Tensor | None = None,
        continuous_audio_output_lengths: torch.Tensor | None = None,
        feature_exist_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Encode discrete speech tokens.

        Args:
            audio_ids: Discrete token IDs [batch, seq_len]
            continuous_audio_features: Optional continuous features to combine
            continuous_audio_output_lengths: Lengths of continuous features
            feature_exist_mask: Mask for samples that have continuous features

        Returns:
            Audio embeddings [batch, grouped_seq_len, output_dim]
        """
        # Embed discrete tokens
        inputs_embeds = self.embed_tokens(audio_ids)  # [batch, seq_len, output_dim]

        # Group embeddings (5 tokens → 1 grouped embedding)
        batch_size = inputs_embeds.shape[0]
        seq_len = inputs_embeds.shape[1]

        # Pad to multiple of group_size
        pad_len = (self.group_size - seq_len % self.group_size) % self.group_size
        if pad_len > 0:
            inputs_embeds = F.pad(inputs_embeds, (0, 0, 0, pad_len))

        # Reshape and average over groups
        grouped_seq_len = (seq_len + pad_len) // self.group_size
        inputs_embeds = inputs_embeds.reshape(batch_size, grouped_seq_len, self.group_size, self.output_dim)
        inputs_embeds_mean = inputs_embeds.mean(dim=2)  # [batch, grouped_seq_len, output_dim]

        # Project
        hidden_states = self.output_matching(inputs_embeds_mean)

        # Combine with continuous features if provided
        if continuous_audio_features is not None:
            # Group and average continuous features similarly
            cont_seq_len = continuous_audio_features.shape[1]
            pad_len = (self.group_size - cont_seq_len % self.group_size) % self.group_size
            if pad_len > 0:
                continuous_audio_features = F.pad(continuous_audio_features, (0, 0, 0, pad_len))

            grouped_cont_len = (cont_seq_len + pad_len) // self.group_size
            continuous_audio_features = continuous_audio_features.reshape(
                continuous_audio_features.shape[0],
                grouped_cont_len,
                self.group_size,
                self.output_dim,
            )
            continuous_audio_features = continuous_audio_features.mean(dim=2)

            continuous_hidden_states = self.continual_output_matching(continuous_audio_features)

            # Combine based on mode
            if feature_exist_mask is not None:
                if self.continuous_features_mode == "add":
                    hidden_states[feature_exist_mask] += continuous_hidden_states
                else:  # "replace"
                    hidden_states[feature_exist_mask] = continuous_hidden_states
            else:
                if self.continuous_features_mode == "add":
                    hidden_states = hidden_states + continuous_hidden_states
                else:
                    hidden_states = continuous_hidden_states

        return hidden_states


__all__ = [
    "FunAudioChatAudioEncoder",
    "FunAudioChatDiscreteEncoder",
    "SinusoidsPositionEmbedding",
]
