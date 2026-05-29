# Copyright 2022 The OpenAI Authors and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""GLM-4-Voice WhisperVQ encoder, trimmed to the inference-only path
needed by Kimi-Audio's input-audio tokenizer."""

import math
from dataclasses import dataclass

import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.whisper.modeling_whisper import WhisperAttention as _WhisperAttention

from vllm_omni.transformers_utils.configs.glm4_voice import WhisperVQConfig


@dataclass
class QuantizedBaseModelOutput(BaseModelOutput):
    quantized_token_ids: torch.LongTensor | None = None


def vector_quantize(inputs, codebook):
    embedding_size = codebook.size(1)
    inputs_flatten = inputs.reshape(-1, embedding_size)
    codebook_sqr = torch.sum(codebook**2, dim=1)
    inputs_sqr = torch.sum(inputs_flatten**2, dim=1, keepdim=True)
    # Compute the distances to the codebook
    distances = torch.addmm(codebook_sqr + inputs_sqr, inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0)

    _, indices_flatten = torch.min(distances, dim=1)
    codes_flatten = torch.index_select(codebook, dim=0, index=indices_flatten)
    codes = codes_flatten.view_as(inputs)
    return codes, indices_flatten, distances


class CausalConv1d(nn.Conv1d):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, **kwargs
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
            **kwargs,
        )

        self.left_padding = dilation * (kernel_size - 1)

    def forward(self, inp):
        x = torch.nn.functional.pad(inp.unsqueeze(2), (self.left_padding, 0, 0, 0)).squeeze(2)

        return super().forward(x)


def sinusoids(length: int, channels: int, max_timescale: float = 10000) -> torch.Tensor:
    """Returns sinusoids for positional embedding"""
    if channels % 2 != 0:
        raise ValueError(f"channels must be divisible by 2 for sinusoidal positional embeddings, got {channels}.")
    log_timescale_increment = math.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length).view(-1, 1) * inv_timescales.view(1, -1)
    return torch.cat([scaled_time.sin(), scaled_time.cos()], dim=1)


class WhisperVQEncoderLayer(nn.Module):
    """Pre-norm Whisper encoder layer (inference-only). Same shape as
    upstream's ``WhisperEncoderLayer`` so the GLM-4-Voice checkpoint
    keys load unchanged; the dropout / training-time fp16 clamp are
    omitted because we never train this module."""

    def __init__(self, config: WhisperVQConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = _WhisperAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            config=config,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.activation_fn = ACT2FN[config.activation_function]
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states
        return (hidden_states, attn_weights) if output_attentions else (hidden_states,)


class WhisperPreTrainedModel(PreTrainedModel):
    config_class = WhisperVQConfig
    base_model_prefix = "model"
    main_input_name = "input_features"
    supports_gradient_checkpointing = True
    _no_split_modules = ["WhisperEncoderLayer", "WhisperDecoderLayer"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, WhisperVQEncoder):
            with torch.no_grad():
                embed_positions = module.embed_positions.weight
                embed_positions.copy_(sinusoids(*embed_positions.shape))

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers
        """
        input_lengths = (input_lengths - 1) // 2 + 1

        return input_lengths


class WhisperVQEncoder(WhisperPreTrainedModel):
    """GLM-4-Voice's VQ-augmented Whisper encoder. Stripped to the
    fixed-config inference path: causal-conv stem, encoder_only layer
    stack, average-pool downsampling, vector-quantization head."""

    def __init__(self, config: WhisperVQConfig):
        super().__init__(config)
        self.config = config

        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        # Hardcoded for GLM-4-Voice: encoder_causal_convolution=True,
        # quantize_encoder_only=True, pooling_type="avg".
        self.conv1 = CausalConv1d(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = CausalConv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)
        self.embed_positions = nn.Embedding(self.max_source_positions, embed_dim)
        self.embed_positions.requires_grad_(False)
        self.layers = nn.ModuleList([WhisperVQEncoderLayer(config) for _ in range(config.quantize_position)])
        self.pooling_layer = nn.AvgPool1d(kernel_size=config.pooling_kernel_size)

        # VQ codebook + a second positional embedding sized to the
        # post-pool sequence length.
        self.codebook = nn.Embedding(config.quantize_vocab_size, embed_dim)
        post_pool_positions = math.ceil(self.max_source_positions / config.pooling_kernel_size)
        self.embed_positions2 = nn.Embedding(post_pool_positions, embed_dim)
        self.embed_positions2.weight.data.copy_(self.embed_positions.weight.data[:post_pool_positions])

        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.conv1

    def set_input_embeddings(self, value: nn.Module):
        self.conv1 = value

    def get_block_causal_attention_mask(self, attention_mask, block_size=50):
        dtype = self.dtype
        batch_size, seq_length = attention_mask.shape
        causal_mask = torch.torch.tril(
            torch.ones(1, seq_length, seq_length, dtype=torch.bool, device=attention_mask.device)
        )
        block_square_mask = []
        for start in range(0, seq_length, block_size):
            end = min(start + block_size, seq_length)
            length = end - start
            block_square_mask.append(causal_mask.new_ones((length, length)))
        block_square_mask = torch.block_diag(*block_square_mask)
        block_causal_mask = causal_mask | block_square_mask
        block_causal_mask = block_causal_mask & attention_mask[:, None, :]
        block_causal_mask = block_causal_mask.to(dtype=dtype)  # fp16 compatibility
        block_causal_mask = (1.0 - block_causal_mask) * torch.finfo(dtype).min
        block_causal_mask = block_causal_mask.unsqueeze(1)
        return block_causal_mask

    @torch.inference_mode()
    def forward(self, input_features, attention_mask=None, **_unused):
        """Mel features → discrete codec IDs. Single inference path:
        causal-conv stem + ``quantize_position`` encoder layers + avg-pool
        + VQ codebook lookup. Returns ``QuantizedBaseModelOutput``."""
        batch_size, _, raw_seq_len = input_features.shape
        stride = self.conv1.stride[0] * self.conv2.stride[0]
        seq_length = raw_seq_len // stride

        attention_mask = attention_mask[:, ::stride]
        extended_attention_mask = self.get_block_causal_attention_mask(
            attention_mask,
            block_size=self.config.quantize_causal_block_size,
        )

        hidden_states = nn.functional.gelu(self.conv1(input_features))
        hidden_states = nn.functional.gelu(self.conv2(hidden_states))
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = hidden_states + self.embed_positions.weight[:seq_length]

        for idx, encoder_layer in enumerate(self.layers):
            (hidden_states,) = encoder_layer(
                hidden_states,
                extended_attention_mask,
                layer_head_mask=None,
                output_attentions=False,
            )

            if idx + 1 == self.config.pooling_position:
                hidden_states = hidden_states.permute(0, 2, 1)
                kernel = self.config.pooling_kernel_size
                if hidden_states.shape[-1] % kernel != 0:
                    hidden_states = torch.nn.functional.pad(
                        hidden_states, (0, kernel - hidden_states.shape[-1] % kernel)
                    )
                hidden_states = self.pooling_layer(hidden_states).permute(0, 2, 1)
                attention_mask = attention_mask[:, ::kernel]
                extended_attention_mask = self.get_block_causal_attention_mask(
                    attention_mask,
                    block_size=self.config.quantize_causal_block_size // kernel,
                )

            if idx + 1 == self.config.quantize_position:
                hidden_quantized, indices_flat, _ = vector_quantize(hidden_states, self.codebook.weight)
                quantized_token_ids = indices_flat.reshape(batch_size, hidden_quantized.shape[1])
                hidden_states = hidden_quantized + self.embed_positions2.weight[: hidden_quantized.shape[1]]

        return QuantizedBaseModelOutput(
            last_hidden_state=hidden_states,
            quantized_token_ids=quantized_token_ids,
        )
