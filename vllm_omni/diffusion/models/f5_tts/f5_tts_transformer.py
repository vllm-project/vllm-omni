# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
F5 TTS DiT Model for vLLM-Omni.

F5TTS-style DiT transformer for flow-matching based TTS.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from cache_dit import ForwardPattern
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.cache.cachedit import CacheDiTAdapterConfig
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.distributed.hsdp_utils import is_transformer_block_module
from vllm_omni.diffusion.layers.rope import RotaryEmbedding

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizationConfig,
    )

logger = init_logger(__name__)


class F5TTSSelfAttention(nn.Module):
    """
    Self-attention for F5 TTS using vLLM parallel layers + shared attention backend.

    Uses QKVParallelLinear for fused Q/K/V projections (TP-ready),
    shared RotaryEmbedding for RoPE with CUDA kernel dispatch,
    and shared Attention layer with pluggable backends (Flash, SDPA, etc.).
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout: float = 0.0,
        pe_attn_head: int | None = None,
        attn_mask_enabled: bool = True,
        quant_config: QuantizationConfig | None = None,
    ):
        super().__init__()

        self.dim = dim
        self.head_dim = attention_head_dim
        self.pe_attn_head = pe_attn_head
        self.attn_mask_enabled = attn_mask_enabled

        self.to_qkv = QKVParallelLinear(
            hidden_size=dim,
            head_size=attention_head_dim,
            total_num_heads=num_attention_heads,
            bias=True,
            quant_config=quant_config,
        )
        # TP-local head counts
        self.num_heads = self.to_qkv.num_heads
        self.num_kv_heads = self.to_qkv.num_kv_heads
        self.inner_dim = self.num_heads * attention_head_dim

        # RowParallelLinear for output projection (TP-aware gather)
        self.to_out = RowParallelLinear(
            num_attention_heads * attention_head_dim,
            dim,
            bias=True,
            input_is_parallel=True,
            return_bias=False,
            quant_config=quant_config,
        )

        # Shared RoPE. is_neox_style=False selects interleaved (GPT-J)
        # rotation, matching the layout the released F5 checkpoints were
        # trained with.
        self.rope = RotaryEmbedding(is_neox_style=False)

        # Shared attention backend (auto-selects Flash / SDPA / etc.)
        self.attn = Attention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            softmax_scale=self.head_dim**-0.5,
            causal=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope_emb: tuple[torch.Tensor, torch.Tensor] | None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, T, D]
            rope_emb: (cos, sin) each [T, head_dim/2] or None
            attention_mask: [B, T] boolean padding mask or None
        Returns:
            [B, T, D]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Fused QKV projection
        hidden_states = hidden_states.contiguous()
        qkv, _ = self.to_qkv(hidden_states)
        q_size = self.num_heads * self.head_dim
        kv_size = self.to_qkv.num_kv_heads * self.head_dim
        query, key, value = qkv.split([q_size, kv_size, kv_size], dim=-1)

        # Reshape to [B, T, H, D] for shared RoPE + attention
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key = key.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        value = value.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Apply rotary embeddings via shared RotaryEmbedding
        if rope_emb is not None:
            cos, sin = rope_emb
            cos = cos.to(device=query.device, dtype=query.dtype)
            sin = sin.to(device=query.device, dtype=query.dtype)
            if self.pe_attn_head is not None:
                # Partial RoPE: only first pe_attn_head heads
                rotary_head_count = min(self.pe_attn_head, self.num_heads)
                query[:, :, :rotary_head_count] = self.rope(
                    query[:, :, :rotary_head_count], cos, sin
                )
                key[:, :, :rotary_head_count] = self.rope(
                    key[:, :, :rotary_head_count], cos, sin
                )
            else:
                query = self.rope(query, cos, sin)
                key = self.rope(key, cos, sin)

        # Build attention metadata for mask
        attn_metadata: AttentionMetadata | None = None
        if self.attn_mask_enabled and attention_mask is not None:
            attn_metadata = AttentionMetadata(attn_mask=attention_mask)

        # Shared attention backend (Flash / SDPA / etc.)
        out = self.attn(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            attn_metadata=attn_metadata,
        )
        # out: [B, T, H, D]

        # Flatten head dim
        out = out.reshape(batch_size, seq_len, -1).contiguous()

        # Output projection (RowParallel gathers across TP ranks)
        out = self.to_out(out)

        if attention_mask is not None:
            out = out.masked_fill(~attention_mask.unsqueeze(-1), 0.0)

        return out


class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x: torch.Tensor):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)  # [B, 1, D]
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class ConvNeXtV2Block(nn.Module):
    kernel_size: int = 7

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        dilation: int = 1,
    ):
        super().__init__()
        assert self.kernel_size % 2 == 1
        # Symmetric padding preserves sequence length through the depthwise conv.
        padding = (dilation * (self.kernel_size - 1)) // 2
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=self.kernel_size, padding=padding, groups=dim, dilation=dilation
        )  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = ReplicatedLinear(dim, intermediate_dim, bias=True, return_bias=False, quant_config=None)
        self.act = nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = ReplicatedLinear(intermediate_dim, dim, bias=True, return_bias=False, quant_config=None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = x.transpose(1, 2)  # b n d -> b d n
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # b d n -> b n d
        x = self.norm(x)
        x = self.pwconv1(x)
        if isinstance(x, tuple):
            x = x[0]
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        if isinstance(x, tuple):
            x = x[0]
        return residual + x


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, theta_rescale_factor=1.0):
    # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
    # has some connection to NTK literature
    # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
    # https://github.com/lucidrains/rotary-embedding-torch/blob/main/rotary_embedding_torch/rotary_embedding_torch.py
    theta *= theta_rescale_factor ** (dim / (dim - 2))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return torch.cat([freqs_cos, freqs_sin], dim=-1)


class TextEmbedding(nn.Module):
    def __init__(self, vocab_size, dim, mask_padding=True, conv_layers=0, conv_mult=2):
        super().__init__()
        self.text_embed = nn.Embedding(vocab_size + 1, dim)  # use 0 as filler token

        self.mask_padding = mask_padding  # mask filler and batch padding tokens or not

        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = 8192  # 8192 is ~87.38s of 24khz audio; 4096 is ~43.69s of 24khz audio
            self.register_buffer("freqs_cis", precompute_freqs_cis(dim, self.precompute_max_pos), persistent=False)
            self.text_blocks = nn.Sequential(*[ConvNeXtV2Block(dim, dim * conv_mult) for _ in range(conv_layers)])
        else:
            self.extra_modeling = False

    def forward(self, text: torch.Tensor, audio_seq_len, drop_text_mask: torch.Tensor | None = None):
        """
        drop_text_mask: True for tokens to drop, False for tokens to keep
        """
        valid_pos_mask = None
        if torch.is_tensor(audio_seq_len):
            audio_seq_len = audio_seq_len.to(device=text.device, dtype=torch.long)
            max_seq_len = int(audio_seq_len.max().item())
        else:
            max_seq_len = int(audio_seq_len)

        text = text + 1
        text = text[:, :max_seq_len]
        text = F.pad(text, (0, max_seq_len - text.shape[1]), value=0)

        if torch.is_tensor(audio_seq_len):
            seq_pos = torch.arange(max_seq_len, device=text.device).unsqueeze(0)
            valid_pos_mask = seq_pos < audio_seq_len.unsqueeze(1)
            text = text.masked_fill(~valid_pos_mask, 0)

        text_mask = text == 0 if self.mask_padding else None

        if drop_text_mask is not None:
            text = torch.where(drop_text_mask.unsqueeze(1), torch.zeros_like(text), text)

        x = self.text_embed(text)
        if valid_pos_mask is not None:
            x = x.masked_fill(~valid_pos_mask.unsqueeze(-1), 0.0)

        if self.extra_modeling:
            freqs = self.freqs_cis[:max_seq_len, :]
            if valid_pos_mask is not None:
                freqs = freqs.unsqueeze(0) * valid_pos_mask.unsqueeze(-1).to(freqs.dtype)
            x = x + freqs

            if text_mask is not None:
                text_mask_exp = text_mask.unsqueeze(-1).expand(-1, -1, x.size(-1))
                x = x.masked_fill(text_mask_exp, 0.0)
                for block in self.text_blocks:
                    x = block(x)
                    x = x.masked_fill(text_mask_exp, 0.0)
            else:
                x = self.text_blocks(x)

        return x


class SinusPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TimestepEmbedding(nn.Module):
    def __init__(self, dim, freq_embed_dim=256):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        # Broken out from nn.Sequential for ReplicatedLinear compatibility.
        # Precision-sensitive (time conditioning) — quant_config=None.
        self.time_mlp_linear1 = ReplicatedLinear(
            freq_embed_dim, dim, bias=True, return_bias=False, quant_config=None,
        )
        self.time_mlp_silu = nn.SiLU()
        self.time_mlp_linear2 = ReplicatedLinear(
            dim, dim, bias=True, return_bias=False, quant_config=None,
        )

    def forward(self, timestep: torch.Tensor):
        time_hidden = self.time_embed(timestep)
        time_hidden = time_hidden.to(timestep.dtype)
        x = self.time_mlp_linear1(time_hidden)
        if isinstance(x, tuple):
            x = x[0]
        x = self.time_mlp_silu(x)
        x = self.time_mlp_linear2(x)
        if isinstance(x, tuple):
            x = x[0]
        return x


class InputEmbedding(nn.Module):
    def __init__(self, mel_dim, text_dim, out_dim, conv_pos_embed_groups=1):
        super().__init__()
        self.proj = ReplicatedLinear(
            mel_dim * 2 + text_dim, out_dim, bias=True, return_bias=False, quant_config=None,
        )
        self.conv_pos_embed = ConvPositionEmbedding(
            dim=out_dim,
            groups=conv_pos_embed_groups,
        )

    def forward(
        self,
        noisy_audio: torch.Tensor,
        cond_audio: torch.Tensor,
        text_embed: torch.Tensor,
        drop_audio_mask: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ):
        if drop_audio_mask is not None:
            cond_audio = torch.where(
                drop_audio_mask.unsqueeze(1).unsqueeze(1), torch.zeros_like(cond_audio), cond_audio
            )

        x = self.proj(torch.cat((noisy_audio, cond_audio, text_embed), dim=-1))
        if isinstance(x, tuple):
            x = x[0]
        x = self.conv_pos_embed(x, mask=mask) + x
        return x


def _compute_rope_freqs(
    seq_len: int,
    head_dim: int,
    theta: float = 10000.0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin for Rotary Embedding.

    Returns (cos, sin) each of shape [seq_len, head_dim // 2].
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)  # [seq_len, head_dim/2]
    return freqs.cos(), freqs.sin()


class F5RoPEPrepare(nn.Module):
    """Compute RoPE cos/sin for the current sequence length.

    Passes through (cos, sin) as a tuple -- compatible with
    shared ``RotaryEmbedding`` from ``vllm_omni.diffusion.layers.rope``.
    """

    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta
        self._cos: torch.Tensor | None = None
        self._sin: torch.Tensor | None = None
        self._cached_len: int = 0

    def forward(self, seq_len: int, device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self._cached_len or self._cos is None or (
            device is not None and self._cos is not None and self._cos.device != device
        ):
            self._cos, self._sin = _compute_rope_freqs(
                seq_len, self.head_dim, self.theta, device=device,
            )
            self._cached_len = seq_len
        return self._cos, self._sin


class ConvPositionEmbedding(nn.Module):
    def __init__(self, dim, kernel_size=31, groups=1):
        super().__init__()
        assert kernel_size % 2 != 0
        self.conv1d = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
        )
        self.layer_need_mask_idx = [i for i, layer in enumerate(self.conv1d) if isinstance(layer, nn.Conv1d)]

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        if mask is not None:
            mask = mask.unsqueeze(1)  # [B 1 N]
        x = x.permute(0, 2, 1)  # [B D N]

        if mask is not None:
            x = x.masked_fill(~mask, 0.0)
        for i, block in enumerate(self.conv1d):
            x = block(x)
            if mask is not None and i in self.layer_need_mask_idx:
                x = x.masked_fill(~mask, 0.0)

        x = x.permute(0, 2, 1)  # [B N D]

        return x


# modeling_qwen3_tts_tokenizer_v1.py also has this
# return with modulated x for attn input, and params for later mlp modulation
class AdaLayerNormZero(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = ReplicatedLinear(
            dim, dim * 6, bias=True, return_bias=False, quant_config=None,
        )

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, hidden_states, emb=None):
        emb = self.linear(self.silu(emb))
        if isinstance(emb, tuple):
            emb = emb[0]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(emb, 6, dim=1)

        hidden_states = self.norm(hidden_states) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp


# AdaLayerNormZero for final layer
# return only with modulated x for attn input, cuz no more mlp modulation
class AdaLayerNormZero_Final(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = ReplicatedLinear(
            dim, dim * 2, bias=True, return_bias=False, quant_config=None,
        )

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb):
        emb = self.linear(self.silu(emb))
        if isinstance(emb, tuple):
            emb = emb[0]
        scale, shift = torch.chunk(emb, 2, dim=1)

        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        dim_out=None,
        mult=4,
        dropout=0.0,
        approximate: str = "none",
        quant_config: QuantizationConfig | None = None,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        self.proj_in = ColumnParallelLinear(
            dim,
            inner_dim,
            bias=True,
            gather_output=False,
            return_bias=False,
            quant_config=quant_config,
        )
        self.activation = nn.GELU(approximate=approximate)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = RowParallelLinear(
            inner_dim,
            dim_out,
            bias=True,
            input_is_parallel=True,
            return_bias=False,
            quant_config=quant_config,
        )

    def forward(self, x):
        x = self.proj_in(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.proj_out(x)
        return x


class DiTBlock(nn.Module):
    """
    DiT block with self-attention and FFN, using AdaLayerNorm modulation.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        ff_mult: int = 4,
        dropout: float = 0.1,
        pe_attn_head: int | None = None,
        attn_mask_enabled: bool = True,
        quant_config: QuantizationConfig | None = None,
    ):
        super().__init__()

        self.attn_norm = AdaLayerNormZero(dim=dim)
        self.attn = F5TTSSelfAttention(
            dim=dim,
            num_attention_heads=heads,
            attention_head_dim=dim_head,
            dropout=0.0,
            pe_attn_head=pe_attn_head,
            attn_mask_enabled=attn_mask_enabled,
            quant_config=quant_config,
        )
        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh", quant_config=quant_config)

    def forward(
        self,
        noisy_audio: torch.Tensor,
        t: torch.Tensor,
        mask: torch.Tensor | None = None,
        rope: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        # pre-norm & modulation for attention input
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(noisy_audio, emb=t)

        # attention
        attn_output = self.attn(norm, rope, mask)

        # process attention output for input x
        noisy_audio = noisy_audio + gate_msa.unsqueeze(1) * attn_output

        norm = self.ff_norm(noisy_audio) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm)
        noisy_audio = noisy_audio + gate_mlp.unsqueeze(1) * ff_output

        return noisy_audio


class F5TTSDiTModel(nn.Module):
    """
    Optimized F5 TTS DiT model using vLLM layers.

    This is an optimized version of the diffusers F5TTSDiTModel that uses
    vLLM's efficient linear layers and attention implementations.

    Architecture:
    - Input: [B, in_channels, L] (e.g., [B, 64, L])
    - preprocess_conv: residual conv layer (keeps 64 channels)
    - proj_in: projects 64 -> 1536 (inner_dim)
    - Global+time embeddings prepended to sequence
    - Transformer blocks work on 1536-dim
    - proj_out: projects 1536 -> 64 (out_channels)
    - postprocess_conv: residual conv layer (keeps 64 channels)
    - Output: [B, out_channels, L]
    """

    _repeated_blocks = ["DiTBlock"]
    _layerwise_offload_blocks_attrs = ["transformer_blocks"]
    _hsdp_shard_conditions = [is_transformer_block_module]

    # Cache-DiT declaration: single sequential block list with hidden-only
    # I/O (Pattern_2). CFG runs as separate conditional/unconditional
    # forwards, so the two branches keep separate cache contexts.
    _cache_dit_adapter_config = CacheDiTAdapterConfig(
        block_forward_patterns={"transformer_blocks": ForwardPattern.Pattern_2},
        has_separate_cfg=True,
    )

    # SP plan — shared RoPE outputs (cos, sin) each of shape [T, D/2],
    # ready for SequenceParallel split on the T dimension.
    #   "rope_prepare": {0: SequenceParallelInput(split_dim=0, ...)},
    #   "transformer_blocks.0": {"hidden_states": SequenceParallelInput(split_dim=1, ...)},
    #   "proj_out": SequenceParallelOutput(gather_dim=1, expected_dims=3),

    def __init__(
        self,
        od_config: OmniDiffusionConfig | None = None,
        dim=1152,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        mel_dim=100,
        text_num_embeds=256,
        text_dim=None,
        text_mask_padding=True,
        pe_attn_head=None,
        conv_layers=0,
        attn_mask_enabled=True,
        long_skip_connection=False,
        checkpoint_activations=False,
        conv_pos_embed_groups=1,
        quant_config: QuantizationConfig | None = None,
    ):
        super().__init__()

        self.mel_dim = mel_dim

        # Store config for compatibility
        self.config = type(
            "Config",
            (),
            {
                "dim": dim,
                "depth": depth,
                "heads": heads,
                "dim_head": dim_head,
                "mel_dim": mel_dim,
                "text_num_embeds": text_num_embeds,
                "pe_attn_head": pe_attn_head,
            },
        )()

        self.time_embed = TimestepEmbedding(dim)
        if text_dim is None:
            text_dim = mel_dim
        self.text_embed = TextEmbedding(
            text_num_embeds,
            text_dim,
            mask_padding=text_mask_padding,
            conv_layers=conv_layers,
        )
        self.input_embed = InputEmbedding(
            mel_dim,
            text_dim,
            dim,
            conv_pos_embed_groups=conv_pos_embed_groups,
        )

        self.rope_prepare = F5RoPEPrepare(head_dim=dim_head)
        self._cached_rope_cos: torch.Tensor | None = None
        self._cached_rope_sin: torch.Tensor | None = None
        self._cached_rope_seq_len: int | None = None

        self.dim = dim
        self.depth = depth

        self.transformer_blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    ff_mult=ff_mult,
                    dropout=dropout,
                    pe_attn_head=pe_attn_head,
                    attn_mask_enabled=attn_mask_enabled,
                    quant_config=quant_config,
                )
                for _ in range(depth)
            ]
        )
        self.long_skip_connection = (
            ReplicatedLinear(dim * 2, dim, bias=False, return_bias=False, quant_config=quant_config)
            if long_skip_connection else None
        )

        self.norm_out = AdaLayerNormZero_Final(dim)  # final modulation
        self.proj_out = ReplicatedLinear(
            dim, mel_dim, bias=True, return_bias=False, quant_config=None,
        )

    @property
    def dtype(self) -> torch.dtype:
        """Return the dtype of the model parameters."""
        return next(self.parameters()).dtype

    def forward(
        self,
        noisy_audio: torch.Tensor,
        cond_audio: torch.Tensor,
        cond_text: torch.Tensor,
        timestep: torch.Tensor,
        mask: torch.Tensor | None = None,
        drop_audio_mask: torch.Tensor | None = None,
        drop_text_mask: torch.Tensor | None = None,
        rotary_embedding: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        return self._forward_impl(
            noisy_audio,
            cond_audio,
            cond_text,
            timestep,
            mask,
            drop_audio_mask,
            drop_text_mask,
            rotary_embedding,
        )

    def _forward_impl(
        self,
        noisy_audio: torch.Tensor,
        cond_audio: torch.Tensor,
        cond_text: torch.Tensor,
        timestep: torch.Tensor,
        mask: torch.Tensor | None = None,
        drop_audio_mask: torch.Tensor | None = None,
        drop_text_mask: torch.Tensor | None = None,
        rotary_embedding: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        batch, seq_len = noisy_audio.shape[0], noisy_audio.shape[1]
        if timestep.ndim == 0:
            timestep = timestep.repeat(batch)

        # t: conditioning time, text: text, x: noised audio + cond audio + text
        time_embedding = self.time_embed(timestep)
        text_seq_len: int | torch.Tensor = mask.sum(dim=1) if mask is not None else seq_len
        text_embed = self.text_embed(
            cond_text,
            text_seq_len,
            drop_text_mask=drop_text_mask,
        )
        noisy_audio = self.input_embed(
            noisy_audio,
            cond_audio,
            text_embed,
            drop_audio_mask=drop_audio_mask,
            mask=mask,
        )

        if self._cached_rope_seq_len == seq_len and self._cached_rope_cos is not None:
            rotary_embedding = (self._cached_rope_cos, self._cached_rope_sin)
        else:
            cos, sin = self.rope_prepare(seq_len, device=noisy_audio.device)
            self._cached_rope_cos = cos
            self._cached_rope_sin = sin
            self._cached_rope_seq_len = seq_len
            rotary_embedding = (cos, sin)

        if self.long_skip_connection is not None:
            residual = noisy_audio

        for block in self.transformer_blocks:
            noisy_audio = block(noisy_audio, time_embedding, mask=mask, rope=rotary_embedding)

        if self.long_skip_connection is not None:
            noisy_audio = self.long_skip_connection(torch.cat((noisy_audio, residual), dim=-1))
            if isinstance(noisy_audio, tuple):
                noisy_audio = noisy_audio[0]

        noisy_audio = self.norm_out(noisy_audio, time_embedding)
        output = self.proj_out(noisy_audio)
        if isinstance(output, tuple):
            output = output[0]

        return output

    packed_modules_mapping = {
        "to_qkv": ["to_q", "to_k", "to_v"],
    }

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            (".to_qkv", ".to_q", "q"),
            (".to_qkv", ".to_k", "k"),
            (".to_qkv", ".to_v", "v"),
        ]

        # Name mapping: upstream checkpoint uses nn.Sequential indices,
        # our version uses named submodules.
        name_mapping = {
            ".ff.ff.0.0.": ".ff.proj_in.",
            ".ff.ff.2.": ".ff.proj_out.",
            "time_embed.time_mlp.0.": "time_embed.time_mlp_linear1.",
            "time_embed.time_mlp.2.": "time_embed.time_mlp_linear2.",
        }

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            original_name = name

            # Map upstream sequential indices to named submodules
            for old_pattern, new_pattern in name_mapping.items():
                if old_pattern in name:
                    name = name.replace(old_pattern, new_pattern)
                    break

            # Handle QKV stacking
            lookup_name = name
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                lookup_name = name.replace(weight_name, param_name)
                if lookup_name in params_dict:
                    param = params_dict[lookup_name]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id)
                    loaded_params.add(lookup_name)
                break
            else:
                # Handle RowParallelLinear to_out naming
                if lookup_name not in params_dict and ".to_out.0." in lookup_name:
                    lookup_name = lookup_name.replace(".to_out.0.", ".to_out.")
                if lookup_name in params_dict:
                    param = params_dict[lookup_name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
                    loaded_params.add(lookup_name)
                else:
                    logger.debug("Skipping weight %s - not found in model", original_name)

        return loaded_params
