# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
F5 TTS DiT Model for vLLM-Omni.

F5TTS-style DiT transformer for flow-matching based TTS.
"""

import math
from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from x_transformers.x_transformers import (
    RotaryEmbedding as XTRotaryEmbedding,
)
from x_transformers.x_transformers import (
    apply_rotary_pos_emb as xt_apply_rotary_pos_emb,
)

from vllm_omni.diffusion.data import OmniDiffusionConfig

logger = init_logger(__name__)


class F5TTSSelfAttention(nn.Module):
    """
    Optimized self-attention for F5 TTS using vLLM layers.

    Self-attention uses full attention (all heads for Q, K, V).
    GQA is only used for cross-attention.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout: float = 0.0,
        pe_attn_head: int | None = None,
        attn_mask_enabled: bool = True,
    ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_attention_heads
        self.head_dim = attention_head_dim
        self.inner_dim = num_attention_heads * attention_head_dim
        self.pe_attn_head = pe_attn_head
        self.attn_mask_enabled = attn_mask_enabled

        # All projections use the full inner_dim with checkpoint-compatible bias terms.
        self.to_q = ReplicatedLinear(dim, self.inner_dim, bias=True)
        self.to_k = ReplicatedLinear(dim, self.inner_dim, bias=True)
        self.to_v = ReplicatedLinear(dim, self.inner_dim, bias=True)

        # Output projection
        self.to_out = nn.ModuleList(
            [
                ReplicatedLinear(self.inner_dim, dim, bias=True),
                nn.Dropout(dropout),
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_embedding: tuple[torch.Tensor, torch.Tensor | None],
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # Projections - all output inner_dim
        query, _ = self.to_q(hidden_states)
        key, _ = self.to_k(hidden_states)
        value, _ = self.to_v(hidden_states)

        # Match the F5-TTS attention layout and masking behavior.
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings
        freqs, xpos_scale = rotary_embedding
        q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)
        if self.pe_attn_head is None:
            query = xt_apply_rotary_pos_emb(query, freqs, q_xpos_scale)
            key = xt_apply_rotary_pos_emb(key, freqs, k_xpos_scale)
        else:
            rotary_head_count = min(self.pe_attn_head, self.num_heads)
            query[:, :rotary_head_count] = xt_apply_rotary_pos_emb(
                query[:, :rotary_head_count],
                freqs,
                q_xpos_scale,
            )
            key[:, :rotary_head_count] = xt_apply_rotary_pos_emb(
                key[:, :rotary_head_count],
                freqs,
                k_xpos_scale,
            )

        if self.attn_mask_enabled and attention_mask is not None:
            attn_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            attn_mask = attn_mask.expand(batch_size, self.num_heads, seq_len, seq_len)
        else:
            attn_mask = None

        hidden_states = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, seq_len, self.inner_dim)

        # Output projection
        hidden_states, _ = self.to_out[0](hidden_states)

        if attention_mask is not None:
            hidden_states = hidden_states.masked_fill(~attention_mask.unsqueeze(-1), 0.0)

        return hidden_states


class SwiGLU(nn.Module):
    """SwiGLU activation - matches diffusers structure."""

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2, bias=bias)
        self.activation = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(hidden_states)
        hidden_states, gate = hidden_states.chunk(2, dim=-1)
        return hidden_states * self.activation(gate)


class F5TTSFeedForward(nn.Module):
    """
    Feed-forward network with SwiGLU activation for F5 TTS.
    Matches diffusers FeedForward structure with activation_fn="swiglu".
    """

    def __init__(self, dim: int, inner_dim: int, bias: bool = True):
        super().__init__()
        # Structure matches diffusers FeedForward:
        # net.0 = SwiGLU (proj.weight, proj.bias)
        # net.1 = Dropout
        # net.2 = Linear (weight, bias)
        self.net = nn.Sequential(
            SwiGLU(dim, inner_dim, bias=bias),
            nn.Dropout(0.0),
            nn.Linear(inner_dim, dim, bias=bias),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_states)


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
        self.pwconv1 = nn.Linear(dim, intermediate_dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = x.transpose(1, 2)  # b n d -> b d n
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # b d n -> b n d
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
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
        self.time_mlp = nn.Sequential(nn.Linear(freq_embed_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, timestep: torch.Tensor):
        time_hidden = self.time_embed(timestep)
        time_hidden = time_hidden.to(timestep.dtype)
        time = self.time_mlp(time_hidden)  # b d
        return time


class InputEmbedding(nn.Module):
    def __init__(self, mel_dim, text_dim, out_dim, conv_pos_embed_groups=1):
        super().__init__()
        self.proj = nn.Linear(mel_dim * 2 + text_dim, out_dim)
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
        x = self.conv_pos_embed(x, mask=mask) + x
        return x


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
        self.linear = nn.Linear(dim, dim * 6)

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, hidden_states, emb=None):
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(emb, 6, dim=1)

        hidden_states = self.norm(hidden_states) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp


# AdaLayerNormZero for final layer
# return only with modulated x for attn input, cuz no more mlp modulation
class AdaLayerNormZero_Final(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb):
        emb = self.linear(self.silu(emb))
        scale, shift = torch.chunk(emb, 2, dim=1)

        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, dropout=0.0, approximate: str = "none"):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        activation = nn.GELU(approximate=approximate)
        project_in = nn.Sequential(nn.Linear(dim, inner_dim), activation)
        self.ff = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))

    def forward(self, x):
        return self.ff(x)


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
        )
        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh")

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

        self.rotary_embed = XTRotaryEmbedding(dim_head)

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
                )
                for _ in range(depth)
            ]
        )
        self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None

        self.norm_out = AdaLayerNormZero_Final(dim)  # final modulation
        self.proj_out = nn.Linear(dim, mel_dim)

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

        rotary_embedding = self.rotary_embed.forward_from_seq_len(seq_len)

        if self.long_skip_connection is not None:
            residual = noisy_audio

        for block in self.transformer_blocks:
            noisy_audio = block(noisy_audio, time_embedding, mask=mask, rope=rotary_embedding)

        if self.long_skip_connection is not None:
            noisy_audio = self.long_skip_connection(torch.cat((noisy_audio, residual), dim=-1))

        noisy_audio = self.norm_out(noisy_audio, time_embedding)
        output = self.proj_out(noisy_audio)

        return output

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """
        Load weights from a pretrained model.

        Maps diffusers weight names to our module structure.

        Returns:
            Set of parameter names that were successfully loaded.
        """
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        # Weight name mapping from diffusers to our implementation
        name_mapping = {
            # Timestep projection - diffusers uses index-based naming
            "timestep_proj.linear_1.weight": "timestep_proj.0.weight",
            "timestep_proj.linear_1.bias": "timestep_proj.0.bias",
            "timestep_proj.linear_2.weight": "timestep_proj.2.weight",
            "timestep_proj.linear_2.bias": "timestep_proj.2.bias",
            # Global projection - diffusers uses index-based naming
            "global_proj.linear_1.weight": "global_proj.0.weight",
            "global_proj.linear_2.weight": "global_proj.2.weight",
        }

        for name, loaded_weight in weights:
            # Apply name mapping if needed
            mapped_name = name_mapping.get(name, name)

            if mapped_name in params_dict:
                param = params_dict[mapped_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(mapped_name)
            else:
                logger.debug(f"Skipping weight {name} - not found in model")

        return loaded_params
