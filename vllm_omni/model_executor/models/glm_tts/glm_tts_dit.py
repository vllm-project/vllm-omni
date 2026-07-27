# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-TTS DiT (Diffusion Transformer) Model.

Flow Matching model that converts speech tokens to mel-spectrogram.
Ported from: https://github.com/zai-org/GLM-TTS/blob/main/flow/dit.py
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from x_transformers.x_transformers import RotaryEmbedding, apply_rotary_pos_emb

from vllm_omni.diffusion.attention.layer import Attention as DiffusionAttention


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, theta_rescale_factor: float = 1.0):
    """Precompute Rotary Positional Embeddings (RoPE)."""
    theta *= theta_rescale_factor ** (dim / (dim - 2))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return torch.cat([freqs_cos, freqs_sin], dim=-1)


def get_pos_embed_indices(start, length, max_pos, scale=1.0):
    """Get position embedding indices."""
    scale = scale * torch.ones_like(start, dtype=torch.float32)
    pos = (
        start.unsqueeze(1)
        + (torch.arange(length, device=start.device, dtype=torch.float32).unsqueeze(0) * scale.unsqueeze(1)).long()
    )
    pos = torch.where(pos < max_pos, pos, max_pos - 1)
    return pos


class SinusPositionEmbedding(nn.Module):
    """Sinusoidal position embedding for timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor, scale: float = 1000) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TimestepEmbedding(nn.Module):
    """Timestep embedding with MLP."""

    def __init__(self, dim: int, freq_embed_dim: int = 256):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.Sequential(nn.Linear(freq_embed_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        time_hidden = self.time_embed(timestep)
        # Cast to model weight dtype (bfloat16) not input dtype (may be float32)
        weight_dtype = self.time_mlp[0].weight.dtype
        time_hidden = time_hidden.to(weight_dtype)
        return self.time_mlp(time_hidden)


class GRN(nn.Module):
    """Global Response Normalization layer."""

    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=1, keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


class ConvNeXtV2Block(nn.Module):
    """ConvNeXt-V2 Block for text embedding."""

    def __init__(self, dim: int, intermediate_dim: int, dilation: int = 1):
        super().__init__()
        padding = (dilation * (7 - 1)) // 2
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=padding, groups=dim, dilation=dilation)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = x.transpose(1, 2)
        x = self.dwconv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        return residual + x


class ConvPositionEmbedding(nn.Module):
    """Convolutional position embedding."""

    def __init__(self, dim: int, kernel_size: int = 31, groups: int = 16):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")
        self.conv1d = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is not None:
            mask = mask[..., None]
            x = x.masked_fill(~mask, 0.0)

        x = rearrange(x, "b n d -> b d n")
        x = self.conv1d(x)
        out = rearrange(x, "b d n -> b n d")

        if mask is not None:
            out = out.masked_fill(~mask, 0.0)
        return out


class TextEmbedding(nn.Module):
    """Text/speech token embedding with optional ConvNeXt modeling."""

    def __init__(
        self,
        text_num_embeds: int,
        output_dim: int,
        conv_layers: int = 0,
        conv_mult: int = 2,
        length_align: str = "fill",
    ):
        super().__init__()
        self.pad_id = int(text_num_embeds)
        self.text_embed = nn.Embedding(text_num_embeds + 1, output_dim, padding_idx=self.pad_id)
        self.length_align = length_align

        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = 4096
            self.register_buffer(
                "freqs_cis", precompute_freqs_cis(output_dim, self.precompute_max_pos), persistent=False
            )
            self.text_blocks = nn.Sequential(
                *[ConvNeXtV2Block(output_dim, output_dim * conv_mult) for _ in range(conv_layers)]
            )
        else:
            self.extra_modeling = False

    def forward(
        self,
        text: torch.Tensor,
        aim_seq_len: int,
        text_lens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, text_len = text.shape[0], text.shape[1]
        if self.length_align == "fill":
            text = text[:, :aim_seq_len]
            text = F.pad(text, (0, aim_seq_len - min(text_len, aim_seq_len)), value=self.pad_id)
        elif self.length_align == "interpolate_token":
            if text_lens is None:
                text = (
                    F.interpolate(
                        text.unsqueeze(1).float(),
                        size=aim_seq_len,
                        mode="nearest",
                    )
                    .squeeze(1)
                    .long()
                )
            else:
                text_lens = text_lens.to(device=text.device, dtype=torch.long)
                text_lens = torch.clamp(text_lens, min=1, max=text_len)
                positions = torch.arange(aim_seq_len, device=text.device, dtype=torch.long)
                src_idx = positions.unsqueeze(0) * text_lens.unsqueeze(1) // int(aim_seq_len)
                src_idx = torch.minimum(src_idx, (text_lens - 1).unsqueeze(1))
                text = torch.gather(text, dim=1, index=src_idx)
        elif self.length_align != "interpolate_feature":
            raise ValueError(f"Unsupported length_align={self.length_align!r}")

        hidden = self.text_embed(text)

        if self.length_align == "interpolate_feature":
            hidden = F.interpolate(
                hidden.permute(0, 2, 1),
                size=aim_seq_len,
                mode="nearest",
            ).permute(0, 2, 1)

        if self.extra_modeling:
            # Get model dtype from text_blocks weights
            model_dtype = self.text_blocks[0].dwconv.weight.dtype
            hidden = hidden.to(model_dtype)

            batch_start = torch.zeros((batch,), dtype=torch.long, device=text.device)
            pos_idx = get_pos_embed_indices(batch_start, aim_seq_len, max_pos=self.precompute_max_pos)
            text_pos_embed = self.freqs_cis[pos_idx].to(model_dtype)
            hidden = hidden + text_pos_embed
            hidden = self.text_blocks(hidden)

        return hidden


class EmbeddingConcater(nn.Module):
    """Concatenates and projects noisy audio, condition, and text embeddings."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        text_embed: torch.Tensor,
        drop_audio_cond: bool = False,
    ) -> torch.Tensor:
        if drop_audio_cond:
            cond = torch.zeros_like(cond)
        x = self.proj(torch.cat((x, cond, text_embed), dim=-1))
        x = self.conv_pos_embed(x) + x
        return x


class AdaLayerNormZero(nn.Module):
    """Adaptive LayerNorm with modulation from time/condition embedding."""

    def __init__(self, dim: int, additional_dim: int = 0):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim + additional_dim, dim * 6)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x: torch.Tensor, emb: torch.Tensor):
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(emb, 6, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormZeroFinal(nn.Module):
    """Adaptive LayerNorm for final layer."""

    def __init__(self, dim: int, additional_dim: int = 0):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim + additional_dim, dim * 2)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        emb = self.linear(self.silu(emb))
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation.

    Matches original GLM-TTS structure with nested Sequential for checkpoint compatibility.
    Weight paths: ff.ff.0.0.* (first Linear), ff.ff.2.* (second Linear)
    """

    def __init__(self, dim: int, dim_out: int | None = None, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        # Nested structure to match original checkpoint paths
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(approximate="tanh"),
        )
        self.ff = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff(x)


class DiTAttention(nn.Module):
    """Attention module using diffusion infrastructure."""

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = dim_head * heads
        self.scale = 1.0 / math.sqrt(dim_head)

        self.to_q = nn.Linear(dim, self.inner_dim)
        self.to_k = nn.Linear(dim, self.inner_dim)
        self.to_v = nn.Linear(dim, self.inner_dim)
        self.to_out = nn.Sequential(nn.Linear(self.inner_dim, dim), nn.Dropout(dropout))

        self.attn = DiffusionAttention(
            num_heads=heads,
            head_size=dim_head,
            softmax_scale=self.scale,
            causal=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        rope=None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len = x.shape[0], x.shape[1]

        query = self.to_q(x)
        key = self.to_k(x)
        value = self.to_v(x)

        # Apply RoPE
        if rope is not None:
            freqs, xpos_scale = rope
            q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)
            query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
            key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)

        # Reshape for attention
        query = query.view(batch_size, seq_len, self.heads, self.dim_head)
        key = key.view(batch_size, seq_len, self.heads, self.dim_head)
        value = value.view(batch_size, seq_len, self.heads, self.dim_head)

        # Block-causal mask requires arbitrary 2D attention mask which
        # FlashAttention does not support.  Fall back to PyTorch SDPA
        # (matches official GLM-TTS flow/modules.py:272).
        if attn_mask is not None:
            q = query.permute(0, 2, 1, 3)  # [B, H, T, D]
            k = key.permute(0, 2, 1, 3)
            v = value.permute(0, 2, 1, 3)
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=False,
            )
            out = out.permute(0, 2, 1, 3)  # [B, T, H, D]
        else:
            out = self.attn(query, key, value, attn_metadata=None)
        out = out.view(batch_size, seq_len, self.inner_dim)
        out = out.to(query.dtype)
        out = self.to_out(out)

        # Apply padding mask
        if padding_mask is not None:
            padding_mask = rearrange(padding_mask, "b 1 n -> b n 1").bool()
            out = out.masked_fill(~padding_mask, 0.0)

        return out


class DiTBlock(nn.Module):
    """DiT block with AdaLayerNorm modulation."""

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        ff_mult: int = 4,
        dropout: float = 0.1,
        additional_condition_dim: int = 0,
    ):
        super().__init__()
        self.attn_norm = AdaLayerNormZero(dim, additional_condition_dim)
        self.attn = DiTAttention(dim=dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        rope=None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Pre-norm & modulation for attention
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)

        # Attention
        attn_output = self.attn(x=norm, padding_mask=padding_mask, rope=rope, attn_mask=attn_mask)
        x = x + gate_msa.unsqueeze(1) * attn_output

        # FeedForward
        norm = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm)
        x = x + gate_mlp.unsqueeze(1) * ff_output

        return x


class GLMTTSDiT(nn.Module):
    """GLM-TTS Diffusion Transformer.

    Converts speech tokens to mel-spectrogram via flow matching.
    """

    _repeated_blocks = ["DiTBlock"]
    _layerwise_offload_blocks_attrs = ["transformer_blocks"]

    def __init__(
        self,
        *,
        trans_dim: int = 768,
        depth: int = 18,
        heads: int = 12,
        dim_head: int = 64,
        ff_mult: int = 2,
        dropout: float = 0.1,
        mel_dim: int = 80,
        text_vocab_size: int = 100000,
        text_emb_dim: int = 512,
        conv_layers: int = 4,
        condition_dim: int = 160,  # mel_dim + spk_embed_dim
        spkr_emb_adaln: bool = False,
        spkr_dim: int = 192,
        use_wavlm_emb: bool = False,
        long_skip_connection: bool = False,
    ):
        super().__init__()

        self.time_embed = TimestepEmbedding(trans_dim)
        self.text_emb_layer = TextEmbedding(
            text_vocab_size,
            text_emb_dim,
            conv_layers=conv_layers,
            length_align="interpolate_token",
        )

        in_dim = mel_dim + text_emb_dim + condition_dim
        self.emb_concator = EmbeddingConcater(in_dim, trans_dim)

        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = trans_dim
        self.depth = depth
        self.num_heads = heads
        self.spkr_emb_adaln = spkr_emb_adaln

        # Speaker embedding dimension for AdaLN
        if spkr_emb_adaln:
            additional_dim = spkr_dim + 256 if use_wavlm_emb else spkr_dim
        else:
            additional_dim = 0

        self.transformer_blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=trans_dim,
                    heads=heads,
                    dim_head=dim_head,
                    ff_mult=ff_mult,
                    dropout=dropout,
                    additional_condition_dim=additional_dim,
                )
                for _ in range(depth)
            ]
        )

        self.long_skip_connection = nn.Linear(trans_dim * 2, trans_dim, bias=False) if long_skip_connection else None

        self.norm_out = AdaLayerNormZeroFinal(trans_dim, additional_dim=additional_dim)
        self.proj_out = nn.Linear(trans_dim, mel_dim)

    # ------------------------------------------------------------------
    # Block-causal attention mask (ported from GLM-TTS block_mask_util)
    # ------------------------------------------------------------------

    def _token_to_mel_size(self, token_count: int) -> int:
        """Convert token block size to mel-frame block size for streaming mask.

        Matches upstream GLM-TTS ``flow/dit.py`` block-mask conversion.
        This is intentionally separate from normal ``feat_len`` calculation,
        which still follows ``input_frame_rate`` and ``mel_framerate``.
        """
        return int(token_count / 12.5 * 22050 / 256)

    def _create_block_causal_mask(
        self,
        block_pattern: list[int],
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build block-causal attention mask [1, 1, T, T].

        Ported from GLM-TTS utils/block_mask_util.py ``create()``.
        Each position can attend to its own block plus a look-ahead into
        the next block (50% for small blocks, 66% for large blocks).

        Vectorized via ``torch.searchsorted`` — no Python loop over T.
        """
        import itertools

        mel_blocks = [self._token_to_mel_size(b) for b in block_pattern]
        # Extend with copies of the last block (enough for any seq_len)
        extended = mel_blocks + [mel_blocks[-1]] * ((seq_len // max(mel_blocks[-1], 1)) + 2)
        accum = list(itertools.accumulate(extended))

        # Vectorized: map each position to its block index via searchsorted
        accum_t = torch.tensor(accum, dtype=torch.long)
        extended_t = torch.tensor(extended, dtype=torch.long)
        positions = torch.arange(seq_len, dtype=torch.long)

        # searchsorted(accum, positions, right=True) gives the block index
        # for each position (accum is strictly increasing).
        block_idx = torch.searchsorted(accum_t, positions, right=True)

        # Per-position: block start, block size, position within block
        block_start = torch.where(block_idx > 0, accum_t[block_idx - 1], torch.zeros_like(block_idx))
        cur_block_size = extended_t[block_idx]
        idx_in_block = positions - block_start

        # Look-ahead: positions near the end of a block can see into the next
        look_future = cur_block_size - 1 - idx_in_block
        min_future = torch.where(
            cur_block_size <= 172,
            cur_block_size // 2,
            (cur_block_size.float() / 1.5).long(),
        )
        delta = torch.clamp(min_future - look_future, min=0)
        aim_len = torch.clamp(block_start + cur_block_size + delta, max=seq_len)

        # Build mask: position i can attend to columns [0, aim_len[i])
        col_idx = torch.arange(seq_len, device=device)
        mask = col_idx.unsqueeze(0) < aim_len.unsqueeze(1).to(device)

        # [1, 1, T, T] for broadcast across batch and heads
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        middle_point: torch.Tensor,
        condition: torch.Tensor,
        text: torch.Tensor,
        time_step: torch.Tensor,
        padding_mask: torch.Tensor,
        spkr_emb: torch.Tensor | None = None,
        text_lens: torch.Tensor | None = None,
        is_causal: bool = False,
        attn_mask: torch.Tensor | None = None,
        block_pattern: list[int] | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            middle_point: Noised audio tensor [B, T, mel_dim]
            condition: Masked audio condition [B, T, condition_dim]
            text: Speech token tensor [B, T_text]
            time_step: Timestep tensor [B]
            padding_mask: Padding mask [B, T]
            spkr_emb: Speaker embedding [B, spkr_dim]
            is_causal: Use block-causal attention
            attn_mask: Custom attention mask (overrides is_causal)
            block_pattern: Token-level block sizes for block-causal mask

        Returns:
            Predicted mel-spectrogram [B, T, mel_dim]
        """
        # Cast floating inputs to model dtype (weights may be bfloat16)
        model_dtype = self.proj_out.weight.dtype
        middle_point = middle_point.to(model_dtype)
        condition = condition.to(model_dtype)
        time_step = time_step.to(model_dtype)
        if spkr_emb is not None:
            spkr_emb = spkr_emb.to(model_dtype)

        _, seq_len = middle_point.shape[0], middle_point.shape[1]

        # Time embedding
        time_emb = self.time_embed(time_step)

        # Concatenate speaker embedding if using AdaLN
        if self.spkr_emb_adaln and spkr_emb is not None:
            time_emb = torch.cat([time_emb, spkr_emb], dim=-1)

        # Text embedding - ensure cast to model dtype
        text_embed = self.text_emb_layer(text, seq_len, text_lens=text_lens).to(model_dtype)

        # Input projection
        x = self.emb_concator(middle_point, condition, text_embed, drop_audio_cond=False)

        # Rotary embeddings
        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        # Build block-causal mask when is_causal and no explicit attn_mask
        if attn_mask is None and is_causal and block_pattern is not None:
            attn_mask = self._create_block_causal_mask(
                block_pattern,
                seq_len,
                middle_point.device,
                model_dtype,
            )

        # Long skip connection
        if self.long_skip_connection is not None:
            residual = x

        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x, time_emb, padding_mask=padding_mask.unsqueeze(1), rope=rope, attn_mask=attn_mask)

        # Long skip connection
        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        # Output projection
        x = self.norm_out(x, time_emb)
        output = self.proj_out(x)

        return output
