# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from LongCat-AudioDiT (https://github.com/meituan-longcat/LongCat-AudioDiT)

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention as DiffusionAttention


class AudioDiTRMSNorm(nn.Module):
    """RMS Normalization for QK."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class AudioDiTSinusPositionEmbedding(nn.Module):
    """Sinusoidal position embedding."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor, scale: float = 1000.0) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class AudioDiTTimestepEmbedding(nn.Module):
    """Timestep embedding with MLP."""

    def __init__(self, dim: int, freq_embed_dim: int = 256):
        super().__init__()
        self.time_embed = AudioDiTSinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(freq_embed_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        time_hidden = self.time_embed(timestep)
        time_hidden = time_hidden.to(timestep.dtype)
        return self.time_mlp(time_hidden)


class AudioDiTRotaryEmbedding(nn.Module):
    """Rotary position embedding."""

    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 100000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self._cos: torch.Tensor | None = None
        self._sin: torch.Tensor | None = None
        self._cached_len: int = 0
        self._cached_device: torch.device | None = None

    def _build(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        t = torch.arange(seq_len, dtype=torch.int64).type_as(inv_freq)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self._cos = emb.cos().to(dtype=dtype, device=device)
        self._sin = emb.sin().to(dtype=dtype, device=device)
        self._cached_len = seq_len
        self._cached_device = device

    def forward(self, x: torch.Tensor, seq_len: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len is None:
            seq_len = x.shape[1]
        if self._cos is None or seq_len > self._cached_len or self._cached_device != x.device:
            self._build(max(seq_len, self.max_position_embeddings), x.device, x.dtype)
        return (
            self._cos[:seq_len].to(dtype=x.dtype),
            self._sin[:seq_len].to(dtype=x.dtype),
        )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _apply_rotary_emb(x: torch.Tensor, freqs_cis: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Apply rotary embeddings.

    Args:
        x: Input tensor of shape [B, H, S, D]
        freqs_cis: (cos, sin) of shape [S, D]
    """
    cos, sin = freqs_cis
    cos = cos[None, None].to(x.device)
    sin = sin[None, None].to(x.device)
    return (x.float() * cos + _rotate_half(x).float() * sin).to(x.dtype)


class AudioDiTGRN(nn.Module):
    """Global Response Normalization."""

    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=1, keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


class AudioDiTConvNeXtV2Block(nn.Module):
    """ConvNeXt-V2 Block for text processing."""

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        dilation: int = 1,
        kernel_size: int = 7,
        bias: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        padding = (dilation * (kernel_size - 1)) // 2
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=kernel_size, padding=padding, groups=dim, dilation=dilation, bias=bias
        )
        self.norm = nn.LayerNorm(dim, eps=eps)
        self.pwconv1 = nn.Linear(dim, intermediate_dim, bias=bias)
        self.act = nn.SiLU()
        self.grn = AudioDiTGRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim, bias=bias)

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


class AudioDiTEmbedder(nn.Module):
    """Input embedding layer."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor, mask: torch.BoolTensor | None = None) -> torch.Tensor:
        if mask is not None:
            x = x.masked_fill(mask.logical_not().unsqueeze(-1), 0.0)
        x = self.proj(x)
        if mask is not None:
            x = x.masked_fill(mask.logical_not().unsqueeze(-1), 0.0)
        return x


class AudioDiTAdaLNMLP(nn.Module):
    """AdaLN MLP for conditioning."""

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(in_dim, out_dim, bias=bias))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class AudioDiTAdaLayerNormZeroFinal(nn.Module):
    """AdaLayerNormZero for final layer."""

    def __init__(self, dim: int, bias: bool = True, eps: float = 1e-6):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2, bias=bias)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        emb = self.linear(self.silu(emb))
        scale, shift = torch.chunk(emb, 2, dim=-1)
        x = self.norm(x.float()).type_as(x)
        if scale.ndim == 2:
            x = x * (1 + scale)[:, None, :] + shift[:, None, :]
        else:
            x = x * (1 + scale) + shift
        return x


def _modulate(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """LayerNorm without affine + modulate."""
    x = F.layer_norm(x.float(), (x.shape[-1],), eps=eps).type_as(x)
    if scale.ndim == 2:
        return x * (1 + scale[:, None]) + shift[:, None]
    return x * (1 + scale) + shift


class AudioDiTSelfAttention(nn.Module):
    """Self-attention using vllm-omni diffusion attention backend with FlashAttention."""

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        dropout: float = 0.0,
        bias: bool = True,
        qk_norm: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.heads = heads
        self.inner_dim = dim_head * heads
        self.to_q = nn.Linear(dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(dim, self.inner_dim, bias=bias)
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = AudioDiTRMSNorm(self.inner_dim, eps=eps)
            self.k_norm = AudioDiTRMSNorm(self.inner_dim, eps=eps)
        self.to_out = nn.ModuleList([nn.Linear(self.inner_dim, dim, bias=bias), nn.Dropout(dropout)])

        self.attn = DiffusionAttention(
            num_heads=heads,
            head_size=dim_head,
            softmax_scale=1.0 / math.sqrt(dim_head),
            causal=False,
            num_kv_heads=heads,
        )

    def forward(self, x: torch.Tensor, mask: torch.BoolTensor | None = None, rope: tuple | None = None) -> torch.Tensor:
        batch_size = x.shape[0]
        query = self.to_q(x)
        key = self.to_k(x)
        value = self.to_v(x)

        if self.qk_norm:
            query = self.q_norm(query)
            key = self.k_norm(key)

        head_dim = self.inner_dim // self.heads
        # FlashAttention needs (batch, seq, heads, head_dim)
        query = query.view(batch_size, -1, self.heads, head_dim)
        key = key.view(batch_size, -1, self.heads, head_dim)
        value = value.view(batch_size, -1, self.heads, head_dim)

        if rope is not None:
            # Rotary embedding needs (batch, heads, seq, head_dim)
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            query = _apply_rotary_emb(query, rope)
            key = _apply_rotary_emb(key, rope)
            query = query.transpose(1, 2).contiguous()
            key = key.transpose(1, 2).contiguous()

        attn_metadata = AttentionMetadata(attn_mask=mask) if mask is not None else None
        out = self.attn(query, key, value, attn_metadata=attn_metadata)

        out = out.reshape(batch_size, -1, self.inner_dim).to(query.dtype)
        out = self.to_out[0](out)
        out = self.to_out[1](out)
        return out


class AudioDiTCrossAttention(nn.Module):
    """Cross-attention using vllm-omni diffusion attention backend."""

    def __init__(
        self,
        q_dim: int,
        kv_dim: int,
        heads: int,
        dim_head: int,
        dropout: float = 0.0,
        bias: bool = True,
        qk_norm: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.heads = heads
        self.inner_dim = dim_head * heads
        self.to_q = nn.Linear(q_dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(kv_dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(kv_dim, self.inner_dim, bias=bias)
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = AudioDiTRMSNorm(self.inner_dim, eps=eps)
            self.k_norm = AudioDiTRMSNorm(self.inner_dim, eps=eps)
        self.to_out = nn.ModuleList([nn.Linear(self.inner_dim, q_dim, bias=bias), nn.Dropout(dropout)])

        self.attn = DiffusionAttention(
            num_heads=heads,
            head_size=dim_head,
            softmax_scale=1.0 / math.sqrt(dim_head),
            causal=False,
            num_kv_heads=heads,
        )

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        mask: torch.BoolTensor | None = None,
        cond_mask: torch.BoolTensor | None = None,
        rope: tuple | None = None,
        cond_rope: tuple | None = None,
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        query = self.to_q(x)
        key = self.to_k(cond)
        value = self.to_v(cond)

        if self.qk_norm:
            query = self.q_norm(query)
            key = self.k_norm(key)

        head_dim = self.inner_dim // self.heads
        # FlashAttention needs (batch, seq, heads, head_dim)
        query = query.view(batch_size, -1, self.heads, head_dim)
        key = key.view(batch_size, -1, self.heads, head_dim)
        value = value.view(batch_size, -1, self.heads, head_dim)

        if rope is not None:
            # Rotary embedding needs (batch, heads, seq, head_dim)
            query = query.transpose(1, 2)
            query = _apply_rotary_emb(query, rope)
            query = query.transpose(1, 2).contiguous()
        if cond_rope is not None:
            key = key.transpose(1, 2)
            key = _apply_rotary_emb(key, cond_rope)
            key = key.transpose(1, 2).contiguous()

        attn_metadata = AttentionMetadata(attn_mask=cond_mask) if cond_mask is not None else None
        out = self.attn(query, key, value, attn_metadata=attn_metadata)

        out = out.reshape(batch_size, -1, self.inner_dim).to(query.dtype)
        out = self.to_out[0](out)
        out = self.to_out[1](out)
        return out


class AudioDiTFeedForward(nn.Module):
    """Feed-forward network."""

    def __init__(self, dim: int, mult: float = 4.0, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        inner_dim = int(dim * mult)
        self.ff = nn.Sequential(
            nn.Linear(dim, inner_dim, bias=bias),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim, bias=bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff(x)


class AudioDiTBlock(nn.Module):
    """Single DiT block with self-attention, optional cross-attention, FFN, and AdaLN modulation."""

    def __init__(
        self,
        dim: int,
        cond_dim: int,
        heads: int,
        dim_head: int,
        dropout: float = 0.0,
        bias: bool = True,
        qk_norm: bool = False,
        eps: float = 1e-6,
        cross_attn: bool = True,
        cross_attn_norm: bool = False,
        adaln_type: str = "global",
        adaln_use_text_cond: bool = True,
        ff_mult: float = 4.0,
    ):
        super().__init__()
        self.adaln_type = adaln_type
        self.adaln_use_text_cond = adaln_use_text_cond

        if adaln_type == "local":
            self.adaln_mlp = AudioDiTAdaLNMLP(dim, dim * 6, bias=bias)
        elif adaln_type == "global":
            self.adaln_scale_shift = nn.Parameter(torch.randn(dim * 6) / dim**0.5)

        self.self_attn = AudioDiTSelfAttention(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            bias=bias,
            qk_norm=qk_norm,
            eps=eps,
        )

        self.use_cross_attn = cross_attn
        if cross_attn:
            self.cross_attn = AudioDiTCrossAttention(
                q_dim=dim,
                kv_dim=cond_dim,
                heads=heads,
                dim_head=dim_head,
                dropout=dropout,
                bias=bias,
                qk_norm=qk_norm,
                eps=eps,
            )
            self.cross_attn_norm = (
                nn.LayerNorm(dim, elementwise_affine=True, eps=eps) if cross_attn_norm else nn.Identity()
            )
            self.cross_attn_norm_c = (
                nn.LayerNorm(cond_dim, elementwise_affine=True, eps=eps) if cross_attn_norm else nn.Identity()
            )

        self.ffn = AudioDiTFeedForward(dim=dim, mult=ff_mult, dropout=dropout, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        mask: torch.BoolTensor | None = None,
        cond_mask: torch.BoolTensor | None = None,
        rope: tuple | None = None,
        cond_rope: tuple | None = None,
        adaln_global_out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.adaln_type == "local" and adaln_global_out is None:
            if self.adaln_use_text_cond:
                cond_mean = cond.sum(1) / cond_mask.sum(1, keepdim=True)
                norm_cond = t + cond_mean
            else:
                norm_cond = t
            adaln_out = self.adaln_mlp(norm_cond)
            gate_sa, scale_sa, shift_sa, gate_ffn, scale_ffn, shift_ffn = torch.chunk(adaln_out, 6, dim=-1)
        else:
            adaln_out = adaln_global_out + self.adaln_scale_shift.unsqueeze(0)
            gate_sa, scale_sa, shift_sa, gate_ffn, scale_ffn, shift_ffn = torch.chunk(adaln_out, 6, dim=-1)

        # Self-attention
        norm = _modulate(x, scale_sa, shift_sa)
        attn_output = self.self_attn(norm, mask=mask, rope=rope)
        if gate_sa.ndim == 2:
            gate_sa = gate_sa.unsqueeze(1)
        x = x + gate_sa * attn_output

        # Cross-attention
        if self.use_cross_attn:
            cross_out = self.cross_attn(
                x=self.cross_attn_norm(x),
                cond=self.cross_attn_norm_c(cond),
                mask=mask,
                cond_mask=cond_mask,
                rope=rope,
                cond_rope=cond_rope,
            )
            x = x + cross_out

        # FFN
        norm = _modulate(x, scale_ffn, shift_ffn)
        ff_output = self.ffn(norm)
        if gate_ffn.ndim == 2:
            gate_ffn = gate_ffn.unsqueeze(1)
        x = x + gate_ffn * ff_output
        return x


class LongCatAudioDiTTransformer(nn.Module):
    """The DiT transformer backbone for LongCat-AudioDiT.

    This is adapted to use vllm-omni's diffusion attention backends.
    """

    def __init__(
        self,
        dit_dim: int = 1536,
        dit_depth: int = 24,
        dit_heads: int = 24,
        dit_text_dim: int = 768,
        latent_dim: int = 64,
        dropout: float = 0.0,
        bias: bool = True,
        cross_attn: bool = True,
        adaln_type: str = "global",
        adaln_use_text_cond: bool = True,
        long_skip: bool = True,
        text_conv: bool = True,
        qk_norm: bool = True,
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        use_latent_condition: bool = True,
    ):
        super().__init__()
        dim = dit_dim
        text_dim = dit_text_dim
        dim_head = dim // dit_heads

        self.dim = dim
        self.depth = dit_depth
        self.long_skip = long_skip
        self.adaln_type = adaln_type
        self.adaln_use_text_cond = adaln_use_text_cond
        self.bias = bias

        self.time_embed = AudioDiTTimestepEmbedding(dim)
        self.input_embed = AudioDiTEmbedder(latent_dim, dim)
        self.text_embed = AudioDiTEmbedder(text_dim, dim)
        self.rotary_embed = AudioDiTRotaryEmbedding(dim_head, 2048, base=100000.0)

        self.blocks = nn.ModuleList(
            [
                AudioDiTBlock(
                    dim=dim,
                    cond_dim=dim,
                    heads=dit_heads,
                    dim_head=dim_head,
                    dropout=dropout,
                    bias=bias,
                    qk_norm=qk_norm,
                    eps=eps,
                    cross_attn=cross_attn,
                    cross_attn_norm=cross_attn_norm,
                    adaln_type=adaln_type,
                    adaln_use_text_cond=adaln_use_text_cond,
                    ff_mult=4.0,
                )
                for _ in range(dit_depth)
            ]
        )

        self.norm_out = AudioDiTAdaLayerNormZeroFinal(dim, bias=bias, eps=eps)
        self.proj_out = nn.Linear(dim, latent_dim, bias=bias)

        if adaln_type == "global":
            self.adaln_global_mlp = AudioDiTAdaLNMLP(dim, dim * 6, bias=bias)

        self.text_conv = text_conv
        if text_conv:
            self.text_conv_layer = nn.Sequential(
                *[AudioDiTConvNeXtV2Block(dim, dim * 2, bias=bias, eps=eps) for _ in range(4)]
            )

        self.use_latent_condition = use_latent_condition
        if use_latent_condition:
            self.latent_embed = AudioDiTEmbedder(latent_dim, dim)
            self.latent_cond_embedder = AudioDiTEmbedder(dim * 2, dim)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights similar to original."""
        if self.adaln_type == "local":
            for block in self.blocks:
                nn.init.constant_(block.adaln_mlp.mlp[-1].weight, 0)
                if block.adaln_mlp.mlp[-1].bias is not None:
                    nn.init.constant_(block.adaln_mlp.mlp[-1].bias, 0)
        elif self.adaln_type == "global":
            nn.init.constant_(self.adaln_global_mlp.mlp[-1].weight, 0)
            if self.adaln_global_mlp.mlp[-1].bias is not None:
                nn.init.constant_(self.adaln_global_mlp.mlp[-1].bias, 0)

        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.proj_out.weight, 0)
        if self.norm_out.linear.bias is not None:
            nn.init.constant_(self.norm_out.linear.bias, 0)
        if self.proj_out.bias is not None:
            nn.init.constant_(self.proj_out.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        text: torch.Tensor,
        text_len: torch.Tensor,
        time: torch.Tensor,
        mask: torch.BoolTensor | None = None,
        cond_mask: torch.BoolTensor | None = None,
        latent_cond: torch.Tensor | None = None,
        return_ith_layer: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Noised latent input (batch, seq, latent_dim)
            text: Text embeddings (batch, text_seq, text_dim)
            text_len: Text sequence lengths (batch,)
            time: Timestep (batch,) or scalar
            mask: Audio mask (batch, seq)
            cond_mask: Text mask (batch, text_seq)
            latent_cond: Latent conditioning for voice cloning

        Returns:
            dict with "last_hidden_state": predicted velocity
        """
        dtype = next(self.parameters()).dtype
        x = x.to(dtype)
        text = text.to(dtype)
        time = time.to(dtype)

        batch = x.shape[0]
        text_seq_len = text.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)

        t = self.time_embed(time)
        text = self.text_embed(text, cond_mask)
        if self.text_conv:
            text = self.text_conv_layer(text)
            text = text.masked_fill(cond_mask.logical_not().unsqueeze(-1), 0.0)

        x = self.input_embed(x, mask)
        if self.use_latent_condition and latent_cond is not None:
            latent_cond = latent_cond.to(dtype)
            latent_cond = self.latent_embed(latent_cond, mask)
            x = self.latent_cond_embedder(torch.cat([x, latent_cond], dim=-1))

        if self.long_skip:
            x_clone = x.clone()

        seq_len = x.shape[1]
        rope = self.rotary_embed(x, seq_len)
        cond_rope = self.rotary_embed(text, text_seq_len)

        if self.adaln_type == "global":
            if self.adaln_use_text_cond:
                text_mean = text.sum(1) / text_len.unsqueeze(1).to(text.dtype)
                norm_cond = t + text_mean
            else:
                norm_cond = t
            adaln_mlp_out = self.adaln_global_mlp(norm_cond)

            for i, block in enumerate(self.blocks):
                x = block(
                    x=x,
                    t=t,
                    cond=text,
                    mask=mask,
                    cond_mask=cond_mask,
                    rope=rope,
                    cond_rope=cond_rope,
                    adaln_global_out=adaln_mlp_out,
                )
                if return_ith_layer is not None and i + 1 == return_ith_layer:
                    if self.long_skip:
                        x = x + x_clone
        else:
            for i, block in enumerate(self.blocks):
                x = block(
                    x=x,
                    t=t,
                    cond=text,
                    mask=mask,
                    cond_mask=cond_mask,
                    rope=rope,
                    cond_rope=cond_rope,
                )
                if return_ith_layer is not None and i + 1 == return_ith_layer:
                    if self.long_skip:
                        x = x + x_clone

        if self.long_skip:
            x = x + x_clone

        x = self.norm_out(x, norm_cond if self.adaln_type == "global" else t)
        x = self.proj_out(x)

        return {"last_hidden_state": x}
