# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn.functional as F
from torch import nn
from x_transformers.x_transformers import apply_rotary_pos_emb


def _apply_rope_cached(t, cos, sin):
    """Bit-exact RoPE for ``scale==1`` / interleaved-half.
    Uses precomputed ``cos``/``sin`` per seq_len to skip per-step trig and cat."""
    rot_dim = cos.shape[-1]
    tr = t[..., :rot_dim]
    x = tr.reshape(*tr.shape[:-1], -1, 2)
    rot_half = torch.stack((-x[..., 1], x[..., 0]), dim=-1).reshape_as(tr)
    out = tr * cos + rot_half * sin
    if rot_dim < t.shape[-1]:  # partial rotary (unused by ming heads); keep general
        out = torch.cat((out, t[..., rot_dim:]), dim=-1)
    return out.type(t.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            x = x.to(self.weight.dtype)
        x = F.rms_norm(x, normalized_shape=(x.shape[-1],), weight=self.weight, eps=self.eps)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, dropout=0.0, approximate="none"):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        activation = nn.GELU(approximate=approximate)
        project_in = nn.Sequential(nn.Linear(dim, inner_dim), activation)
        self.ff = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))

    def forward(self, x):
        return self.ff(x)


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.0,
        qk_norm=None,
        pe_attn_head=None,
        attn_mask_enabled=True,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.inner_dim = dim_head * heads
        self.dropout = dropout
        self.to_q = nn.Linear(dim, self.inner_dim)
        self.to_k = nn.Linear(dim, self.inner_dim)
        self.to_v = nn.Linear(dim, self.inner_dim)
        if qk_norm is None:
            self.q_norm = None
            self.k_norm = None
        elif qk_norm == "rms_norm":
            self.q_norm = RMSNorm(dim_head)
            self.k_norm = RMSNorm(dim_head)
        else:
            raise ValueError(f"Unimplemented qk_norm: {qk_norm}")

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(self.inner_dim, dim))
        self.to_out.append(nn.Dropout(dropout))
        self.pe_attn_head = pe_attn_head
        self.attn_mask_enabled = attn_mask_enabled
        # cos/sin cache keyed by seq_len (freqs are deterministic in seq_len).
        self._rope_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        # TODO: materialize fused weights eagerly outside any torch.compile
        # scope (e.g. post-weight-load);
        self._qkv_w: torch.Tensor | None = None
        self._qkv_b: torch.Tensor | None = None

    def _qkv(self, x):
        if self._qkv_w is None:
            self._qkv_w = torch.cat([self.to_q.weight, self.to_k.weight, self.to_v.weight], dim=0)
            if self.to_q.bias is not None:
                self._qkv_b = torch.cat([self.to_q.bias, self.to_k.bias, self.to_v.bias], dim=0)
        qkv = F.linear(x, self._qkv_w, self._qkv_b)
        return qkv.split(self.inner_dim, dim=-1)

    def _rope_cos_sin(self, freqs):
        seq_len = freqs.shape[-2]
        cached = self._rope_cache.get(seq_len)
        if cached is None:
            f = freqs.unsqueeze(1) if freqs.ndim == 3 else freqs  # b n d -> b 1 n d for 4D q/k
            cached = (f.cos(), f.sin())
            self._rope_cache[seq_len] = cached
        return cached

    def forward(self, x, mask=None, rope=None):
        batch_size = x.shape[0]
        query, key, value = self._qkv(x)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads
        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        if self.q_norm is not None:
            query = self.q_norm(query)
        if self.k_norm is not None:
            key = self.k_norm(key)

        if rope is not None:
            freqs, xpos_scale = rope
            scale_is_one = xpos_scale is None or (not torch.is_tensor(xpos_scale) and xpos_scale == 1.0)
            if scale_is_one and self.pe_attn_head is None:
                # Fast path: cached, cat-free RoPE (bit-exact for scale==1, full rotary).
                cos, sin = self._rope_cos_sin(freqs)
                query = _apply_rope_cached(query, cos, sin)
                key = _apply_rope_cached(key, cos, sin)
            else:
                q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)
                if self.pe_attn_head is not None:
                    on = self.pe_attn_head
                    query[:, :on, :, :] = apply_rotary_pos_emb(query[:, :on, :, :], freqs, q_xpos_scale)
                    key[:, :on, :, :] = apply_rotary_pos_emb(key[:, :on, :, :], freqs, k_xpos_scale)
                else:
                    query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
                    key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)

        if self.attn_mask_enabled and mask is not None:
            valid_sample_indices = mask.any(dim=1)
            final_output = torch.zeros_like(query).to(query.device)
            attn_mask = mask[valid_sample_indices]
            query = query[valid_sample_indices]
            key = key[valid_sample_indices]
            value = value[valid_sample_indices]
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)
            attn_mask = attn_mask.expand(valid_sample_indices.sum().item(), self.heads, query.shape[-2], key.shape[-2])
        else:
            attn_mask = None

        x = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        if self.attn_mask_enabled and mask is not None:
            final_output[valid_sample_indices] = x
            x = final_output

        x = x.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
        x = x.to(query.dtype)
        x = self.to_out[0](x)
        x = self.to_out[1](x)

        if mask is not None:
            mask = mask.unsqueeze(-1)
            x = x.masked_fill(~mask, 0.0)

        return x


class DiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        dropout=0.1,
        qk_norm=None,
        pe_attn_head=None,
        attn_mask_enabled=True,
        **kwargs,
    ):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = Attention(
            dim=hidden_size,
            heads=num_heads,
            dim_head=hidden_size // num_heads,
            dropout=dropout,
            qk_norm=qk_norm,
            pe_attn_head=pe_attn_head,
            attn_mask_enabled=attn_mask_enabled,
        )
        self.norm2 = RMSNorm(hidden_size)
        self.mlp = FeedForward(dim=hidden_size, mult=mlp_ratio, dropout=dropout, approximate="tanh")

    def forward(self, x, mask, rope):
        x = x + self.attn(self.norm1(x), mask=mask, rope=rope)
        x = x + self.mlp(self.norm2(x))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

    def forward(self, x):
        x = self.norm_final(x)
        x = self.linear(x)
        return x


class CondEmbedder(nn.Module):
    def __init__(self, input_feature_size, hidden_size):
        super().__init__()
        self.cond_embedder = nn.Linear(input_feature_size, hidden_size)

    def forward(self, llm_cond):
        return self.cond_embedder(llm_cond)


def get_epss_timesteps(n, device, dtype):
    dt = 1 / 32
    predefined_timesteps = {
        5: [0, 2, 4, 8, 16, 32],
        6: [0, 2, 4, 6, 8, 16, 32],
        7: [0, 2, 4, 6, 8, 16, 24, 32],
        10: [0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32],
        12: [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32],
        16: [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32],
    }
    t = predefined_timesteps.get(n, [])
    if not t:
        return torch.linspace(0, 1, n + 1, device=device, dtype=dtype)
    return dt * torch.tensor(t, device=device, dtype=dtype)
