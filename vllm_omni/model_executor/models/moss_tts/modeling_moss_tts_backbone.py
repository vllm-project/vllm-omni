# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# ruff: noqa: N803
"""Qwen3 backbone of the MOSS-TTS Local Talker, with explicit kernels.

The Talker runs without ``torch.compile``. At MOSS decode sizes vLLM's CUDA
custom ops for the backbone's residual + RMSNorm, SiLU-and-mul and Q/K
norm + RoPE are slower than the kernels Inductor generated, so these classes
run Triton kernels for them (launched with PDL on Hopper). Weights, their
names and every other op are vLLM's Qwen3.
"""

import torch
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.models.qwen2 import Qwen2Model
from vllm.model_executor.models.qwen3 import Qwen3DecoderLayer
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

from .triton_launch import cuda_pdl_launch_kwargs

_USE_PDL = current_platform.is_cuda() and current_platform.has_device_capability(90)


@triton.jit
def _rms_norm(
    x_ptr,
    r_ptr,
    w_ptr,
    out_ptr,
    res_ptr,
    sx,
    sr,
    N: tl.constexpr,
    eps,
    BN: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BN)
    m = cols < N
    x = tl.load(x_ptr + row * sx + cols, m, 0.0).to(tl.float32)
    if HAS_RESIDUAL:
        x = x + tl.load(r_ptr + row * sr + cols, m, 0.0).to(tl.float32)
        tl.store(res_ptr + row * N + cols, x.to(res_ptr.dtype.element_ty), m)
    rstd = tl.rsqrt(tl.sum(x * x, 0) / N + eps)
    # As vLLM's rms_norm: normalize in FP32, round, then scale by the weight.
    y = (x * rstd).to(out_ptr.dtype.element_ty).to(tl.float32) * tl.load(w_ptr + cols, m, 0.0).to(tl.float32)
    tl.store(out_ptr + row * N + cols, y.to(out_ptr.dtype.element_ty), m)


def rms_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float, residual: torch.Tensor | None = None
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """vLLM's ``rms_norm``, or ``fused_add_rms_norm`` with ``residual``, in one kernel."""
    N = x.shape[-1]
    x2 = x.reshape(-1, N)
    rows = x2.shape[0]
    out = torch.empty((rows, N), dtype=x.dtype, device=x.device)
    r2 = residual.reshape(-1, N) if residual is not None else x2
    res = torch.empty_like(out) if residual is not None else out
    if rows:
        _rms_norm[(rows,)](
            x2,
            r2,
            weight,
            out,
            res,
            x2.stride(0),
            r2.stride(0),
            N,
            eps,
            triton.next_power_of_2(N),
            residual is not None,
            USE_PDL=_USE_PDL,
            num_warps=4 if N <= 4096 else 8,
            **cuda_pdl_launch_kwargs(_USE_PDL),
        )
    if residual is None:
        return out.view(x.shape)
    return out.view(x.shape), res.view(residual.shape)


@triton.jit
def _silu_and_mul(
    x_ptr,
    out_ptr,
    sx,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = cols < D
    gate = tl.load(x_ptr + row * sx + cols, m, 0.0).to(tl.float32)
    up = tl.load(x_ptr + row * sx + D + cols, m, 0.0).to(tl.float32)
    # As F.silu(gate) * up on BF16 tensors: two roundings.
    act = (gate / (1.0 + tl.exp(-gate))).to(out_ptr.dtype.element_ty).to(tl.float32)
    tl.store(out_ptr + row * D + cols, (act * up).to(out_ptr.dtype.element_ty), m)


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """``silu(x[..., :d]) * x[..., d:]`` in one kernel."""
    d = x.shape[-1] // 2
    x2 = x.reshape(-1, x.shape[-1])
    out = torch.empty((x2.shape[0], d), dtype=x.dtype, device=x.device)
    if x2.shape[0]:
        block = 1024
        _silu_and_mul[(x2.shape[0], triton.cdiv(d, block))](
            x2,
            out,
            x2.stride(0),
            d,
            block,
            USE_PDL=_USE_PDL,
            num_warps=4,
            **cuda_pdl_launch_kwargs(_USE_PDL),
        )
    return out.view(*x.shape[:-1], d)


@triton.jit
def _qk_norm_rope(
    qkv_ptr,
    pos_ptr,
    qw_ptr,
    kw_ptr,
    cache_ptr,
    sqkv,
    scache,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    eps,
    BH: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    t = tl.program_id(0).to(tl.int64)
    heads = tl.program_id(1) * BH + tl.arange(0, BH)
    hmask = (heads < HQ + HK)[:, None]
    half: tl.constexpr = D // 2
    i = tl.arange(0, half)
    pos = tl.load(pos_ptr + t)
    cos = tl.load(cache_ptr + pos * scache + i).to(tl.float32)[None, :]
    sin = tl.load(cache_ptr + pos * scache + half + i).to(tl.float32)[None, :]
    base = qkv_ptr + t * sqkv + heads[:, None] * D + i[None, :]
    x1 = tl.load(base, hmask, 0.0).to(tl.float32)
    x2 = tl.load(base + half, hmask, 0.0).to(tl.float32)
    rstd = tl.rsqrt((tl.sum(x1 * x1, 1) + tl.sum(x2 * x2, 1)) / D + eps)[:, None]
    is_q = (heads < HQ)[:, None]
    w1 = tl.where(is_q, tl.load(qw_ptr + i)[None, :], tl.load(kw_ptr + i)[None, :]).to(tl.float32)
    w2 = tl.where(is_q, tl.load(qw_ptr + half + i)[None, :], tl.load(kw_ptr + half + i)[None, :]).to(tl.float32)
    dtype = qkv_ptr.dtype.element_ty
    # vLLM's rms_norm per head (normalize, round, scale, round), then NeoX RoPE.
    n1 = ((x1 * rstd).to(dtype).to(tl.float32) * w1).to(dtype).to(tl.float32)
    n2 = ((x2 * rstd).to(dtype).to(tl.float32) * w2).to(dtype).to(tl.float32)
    tl.store(base, (n1 * cos - n2 * sin).to(dtype), hmask)
    tl.store(base + half, (n2 * cos + n1 * sin).to(dtype), hmask)


def qk_norm_rope_(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    eps: float,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> None:
    """Per-head RMSNorm of Q and K, then NeoX RoPE, in place on ``qkv`` (T, (Hq + 2 Hk) * D)."""
    tokens = qkv.shape[0]
    if tokens:
        heads_per_program = 8
        _qk_norm_rope[(tokens, triton.cdiv(num_heads + num_kv_heads, heads_per_program))](
            qkv,
            positions,
            q_weight,
            k_weight,
            cos_sin_cache,
            qkv.stride(0),
            cos_sin_cache.stride(0),
            num_heads,
            num_kv_heads,
            head_dim,
            eps,
            heads_per_program,
            USE_PDL=_USE_PDL,
            num_warps=4,
            **cuda_pdl_launch_kwargs(_USE_PDL),
        )


class MossRMSNorm(RMSNorm):
    """``RMSNorm`` (with optional residual add) as one kernel."""

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return rms_norm(x, self.weight, self.variance_epsilon, residual)


class MossSiluAndMul(SiluAndMul):
    """``SiluAndMul`` as one kernel."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return silu_and_mul(x)


class MossQwen3DecoderLayer(Qwen3DecoderLayer):
    """Qwen3 decoder layer whose norms, activation and Q/K norm + RoPE run the kernels above."""

    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
        self.input_layernorm = MossRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MossRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp.act_fn = MossSiluAndMul()
        rope = self.self_attn.rotary_emb
        if not rope.is_neox_style or rope.cos_sin_cache.shape[-1] != self.self_attn.head_dim:
            raise ValueError("MOSS backbone Q/K kernel needs full-dimension NeoX RoPE")

    def _attention(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        attn = self.self_attn
        qkv, _ = attn.qkv_proj(hidden_states)
        qk_norm_rope_(
            qkv,
            positions,
            attn.num_heads,
            attn.num_kv_heads,
            attn.head_dim,
            attn.q_norm.variance_epsilon,
            attn.q_norm.weight,
            attn.k_norm.weight,
            attn.rotary_emb.cos_sin_cache,
        )
        q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
        output, _ = attn.o_proj(attn.attn(q, k, v))
        return output

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self._attention(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        return self.mlp(hidden_states), residual


class MossQwen3Model(Qwen2Model):
    """vLLM's ``Qwen3Model`` built from ``MossQwen3DecoderLayer`` and a ``MossRMSNorm`` final norm."""

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix, decoder_layer_type=MossQwen3DecoderLayer)
        if isinstance(self.norm, RMSNorm):
            self.norm = MossRMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)


__all__ = ["MossQwen3Model", "qk_norm_rope_", "rms_norm", "silu_and_mul"]
