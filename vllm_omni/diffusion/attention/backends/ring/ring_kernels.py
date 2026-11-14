# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Copyright (c) 2024, Jiarui Fang.
# Adapted from https://github.com/feifeibear/long-context-attention

import functools
import math

import torch

from vllm_omni.diffusion.attention.backends.utils.fa import vllm_flash_attn_dense_with_lse

from .ring_globals import (
    HAS_AITER,
    HAS_FA3,
    HAS_FA4,
    HAS_FLASH_ATTN,
    HAS_FLASHINFER,
    fa3_fwd_func,
    fa4_attn_func,
)

_scaled_dot_product_flash_attention = torch.ops.aten._scaled_dot_product_flash_attention
_scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention

try:
    import torch_musa  # noqa: F401

    _scaled_dot_product_flash_attention = torch.ops.aten._scaled_dot_product_attention_flash_musa
    _scaled_dot_product_efficient_attention = None
except ModuleNotFoundError:
    pass

if HAS_AITER:
    from aiter import flash_attn_func as flash_attn_func_aiter

if HAS_FLASH_ATTN:
    import flash_attn
    from flash_attn.flash_attn_interface import _flash_attn_forward

if HAS_FLASHINFER:
    from flashinfer.prefill import single_prefill_with_kv_cache

    _LOG2_E = math.log2(math.e)


@functools.cache
def _xpu_has_aten_flash() -> bool:
    """Whether this torch build has the XPU aten flash SDPA op (it needs SYCL-TLA; some builds lack it)."""
    q = torch.zeros(1, 1, 16, 64, device="xpu", dtype=torch.bfloat16)
    try:
        _scaled_dot_product_flash_attention(q, q, q)
    except RuntimeError:
        return False
    return True


def _xpu_sdpa_op_type(
    op_type: str, dtype: torch.dtype, causal: bool, q_len: int, kv_len: int, has_flash: bool = True
) -> str:
    """Pick the aten SDPA op that returns a usable LSE on XPU.

    XPU's ``_scaled_dot_product_efficient_attention`` returns a correct output but an all-zero
    log-sum-exp, which silently breaks the ring merge. Its flash op returns the right fp32 LSE, but
    only for fp16/bf16, aligns a causal mask to the bottom-right when ``q_len != kv_len`` (the
    efficient op aligns it to the top-left), and is missing from torch builds without SYCL-TLA.
    Those cases use the fp32 ``"math"`` path instead.
    """
    if op_type != "efficient":
        return op_type
    if has_flash and dtype in (torch.float16, torch.bfloat16) and (not causal or q_len == kv_len):
        return "flash"
    return "math"


def _math_attn_with_lse(q, k, v, softmax_scale, causal, q_chunk: int = 1024):
    """Reference attention in fp32 on (B, H, S, D) tensors: returns (out, lse) like the aten SDPA ops.

    Query rows are processed ``q_chunk`` at a time, so the fp32 score block stays at
    ``B * H * q_chunk * kv_len`` instead of the full ``q_len * kv_len``.
    """
    q_len, kv_len = q.shape[-2], k.shape[-2]
    kf, vf = k.float(), v.float()
    outs, lses = [], []
    for start in range(0, q_len, q_chunk):
        scores = torch.matmul(q[..., start : start + q_chunk, :].float(), kf.transpose(-2, -1)) * softmax_scale
        if causal:
            rows = torch.arange(start, start + scores.shape[-2], device=scores.device).view(-1, 1)
            keep = rows >= torch.arange(kv_len, device=scores.device).view(1, -1)  # top-left, like the efficient op
            scores = scores.masked_fill(~keep, float("-inf"))
        lses.append(torch.logsumexp(scores, dim=-1))
        outs.append(torch.matmul(torch.softmax(scores, dim=-1), vf).to(q.dtype))
    return torch.cat(outs, dim=-2), torch.cat(lses, dim=-1)


def pytorch_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p=0.0,
    softmax_scale=None,
    causal=True,
    window_size=(-1, -1),
    softcap=None,
    alibi_slopes=None,
    return_softmax=False,
    op_type="efficient",
):
    assert op_type in ["flash", "efficient"], f"Invalid op_type: {op_type}"
    """
    q shape (bs, seqlen, nhead, hs)
    k shape (bs, seqlen, nhead, hs)
    v shape (bs, seqlen, nhead, hs)
    """
    if op_type == "flash" and q.dtype == torch.float32:
        raise ValueError("The explicitly requested Flash ring kernel does not support float32.")

    # Volta (V100, sm70) does not have native bf16 support for the efficient
    # SDPA kernel. When running Ring attention in bf16 on such GPUs, the
    # efficient kernel can fail with "cutlassF: no kernel found to launch".
    #
    # We keep the overall model in bf16 for numerical stability, but cast
    # the attention inputs to fp16 for the SDPA call. The output is cast
    # back to bf16 to match the caller's expected dtype.
    orig_dtype = q.dtype
    if q.is_cuda and q.dtype == torch.bfloat16 and op_type == "efficient":
        major_cc, _minor_cc = torch.cuda.get_device_capability(q.device)
        if major_cc < 8:
            q = q.to(torch.float16)
            k = k.to(torch.float16)
            v = v.to(torch.float16)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    # GQA/MQA support. Ring attention only ever *communicates* K/V with H_kv
    # heads -- that is precisely the benefit of GQA -- so expanding here does
    # not change communication volume. The expansion is needed because the aten
    # SDPA ops below are required for the LSE output that ring accumulation
    # consumes, and they do not accept `enable_gqa`.
    if k.shape[1] != q.shape[1]:
        num_heads, num_kv_heads = q.shape[1], k.shape[1]
        if num_heads % num_kv_heads != 0:
            raise ValueError(f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads}) for GQA/MQA.")
        n_rep = num_heads // num_kv_heads
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)

    if q.device.type == "xpu":
        op_type = _xpu_sdpa_op_type(op_type, q.dtype, causal, q.shape[2], k.shape[2], _xpu_has_aten_flash())

    if op_type == "math":
        if dropout_p:
            raise ValueError("The math ring kernel does not support dropout.")
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** -0.5
        out, lse = _math_attn_with_lse(q, k, v, softmax_scale, causal)
    elif op_type == "flash":
        out, lse = _scaled_dot_product_flash_attention(
            q,
            k,
            v,
            dropout_p=dropout_p,
            is_causal=causal,
            scale=softmax_scale,
        )[:2]
    elif op_type == "efficient":
        out, lse = _scaled_dot_product_efficient_attention(
            q,
            k,
            v,
            attn_bias=None,
            compute_log_sumexp=True,
            dropout_p=dropout_p,
            is_causal=causal,
            scale=softmax_scale,
        )[:2]
    else:
        raise ValueError(f"Invalid op_type: {op_type}")

    out = out.transpose(1, 2)
    # Keep LSE in fp32 for numerical stability when accumulating across ring
    # steps (update_out_and_lse uses sigmoid/logsigmoid on LSE diffs).
    # Casting LSE down to fp16/bf16 can introduce NaNs on some GPUs/shapes.
    # PyTorch SDPA returns LSE as (B, H, S).
    lse = lse.to(torch.float32)

    if out.dtype != orig_dtype:
        out = out.to(orig_dtype)

    return out, lse


def flash_attn_forward(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=None,
    alibi_slopes=None,
    return_softmax=False,
):
    assert HAS_FLASH_ATTN, "FlashAttention is not available"
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    if flash_attn.__version__ < "2.6.3":
        block_out, _, _, _, _, block_lse, _, _ = _flash_attn_forward(
            q,
            k,
            v,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax,
        )
    else:
        block_out, block_lse, _, _ = _flash_attn_forward(
            q,
            k,
            v,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax,
        )
    # FA2 softmax_lse is (B, H, S), optionally padded on S.
    return block_out, block_lse


def fa3_forward(q, k, v, dropout_p, softmax_scale, causal, window_size, softcap, alibi_slopes, return_softmax):
    """FA3 forward pass for inference.

    This optional source-built FA3 fallback supports Hopper GPUs. Dropout is
    ignored since FA3 is inference-only. Its low-level API
    (``_flash_attn_forward``) always returns the softmax LSE required for Ring
    Attention's correct accumulation.
    """
    assert HAS_FA3, "FA3 is not available"
    assert fa3_fwd_func is not None, "FA3 low-level API (fa3_fwd_func) not available"

    # Low-level API always returns (out, softmax_lse, S_dmask, rng_state)
    out, softmax_lse, *_ = fa3_fwd_func(
        q,
        k,
        v,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size_left=window_size[0] if window_size else -1,
        window_size_right=window_size[1] if window_size else -1,
        softcap=softcap if softcap else 0.0,
    )

    # FA3 softmax_lse is (B, H, S).
    return out, softmax_lse


def vllm_flash_attn_forward(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=None,
    alibi_slopes=None,
    return_softmax=False,
):
    """vLLM-bundled FlashAttention forward pass with ring-compatible LSE."""
    del dropout_p, window_size, alibi_slopes, return_softmax
    return vllm_flash_attn_dense_with_lse(
        q,
        k,
        v,
        softmax_scale=softmax_scale,
        causal=causal,
        softcap=softcap or 0.0,
    )


# Legacy alias for backward compatibility
flash_attn3_func_forward = fa3_forward


if not hasattr(torch.ops.vllm_omni, "fa4_ring_attention"):

    @torch.library.custom_op("vllm_omni::fa4_ring_attention", mutates_args=())
    def _fa4_ring_attention_op(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        softmax_scale: float | None,
        causal: bool,
        window_left: int,
        window_right: int,
        softcap: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # The CuTe Python launcher is opaque; its attention arithmetic is unchanged.
        if fa4_attn_func is None:
            raise RuntimeError("CuTe FlashAttention-4 is unavailable")
        out, lse = fa4_attn_func(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=(None if window_left == -1 else window_left, None if window_right == -1 else window_right),
            softcap=softcap,
            return_lse=True,
        )
        return out, lse

    @_fa4_ring_attention_op.register_fake
    def _fa4_ring_attention_op_fake(q, k, v, softmax_scale, causal, window_left, window_right, softcap):
        # Match FA4's contiguous BSHD output and contiguous FP32 BHS LSE.
        return (
            torch.empty((q.shape[0], q.shape[1], q.shape[2], v.shape[-1]), dtype=q.dtype, device=q.device),
            torch.empty((q.shape[0], q.shape[2], q.shape[1]), dtype=torch.float32, device=q.device),
        )


_fa4_ring_attention_op = torch.ops.vllm_omni.fa4_ring_attention


def flash_attn4_func_forward(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=None,
    alibi_slopes=None,
    return_softmax=False,
):
    """FA4 forward pass returning the LSE required by Ring Attention."""
    del dropout_p, alibi_slopes, return_softmax
    assert HAS_FA4, "FA4 is not available"
    assert fa4_attn_func is not None, "FA4 high-level API is not available"

    left, right = window_size if window_size else (-1, -1)
    if torch.compiler.is_compiling() and not torch.is_grad_enabled():
        return _fa4_ring_attention_op(q, k, v, softmax_scale, causal, left, right, softcap or 0.0)
    out, softmax_lse = fa4_attn_func(
        q,
        k,
        v,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=(
            None if left == -1 else left,
            None if right == -1 else right,
        ),
        softcap=softcap or 0.0,
        return_lse=True,
    )
    # FA4 CuTe softmax_lse is (B, H, S).
    return out, softmax_lse


def flash_attn_forward_aiter(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=None,
    alibi_slopes=None,
    return_softmax=False,
):
    assert HAS_AITER, "Aiter is not available"
    block_out, block_lse = flash_attn_func_aiter(
        q,
        k,
        v,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
        return_lse=True,
    )

    # Aiter softmax_lse is (B, H, S), matching FA2.
    return block_out, block_lse


def flashinfer_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    softcap: float | None = None,
    alibi_slopes: torch.Tensor | None = None,
    return_softmax: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert HAS_FLASHINFER, "FlashInfer is not available"
    if q.ndim == 4:
        if q.shape[0] > 1:
            raise ValueError("batch size > 1 is not supported")
        out, lse = single_prefill_with_kv_cache(
            q[0],
            k[0],
            v[0],
            sm_scale=softmax_scale,
            causal=causal,
            logits_soft_cap=softcap,
            window_left=window_size[0],
            return_lse=True,
        )
        # FlashInfer unbatched LSE is (H, S). Canonical ring layout is (B, H, S).
        out, lse = out.unsqueeze(0), lse.unsqueeze(0)
    elif q.ndim == 3:
        out, lse = single_prefill_with_kv_cache(
            q,
            k,
            v,
            sm_scale=softmax_scale,
            causal=causal,
            logits_soft_cap=softcap,
            window_left=window_size[0],
            return_lse=True,
        )
        # Unbatched (H, S) matches canonical heads-major layout without a batch dim.
    else:
        raise ValueError(f"Invalid input shape: {q.shape}")
    lse = lse / _LOG2_E
    return out, lse
