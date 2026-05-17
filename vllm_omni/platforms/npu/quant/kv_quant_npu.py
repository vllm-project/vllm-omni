# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 quantization utilities for diffusion attention tensors.

Provides per-tensor dynamic quantization of Q/K/V tensors to
float8_e4m3fn format. Designed for diffusion models where Q/K/V are
computed fresh each forward pass (no persistent KV cache).
"""

from __future__ import annotations

import math
import threading
from functools import lru_cache

import torch

# Hadamard rotation matrix for QuaRot-style preprocessing
# keyed by (device, dtype, head_dim) to avoid matmul dtype mismatch.
_ROT_MATRIXS: dict[tuple[torch.device, torch.dtype, int], torch.Tensor] = {}
_ROT_MATRIX_LOCK = threading.Lock()

_FP8_KV_LABELS = frozenset({"fp8"})


def is_quantized_kv_cache(kv_cache_dtype: str | None) -> bool:
    """True if config requests FP8-style KV / QKV quantization for the NPU FA path."""
    return kv_cache_dtype in _FP8_KV_LABELS


@lru_cache(maxsize=1)
def _load_quant_ops():
    try:
        import torch_npu
        from mindiesd.layers.quant.block_quant import fa_block_quant_preprocess
        from msmodelslim.processor.quarot.common.quarot_utils import QuaRotMode, create_rot
    except ImportError as e:
        raise ImportError(
            "fp8_rotate_quant_fa requires torch_npu, MindIE-SD (mindiesd), and MSModelSlim. "
            "See https://gitcode.com/Ascend/MindIE-SD and https://gitcode.com/Ascend/msmodelslim"
        ) from e
    return torch_npu, fa_block_quant_preprocess, QuaRotMode, create_rot


def _get_rot_matrix(
    device: torch.device,
    dtype: torch.dtype,
    head_dim: int,
    qua_rot_mode,
    create_rot,
) -> torch.Tensor:
    key = (device, dtype, head_dim)
    with _ROT_MATRIX_LOCK:
        rot = _ROT_MATRIXS.get(key)
        if rot is None:
            rot = create_rot(qua_rot_mode.HADAMARD, head_dim, seed=425500).to(device=device, dtype=dtype)
            _ROT_MATRIXS[key] = rot
    return rot


def fp8_rotate_quant_fa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    layout: str = "BNSD",
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Run NPU fused attention with dynamic FP8 Q/K/V and optional QuaRot preprocess.

    Args:
        query: Query tensor in ``layout`` order (default BNSD: batch, heads, seq, dim).
        key: Key tensor in ``layout`` order (default BNSD: batch, heads, seq, dim).
        value: Value tensor in ``layout`` order (default BNSD: batch, heads, seq, dim).
        layout: ``BNSD`` or ``BSND`` for ``npu_fused_infer_attention_score_v2``.
        softmax_scale: If None, uses ``1 / sqrt(head_dim)``.

    Returns:
        Attention output in the same layout as inputs.
    """
    torch_npu, fa_block_quant_preprocess, qua_rot_mode, create_rot = _load_quant_ops()

    out_dtype = query.dtype
    device = query.device

    if layout == "BNSD":
        _, n, s, d = query.shape
    elif layout == "BSND":
        _, s, n, d = query.shape
    else:
        raise ValueError(f"fp8_rotate_quant_fa: unsupported layout {layout!r}, expected BNSD or BSND")

    rot = _get_rot_matrix(device, query.dtype, d, qua_rot_mode, create_rot)
    q_f = torch.matmul(query, rot)
    k_f = torch.matmul(key, rot)

    q, q_scale = fa_block_quant_preprocess(q_f, block_size=128, dst_type=torch_npu.float8_e4m3fn, layout=layout)
    k, k_scale = fa_block_quant_preprocess(k_f, block_size=256, dst_type=torch_npu.float8_e4m3fn, layout=layout)
    v, v_scale = fa_block_quant_preprocess(value, block_size=256, dst_type=torch_npu.float8_e4m3fn, layout=layout)

    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(d)

    out = torch_npu.npu_fused_infer_attention_score_v2(
        q,
        k,
        v,
        input_layout=layout,
        num_query_heads=n,
        softmax_scale=scale,
        pre_tokens=2147483647,  # INT32_MAX: no left-context truncation.
        next_tokens=2147483647,  # INT32_MAX: no right-context truncation.
        query_quant_mode=7,  # NPU mode id for block FP8 dequant path.
        key_quant_mode=7,  # Same quant mode as query branch.
        value_quant_mode=7,  # Same quant mode as key/query branches.
        dequant_scale_query=q_scale,
        dequant_scale_key=k_scale,
        dequant_scale_value=v_scale,
        out_dtype=out_dtype,
    )[0]

    if out.shape[2] != s:
        if layout == "BNSD":
            out = out[:, :, :s, :]
        elif layout == "BSND":
            out = out[:, :s, :, :]

    return out
