# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 quantization utilities for diffusion attention tensors.

Provides per-tensor dynamic quantization of Q/K/V tensors to
float8_e4m3fn format. Designed for diffusion models where Q/K/V are
computed fresh each forward pass (no persistent KV cache).
"""

from __future__ import annotations

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
def _load_sd_ops():
    try:
        import torch_npu  # noqa: F401  # availability check: registers NPU ops
        from mindiesd.quantization.layer import fp8_rotate_quant_fa_op
        from msmodelslim.processor.quarot.common.quarot_utils import QuaRotMode, create_rot
    except ImportError as e:
        raise ImportError(
            "fp8_rotate_quant_fa requires torch_npu, MindIE-SD (mindiesd), and MSModelSlim. "
            "See https://gitcode.com/Ascend/MindIE-SD and https://gitcode.com/Ascend/msmodelslim"
        ) from e
    return fp8_rotate_quant_fa_op, QuaRotMode, create_rot


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
        layout: Input/output tensor layout, either ``BNSD`` or ``BSND``.
        softmax_scale: If None, uses ``1 / sqrt(head_dim)``.

    Returns:
        Attention output in the same layout as inputs.
    """
    fp8_op, qua_rot_mode, create_rot = _load_sd_ops()
    device = query.device

    if layout == "BNSD":
        _, n, s, d = query.shape
    elif layout == "BSND":
        _, s, n, d = query.shape
    else:
        raise ValueError(f"fp8_rotate_quant_fa: unsupported layout {layout!r}, expected BNSD or BSND")

    rot = _get_rot_matrix(device, query.dtype, d, qua_rot_mode, create_rot)

    return fp8_op(query, key, value, q_rot=rot, k_rot=rot, layout=layout, softmax_scale=softmax_scale)
