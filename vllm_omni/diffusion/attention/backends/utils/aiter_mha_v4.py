# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import inspect
from collections.abc import Callable

import torch

# Guarded so that importing this module never fails on a machine without a usable
# aiter.ops.mha_v4; ``check_aiter_mha_v4_available`` reports the problem instead.
try:
    from aiter.ops.mha_v4 import (
        AttentionFormat as _AiterAttentionFormat,
    )
    from aiter.ops.mha_v4 import (
        AttentionScaleMode as _AiterAttentionScaleMode,
    )
    from aiter.ops.mha_v4 import (
        mha_v4 as _aiter_mha_v4,
    )
    from aiter.ops.mha_v4 import (
        native_fp8_format as _aiter_native_fp8_format,
    )

    _MHA_V4_IMPORT_ERROR: ImportError | None = None
except ImportError as exc:
    _AiterAttentionFormat = None
    _AiterAttentionScaleMode = None
    _aiter_mha_v4 = None
    _aiter_native_fp8_format = None
    _MHA_V4_IMPORT_ERROR = exc

# Keyword arguments ``_aiter_mixed_attn_call`` passes to ``mha_v4``. Keep in sync with that
# call: ``check_aiter_mha_v4_available`` verifies the installed aiter accepts every one.
_MHA_V4_REQUIRED_KWARGS = ("softmax_scale", "q_scale_mode", "k_scale_mode", "v_scale_mode")


def check_aiter_mha_v4_available() -> None:
    """Raise ``RuntimeError`` if the installed AITER lacks the ``mha_v4`` API used in this module."""
    if _MHA_V4_IMPORT_ERROR is not None:
        raise RuntimeError(
            "AITER_QUANT_ATTN requires an AITER build providing aiter.ops.mha_v4 "
            f"(AttentionFormat, AttentionScaleMode, mha_v4, native_fp8_format): {_MHA_V4_IMPORT_ERROR}. "
            "Please install or upgrade aiter."
        ) from _MHA_V4_IMPORT_ERROR
    accepted = inspect.signature(_aiter_mha_v4).parameters
    missing = [name for name in _MHA_V4_REQUIRED_KWARGS if name not in accepted]
    if missing:
        raise RuntimeError(
            f"AITER_QUANT_ATTN requires a newer AITER: mha_v4 does not accept {missing}. Please upgrade aiter."
        )


def _aiter_mixed_attn_call(query, key, value, qk_format, v_format, scale_modes=None, softmax_scale=None):
    """Call MHA v4; ``scale_modes`` is a (q, k, v) triple, or None for the canonical modes.

    ``softmax_scale=None`` lets AITER use its default of ``128**-0.5``.
    """
    scale_kwargs = {}
    if scale_modes is not None:
        q_scale_mode, k_scale_mode, v_scale_mode = scale_modes
        scale_kwargs = {
            "q_scale_mode": q_scale_mode,
            "k_scale_mode": k_scale_mode,
            "v_scale_mode": v_scale_mode,
        }

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    return _aiter_mha_v4(query, key, value, qk_format, qk_format, v_format, softmax_scale=softmax_scale, **scale_kwargs)


def _aiter_forward_bf16(query, key, value, softmax_scale=None):
    """Run the AITER MHA v4 BF16 Q/K/V recipe."""
    return _aiter_mixed_attn_call(
        query,
        key,
        value,
        _AiterAttentionFormat.BF16,
        _AiterAttentionFormat.BF16,
        softmax_scale=softmax_scale,
    )


def _aiter_forward_fp8(query, key, value, softmax_scale=None):
    """Run the AITER per-tensor FP8 Q/K/V recipe."""
    fp8_format = _aiter_native_fp8_format()
    return _aiter_mixed_attn_call(
        query,
        key,
        value,
        fp8_format,
        fp8_format,
        softmax_scale=softmax_scale,
    )


def _aiter_forward_i8fp8(query, key, value, softmax_scale=None):
    """Run the AITER INT8 Q/K and FP8 V recipe."""
    return _aiter_mixed_attn_call(
        query,
        key,
        value,
        _AiterAttentionFormat.INT8,
        _aiter_native_fp8_format(),
        softmax_scale=softmax_scale,
    )


def _aiter_forward_mxfp8(query, key, value, softmax_scale=None):
    """Run the AITER MXFP8 Q/K and per-tensor FP8 V recipe."""
    fp8_format = _aiter_native_fp8_format()
    return _aiter_mixed_attn_call(
        query,
        key,
        value,
        fp8_format,
        fp8_format,
        scale_modes=(
            _AiterAttentionScaleMode.E8M0_PER_1X32,
            _AiterAttentionScaleMode.E8M0_PER_1X32,
            _AiterAttentionScaleMode.F32_PER_TENSOR,
        ),
        softmax_scale=softmax_scale,
    )


def _aiter_forward_mxfp6(query, key, value, softmax_scale=None):
    """Run the AITER MXFP6 Q/K and FP8 V recipe."""
    return _aiter_mixed_attn_call(
        query,
        key,
        value,
        _AiterAttentionFormat.MXFP6,
        _aiter_native_fp8_format(),
        softmax_scale=softmax_scale,
    )


def _aiter_forward_mxfp4(query, key, value, softmax_scale=None):
    """Run the AITER MXFP4 Q/K/V recipe."""
    return _aiter_mixed_attn_call(
        query,
        key,
        value,
        _AiterAttentionFormat.MXFP4,
        _AiterAttentionFormat.MXFP4,
        softmax_scale=softmax_scale,
    )


def _aiter_forward_f8f6(query, key, value, softmax_scale=None):
    """Run the AITER per-tensor FP8 Q/K and MXFP6 V recipe."""
    return _aiter_mixed_attn_call(
        query,
        key,
        value,
        _aiter_native_fp8_format(),
        _AiterAttentionFormat.MXFP6,
        softmax_scale=softmax_scale,
    )


def _aiter_forward_f6f4(query, key, value, softmax_scale=None):
    """Run the AITER MXFP6 Q/K and MXFP4 V recipe."""
    return _aiter_mixed_attn_call(
        query,
        key,
        value,
        _AiterAttentionFormat.MXFP6,
        _AiterAttentionFormat.MXFP4,
        softmax_scale=softmax_scale,
    )


_FORWARD_FNS: dict[str, Callable[..., torch.Tensor]] = {
    "bf16": _aiter_forward_bf16,
    "fp8": _aiter_forward_fp8,
    "i8fp8": _aiter_forward_i8fp8,
    "mxfp8": _aiter_forward_mxfp8,
    "mxfp6": _aiter_forward_mxfp6,
    "mxfp4": _aiter_forward_mxfp4,
    "f8f6": _aiter_forward_f8f6,
    "f6f4": _aiter_forward_f6f4,
}


def get_forward_fn(format_name: str) -> Callable[..., torch.Tensor]:
    return _FORWARD_FNS[format_name.lower()]
