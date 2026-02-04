# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2024, Jiarui Fang.
# Adapted from https://github.com/feifeibear/long-context-attention

from collections.abc import Callable
from enum import Enum
from functools import partial

import torch

from .ring_globals import (
    HAS_SAGE_ATTENTION,
    HAS_SPARSE_SAGE_ATTENTION,
)
from .ring_kernels import (
    flash_attn3_func_forward,
    flash_attn_forward,
    flash_attn_forward_aiter,
    flashinfer_attn_forward,
    pytorch_attn_forward,
)

if HAS_SAGE_ATTENTION:
    import sageattention

if HAS_SPARSE_SAGE_ATTENTION:
    from spas_sage_attn.autotune import SparseAttentionMeansim


class AttnType(Enum):
    AITER = "aiter"
    FA = "fa"
    FA3 = "fa3"
    FLASHINFER = "flashinfer"
    TORCH = "torch"
    SAGE_AUTO = "sage_auto"
    SAGE_FP16 = "sage_fp16"
    SAGE_FP16_TRITON = "sage_fp16_triton"
    SAGE_FP8 = "sage_fp8"
    SAGE_FP8_SM90 = "sage_fp8_sm90"
    SPARSE_SAGE = "sparse_sage"

    @classmethod
    def from_string(cls, s: str):
        for member in cls:
            if member.value == s:
                return member
        raise ValueError(f"'{s}' is not a valid {cls.__name__}")


def select_flash_attn_impl(
    impl_type: AttnType,
    stage: str = "fwd-only",
    attn_processor: torch.nn.Module | None = None,
) -> Callable[..., tuple[torch.Tensor, torch.Tensor | None]]:
    """Select attention implementation for forward pass (inference only).

    Args:
        impl_type: The attention implementation type.
        stage: Must be "fwd-only" (backward not supported for inference).
        attn_processor: Optional custom attention processor.

    Returns:
        Callable[..., tuple[torch.Tensor, torch.Tensor | None]]: The attention
            forward function for the specified implementation.
    """
    if stage != "fwd-only":
        raise ValueError(f"Only 'fwd-only' stage is supported for inference. Got: {stage}")

    if impl_type == AttnType.AITER:
        return flash_attn_forward_aiter

    elif impl_type == AttnType.FA:
        return flash_attn_forward

    elif impl_type == AttnType.FA3:
        return flash_attn3_func_forward

    elif impl_type == AttnType.FLASHINFER:
        return flashinfer_attn_forward

    elif impl_type == AttnType.TORCH:
        return pytorch_attn_forward

    elif impl_type == AttnType.SAGE_AUTO:
        if not HAS_SAGE_ATTENTION:
            raise ImportError("SageAttention is not available!")
        return partial(
            sageattention.sageattn,
            tensor_layout="NHD",
            return_lse=True,
        )

    elif impl_type == AttnType.SAGE_FP16:
        if not HAS_SAGE_ATTENTION:
            raise ImportError("SageAttention is not available!")
        return partial(
            sageattention.sageattn_qk_int8_pv_fp16_cuda,
            pv_accum_dtype="fp32",
            tensor_layout="NHD",
            return_lse=True,
        )

    elif impl_type == AttnType.SAGE_FP16_TRITON:
        if not HAS_SAGE_ATTENTION:
            raise ImportError("SageAttention is not available!")
        return partial(
            sageattention.sageattn_qk_int8_pv_fp16_triton,
            tensor_layout="NHD",
            return_lse=True,
        )

    elif impl_type == AttnType.SAGE_FP8:
        if not HAS_SAGE_ATTENTION:
            raise ImportError("SageAttention is not available!")
        return partial(
            sageattention.sageattn_qk_int8_pv_fp8_cuda,
            pv_accum_dtype="fp32+fp32",
            tensor_layout="NHD",
            return_lse=True,
        )

    elif impl_type == AttnType.SAGE_FP8_SM90:
        if not HAS_SAGE_ATTENTION:
            raise ImportError("SageAttention is not available!")
        return partial(
            sageattention.sageattn_qk_int8_pv_fp8_cuda_sm90,
            pv_accum_dtype="fp32+fp32",
            tensor_layout="NHD",
            return_lse=True,
        )

    elif impl_type == AttnType.SPARSE_SAGE:
        if not HAS_SPARSE_SAGE_ATTENTION:
            raise ImportError("SparseSageAttention is not available!")
        if not isinstance(attn_processor, SparseAttentionMeansim):
            raise ImportError("SparseSageAttention is only available with a SparseAttentionProcessor class passed in")

        def fn(q, k, v, causal=False, softmax_scale=None, *args, **kwargs):
            return (
                attn_processor(
                    q,
                    k,
                    v,
                    is_causal=causal,
                    scale=softmax_scale,
                    tensor_layout="NHD",
                ),
                None,
            )

        return fn

    elif attn_processor is not None:
        return attn_processor

    else:
        raise ValueError(f"Unknown flash attention implementation: {impl_type}")
