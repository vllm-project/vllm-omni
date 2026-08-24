# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)

_SAGE_ATTN_TENSOR_LAYOUT = os.environ.get("SAGE_ATTN_TENSOR_LAYOUT", "NHD").upper()
assert _SAGE_ATTN_TENSOR_LAYOUT in ("NHD", "HND"), (
    f"SAGE_ATTN_TENSOR_LAYOUT must be 'NHD' or 'HND', got '{_SAGE_ATTN_TENSOR_LAYOUT}'"
)

if current_omni_platform.is_xpu():
    try:
        import inspect

        from auto_round_kernel import ARK

        _ark = ARK()
        xpu_sageattn = _ark.sagev1
        _sagev1_params = inspect.signature(xpu_sageattn).parameters
        _sagev1_has_tensor_layout = "tensor_layout" in _sagev1_params
        _sagev1_scale_param = "sm_scale" if "sm_scale" in _sagev1_params else "scale"
    except ImportError:
        logger.warning(
            "XPU SageAttention (auto_round_kernel.ARK.sagev1) is not available. "
            "Install auto-round-lib for XPU sage attention support."
        )
        xpu_sageattn = None
        _sagev1_has_tensor_layout = False
        _sagev1_scale_param = "scale"
else:
    try:
        from sageattention import sageattn
    except ImportError:
        logger.warning(
            "SageAttentionBackend is not available. You may install sage-attention"
            " by pip install git+https://github.com/thu-ml/SageAttention.git"
        )
        raise ImportError

# TODO add sage3 attention backend


class SageAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_name() -> str:
        return "SAGE_ATTN"

    @staticmethod
    def get_impl_cls() -> type["SageAttentionImpl"]:
        return SageAttentionImpl


class SageAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        backend_kwargs: dict | None = None,
        **extra_impl_args,
    ) -> None:
        self.causal = causal
        self.softmax_scale = softmax_scale
        if backend_kwargs:
            logger.warning("SageAttentionImpl ignoring backend_kwargs: %s", list(backend_kwargs.keys()))

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata = None,
    ) -> torch.Tensor:
        output = sageattn(
            query,
            key,
            value,
            tensor_layout="NHD",
            is_causal=self.causal,
            sm_scale=self.softmax_scale,
        )
        return output

    def forward_xpu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata = None,
    ) -> torch.Tensor:
        if xpu_sageattn is None:
            raise ImportError("XPU SageAttention requires auto-round-lib. Install with: pip install auto-round-lib")
        orig_dtype = query.dtype
        q = query.to(torch.float16) if orig_dtype != torch.float16 else query
        k = key.to(torch.float16) if orig_dtype != torch.float16 else key
        v = value.to(torch.float16) if orig_dtype != torch.float16 else value

        if _sagev1_has_tensor_layout:
            if _SAGE_ATTN_TENSOR_LAYOUT == "HND":
                q = q.transpose(1, 2).contiguous()
                k = k.transpose(1, 2).contiguous()
                v = v.transpose(1, 2).contiguous()
            output = xpu_sageattn(
                q,
                k,
                v,
                tensor_layout=_SAGE_ATTN_TENSOR_LAYOUT,
                is_causal=self.causal,
                **{_sagev1_scale_param: self.softmax_scale},
            )
            if _SAGE_ATTN_TENSOR_LAYOUT == "HND":
                output = output.transpose(1, 2).contiguous()
        else:
            # No tensor_layout support: kernel expects HND [B, H, S, D]
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()
            output = xpu_sageattn(
                q,
                k,
                v,
                is_causal=self.causal,
                **{_sagev1_scale_param: self.softmax_scale},
            )
            output = output.transpose(1, 2).contiguous()

        if orig_dtype != torch.float16:
            output = output.to(orig_dtype)
        return output
