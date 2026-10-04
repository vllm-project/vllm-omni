# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""AITER MHA v4 quantized diffusion attention backend."""

import torch

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)
from vllm_omni.diffusion.attention.backends.utils.aiter_mha_v4 import (
    get_forward_fn,
    require_mha_v4,
)
from vllm_omni.diffusion.config import get_current_diffusion_config_or_none
from vllm_omni.platforms import current_omni_platform

_REQUIRED_HEAD_DIM = 128
_DEFAULT_FORMAT = "fp8"
_FORMATS_BY_ARCH = {
    "gfx942": frozenset({"fp8", "i8fp8"}),
    "gfx950": frozenset({"bf16", "f6f4", "fp8", "i8fp8", "mxfp4", "mxfp6", "mxfp8"}),
}
_ALL_FORMATS = frozenset(
    format_name for supported_formats in _FORMATS_BY_ARCH.values() for format_name in supported_formats
)
_SUPPORTED_LAYOUTS = frozenset({"BSND", "BSHD"})


class AiterQuantBackend(AttentionBackend):
    supported_platforms = ("rocm",)

    @classmethod
    def validate_available(cls) -> None:
        require_mha_v4()

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [_REQUIRED_HEAD_DIM]

    @staticmethod
    def get_name() -> str:
        return "AITER_QUANT_ATTN"

    @staticmethod
    def get_impl_cls() -> type["AiterQuantImpl"]:
        return AiterQuantImpl

    @staticmethod
    def get_metadata_cls() -> type[AttentionMetadata]:
        return AttentionMetadata

    @staticmethod
    def get_builder_cls():
        raise NotImplementedError("AITER_QUANT_ATTN does not use a metadata builder.")


class AiterQuantImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        qkv_layout: str | None = None,
        backend_kwargs: dict | None = None,
        **extra_impl_args,
    ) -> None:
        options = backend_kwargs or {}
        format_name = str(options.get("format", _DEFAULT_FORMAT)).lower()
        if format_name not in _ALL_FORMATS:
            raise ValueError(f"Unknown AITER quant format {format_name!r}; expected one of {sorted(_ALL_FORMATS)}.")
        gfx_arch = current_omni_platform.get_gfx_arch()
        supported_formats = _FORMATS_BY_ARCH.get(gfx_arch) if gfx_arch is not None else None
        if supported_formats is None:
            raise RuntimeError(
                f"AITER_QUANT_ATTN requires a gfx942 or gfx950 ROCm GPU; detected {gfx_arch or 'unknown'}."
            )
        if format_name not in supported_formats:
            raise RuntimeError(
                f"AITER_QUANT_ATTN format={format_name!r} is not available on {gfx_arch}; "
                f"use one of {sorted(supported_formats)}."
            )
        if causal:
            raise NotImplementedError("AITER_QUANT_ATTN does not support causal attention.")
        if head_size != _REQUIRED_HEAD_DIM:
            raise NotImplementedError(f"AITER_QUANT_ATTN requires head_dim={_REQUIRED_HEAD_DIM}; got {head_size}.")
        kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        if num_heads <= 0 or kv_heads <= 0:
            raise ValueError(
                "AITER_QUANT_ATTN requires positive query and KV head counts; "
                f"got num_heads={num_heads}, num_kv_heads={kv_heads}."
            )
        if num_heads % kv_heads != 0:
            raise ValueError(
                "AITER_QUANT_ATTN requires query heads to be divisible by KV heads; "
                f"got num_heads={num_heads}, num_kv_heads={kv_heads}."
            )
        gqa_ratio = num_heads // kv_heads
        if gqa_ratio not in (1, 2, 4, 8, 16):
            raise ValueError(f"AITER_QUANT_ATTN supports GQA ratios 1, 2, 4, 8, and 16; got ratio={gqa_ratio}.")
        if qkv_layout is not None and qkv_layout.upper() not in _SUPPORTED_LAYOUTS:
            raise ValueError(
                f"AITER_QUANT_ATTN expects [B, S, H, D] tensors (BSND/BSHD), not qkv_layout={qkv_layout!r}."
            )
        config = get_current_diffusion_config_or_none()
        if config is not None and config.dtype not in (torch.float16, torch.bfloat16):
            raise TypeError(f"AITER_QUANT_ATTN requires float16 or bfloat16 model inputs; got dtype={config.dtype}.")

        AiterQuantBackend.validate_available()
        self.format = format_name
        self.qkv_layout = qkv_layout
        self.softmax_scale = softmax_scale
        self.causal = causal
        self._forward = get_forward_fn(format_name)

    def forward_hip(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        if attn_metadata is not None and attn_metadata.attn_mask is not None:
            raise NotImplementedError("AITER_QUANT_ATTN does not support attention masks.")
        return self._forward(
            query,
            key,
            value,
            softmax_scale=self.softmax_scale,
            causal=self.causal,
        )
