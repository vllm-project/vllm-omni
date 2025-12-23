# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)

logger = init_logger(__name__)

try:
    # only tested with flash_attn v3
    # from flash_attn_interface import flash_attn_func as flash_attn_3_func  # not available in flash-attn 2.8.1
    from flash_attn import flash_attn_func  # can be FA2 or FA3
except ImportError:
    logger.warning(
        "FlashAttentionBackend is not available. You may install flash-attn "
        "by running `uv pip install flash-attn==2.8.1 --no-build-isolation`"
        " or install pre-built flash-attn from https://github.com/Dao-AILab/flash-attention/releases"
    )
    raise ImportError


class FlashAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 96, 128, 192, 256]

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN"

    @staticmethod
    def get_impl_cls() -> type["FlashAttentionImpl"]:
        return FlashAttentionImpl


class FlashAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        self.num_heads = num_heads
        self.causal = causal
        self.softmax_scale = softmax_scale

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata = None,
    ) -> torch.Tensor:
        # TODO: flash_attn_func does not support attn_mask.
        out: torch.Tensor = flash_attn_func(
            query,
            key,
            value,
            causal=self.causal,
            softmax_scale=self.softmax_scale,
        )
        return out
