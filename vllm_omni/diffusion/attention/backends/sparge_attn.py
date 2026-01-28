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

logger = init_logger(__name__)

try:
    from spas_sage_attn import spas_sage2_attn_meansim_topk_cuda
except ImportError:
    logger.warning(
        "SpargeAttentionBackend is not available. You may install spas_sage_attn"
        " by pip install git+https://github.com/thu-ml/SpargeAttn.git@bfd980b781784c04ad6a53e7ee657c0645d99171"
    )
    raise ImportError


class SpargeAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 128]

    @staticmethod
    def get_name() -> str:
        return "SPARGE_ATTN"

    @staticmethod
    def get_impl_cls() -> type["SpargeAttentionImpl"]:
        return SpargeAttentionImpl


class SpargeAttentionImpl(AttentionImpl):
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
        self.causal = causal
        self.softmax_scale = softmax_scale

        # TODO Add this to attention configuration when available
        sparge_attn_topk = os.environ.get("SPARGE_ATTN_TOPK")
        self.topk = float(sparge_attn_topk) if sparge_attn_topk is not None else 0.5

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata = None,
    ) -> torch.Tensor:
        output = spas_sage2_attn_meansim_topk_cuda(
            query,
            key,
            value,
            topk=self.topk,
            tensor_layout="NHD",
            is_causal=self.causal,
            scale=self.softmax_scale,
        )
        return output
