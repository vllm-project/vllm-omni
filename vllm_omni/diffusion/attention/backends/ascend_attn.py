# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from importlib.util import find_spec
import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)

logger = init_logger(__name__)


class AscendAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [x for x in range(1024)]  # todo

    @staticmethod
    def get_name() -> str:
        return "ASCEND"

    @staticmethod
    def get_impl_cls() -> type["AscendAttentionBackendImpl"]:
        return AscendAttentionBackendImpl


class AscendAttentionBackendImpl(AttentionImpl):
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

    def forward_native(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata = None,
    ) -> torch.Tensor:
        query, key, value = (x.permute(0, 2, 1, 3) for x in (query, key, value))
        attention_mask = attn_metadata.attn_mask if attn_metadata else None

        output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=self.causal,
            scale=self.softmax_scale,
        )
        out = output.permute(0, 2, 1, 3)
        return out

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata = None,
    ) -> torch.Tensor:
        if os.environ.get("ENABLE_MINDIE_SD", "").lower() in ("true", "1") and find_spec("mindiesd"):
            from mindiesd import attention_forward
            attention_mask = attn_metadata.attn_mask if attn_metadata else None
            output = attention_forward(
                query, 
                key, 
                value, 
                attn_mask=attention_mask, 
                opt_mode="manual", 
                op_type="fused_attn_score", 
                layout="BNSD"
            )
        else:
            return self.forward_native(query, key, value, attn_metadata)

        return output