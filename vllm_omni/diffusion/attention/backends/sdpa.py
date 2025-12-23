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


class SDPABackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [x for x in range(1024)]  # todo

    @staticmethod
    def get_name() -> str:
        return "SDPA"

    @staticmethod
    def get_impl_cls() -> type["SDPAImpl"]:
        return SDPAImpl


class SDPAImpl(AttentionImpl):
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

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata = None,
    ) -> torch.Tensor:
        query, key, value = (x.permute(0, 2, 1, 3) for x in (query, key, value))
        attention_mask = attn_metadata.attn_mask if attn_metadata else None
        if attention_mask is not None:
            if attention_mask.ndim == 2:
                # self-attention mask
                bs, seq_len = attention_mask.shape
                assert seq_len == query.shape[-2], (
                    f"attention mask seq_len != query.shape[-2], {seq_len} != {query.shape[-2]}"
                )
                attention_mask = torch.expand(
                    attention_mask.unsqueeze(1).unsqueeze(1), (bs, 1, seq_len, seq_len)
                )  # (bs, 1, seq_len, seq_len)
            elif attention_mask.ndim == 4:
                pass
            else:
                raise ValueError(f"Invalid attention mask dimension: {attention_mask.ndim}")

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
