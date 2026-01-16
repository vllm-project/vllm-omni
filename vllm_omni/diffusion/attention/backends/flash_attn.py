# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)
from vllm_omni.diffusion.attention.backends.utils.fa import _pad_input, _unpad_input, _upad_input

logger = init_logger(__name__)

try:
    # only tested with flash_attn v3
    # from flash_attn_interface import flash_attn_func as flash_attn_3_func  # not available in flash-attn 2.8.1
    from flash_attn import flash_attn_func, flash_attn_varlen_func  # can be FA2 or FA3
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
        """
        Flash attention implementation.

        Args:
            query: (batch_size, seq_len, num_heads, head_dim)
            key: (batch_size, seq_len, num_heads, head_dim)
            value: (batch_size, seq_len, num_heads, head_dim)
            attn_metadata: AttentionMetadata. Attention mask is supported as attn_metadata.attn_mask

        Returns:
            (batch_size, seq_len, num_heads, head_dim)
        """
        query_length = query.size(1)
        attention_mask = attn_metadata.attn_mask if attn_metadata is not None else None
        #  Contains at least one padding token in the sequence
        if attention_mask is not None and torch.any(~attention_mask):
            assert attention_mask.ndim == 2, "attention_mask must be 2D, (batch_size, seq_len)"
            q, k, v, indices_q, (cu_seq_lens_q, cu_seq_lens_k), (max_length_q, max_length_k) = _upad_input(
                query, key, value, attention_mask, query_length, _unpad_input
            )

            out_unpad = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=cu_seq_lens_q,
                cu_seqlens_k=cu_seq_lens_k,
                max_seqlen_q=max_length_q,
                max_seqlen_k=max_length_k,
                **{
                    "causal": self.causal,
                    "softmax_scale": self.softmax_scale,
                },
            )
            if isinstance(out_unpad, tuple):
                out_unpad = out_unpad[0]

            out = _pad_input(out_unpad, indices_q, query.size(0), query_length)

        else:
            out: torch.Tensor = flash_attn_func(
                query,
                key,
                value,
                causal=self.causal,
                softmax_scale=self.softmax_scale,
            )
        return out
