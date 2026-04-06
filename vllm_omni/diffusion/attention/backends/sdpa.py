# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Literal

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)

logger = init_logger(__name__)


SDPAMaskMode = Literal["broadcast_k", "full_qk"]


def _maybe_reshape_attn_mask(
    query: torch.Tensor,
    key: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    mask_mode: SDPAMaskMode = "broadcast_k",
):
    """
    Reshape Attention Mask
    2D [batch_size, seq_len_k] ->
      - broadcast_k: [batch_size, 1, 1, seq_len_k]
      - full_qk: [batch_size, 1, seq_len_q, seq_len_k]
    """
    # Skip Attention Mask if all values are 1, `None` mask can speedup the computation
    if attn_mask is not None and torch.all(attn_mask != 0):
        attn_mask = None

    # Reshape Attention Mask
    # 2D [batch_size, seq_len_k] mask only.
    if (
        attn_mask is not None
        and attn_mask.ndim == 2
        and attn_mask.shape[0] == query.shape[0]
        and attn_mask.shape[1] == key.shape[1]
    ):
        B, Sq, Skv = attn_mask.shape[0], query.shape[1], key.shape[1]
        attn_mask = attn_mask.to(torch.bool)
        if mask_mode == "full_qk":
            # NPU path requires explicit [B, 1, Q, K] mask layout.
            attn_mask = attn_mask.unsqueeze(1).expand(B, Sq, Skv).unsqueeze(1).contiguous()
        elif mask_mode == "broadcast_k":
            # CUDA-like backends prefer [B, 1, 1, K] and rely on SDPA broadcast.
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
        else:
            raise ValueError(f"Unsupported SDPA mask mode: {mask_mode}")
    return attn_mask


class SDPABackend(AttentionBackend):
    accept_output_buffer: bool = True

    @classmethod
    def supports_attention_mask(cls) -> bool:
        return True

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

    def _forward_impl(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
        mask_mode: SDPAMaskMode = "broadcast_k",
    ) -> torch.Tensor:
        # Normalize mask before permuting q/k/v.
        # _maybe_reshape_attn_mask expects sequence length on dim=1.
        attention_mask = None
        if attn_metadata:
            attention_mask = _maybe_reshape_attn_mask(query, key, attn_metadata.attn_mask, mask_mode=mask_mode)

        query, key, value = (x.permute(0, 2, 1, 3) for x in (query, key, value))
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

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        return self._forward_impl(query, key, value, attn_metadata, mask_mode="broadcast_k")

    def forward_xpu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        return self._forward_impl(query, key, value, attn_metadata, mask_mode="broadcast_k")

    def forward_hip(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        return self._forward_impl(query, key, value, attn_metadata, mask_mode="broadcast_k")

    def forward_npu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        return self._forward_impl(query, key, value, attn_metadata, mask_mode="full_qk")
