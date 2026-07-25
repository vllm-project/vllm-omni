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
from vllm_omni.diffusion.attention.backends.utils.attn_runtime_selector import can_sdpa_use_fused_gqa

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
        backend_kwargs: dict | None = None,
        **extra_impl_args,
    ) -> None:
        self.causal = causal
        self.softmax_scale = softmax_scale
        if backend_kwargs:
            logger.warning("SDPAImpl ignoring backend_kwargs: %s", list(backend_kwargs.keys()))

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

        enable_gqa = query.shape[2] != key.shape[2]
        query, key, value = (x.permute(0, 2, 1, 3) for x in (query, key, value))
        # Only the PyTorch SDPA backend needs this dispatch check. If SDPA
        # cannot select a fused GQA kernel for the runtime shape/mask, expand
        # K/V locally so it can use the better-supported equal-head path.
        if enable_gqa and not can_sdpa_use_fused_gqa(query, key, value, attention_mask, self.causal):
            if query.shape[1] % key.shape[1] != 0:
                raise ValueError(
                    "GQA requires query heads to be a multiple of KV heads, "
                    f"got q_heads={query.shape[1]} and kv_heads={key.shape[1]}."
                )
            repeat_num = query.shape[1] // key.shape[1]
            key = key.repeat_interleave(repeat_num, dim=1)
            value = value.repeat_interleave(repeat_num, dim=1)
            enable_gqa = False
            logger.debug(
                "CUDA SDPA cannot use a fused native-GQA kernel for this shape; expanding K/V heads before SDPA."
            )
        output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=self.causal,
            scale=self.softmax_scale,
            enable_gqa=enable_gqa,
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
