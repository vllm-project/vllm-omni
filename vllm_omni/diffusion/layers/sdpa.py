# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn.functional as F

from vllm_omni.diffusion.layers.custom_op import CustomOp


class ScaledDotProductAttention(CustomOp):
    """Platform-dispatched SDPA for BNSD query, key, and value tensors."""

    def __init__(self, causal: bool = False) -> None:
        super().__init__()
        self.causal = causal

    @staticmethod
    def _get_num_key_value_groups(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> int:
        num_heads = query.shape[1]
        num_key_value_heads = key.shape[1]
        num_value_heads = value.shape[1]
        if num_key_value_heads != num_value_heads:
            raise ValueError(
                "GQA requires key and value to have the same number of heads, "
                f"got k_heads={num_key_value_heads} and v_heads={num_value_heads}."
            )
        if num_key_value_heads == 0:
            raise ValueError("GQA requires at least one KV head.")
        if num_heads % num_key_value_heads != 0:
            raise ValueError(
                "GQA requires query heads to be a multiple of KV heads, "
                f"got q_heads={num_heads} and kv_heads={num_key_value_heads}."
            )
        return num_heads // num_key_value_heads

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward_native(query, key, value, attn_mask)

    def forward_hip(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward_native(query, key, value, attn_mask)

    def forward_npu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_key_value_groups = self._get_num_key_value_groups(query, key, value)
        enable_gqa = num_key_value_groups != 1
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=self.causal,
            enable_gqa=enable_gqa,
        )

    def forward_xpu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward_native(query, key, value, attn_mask)

    def forward_native(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_key_value_groups = self._get_num_key_value_groups(query, key, value)
        if num_key_value_groups != 1:
            key = key.repeat_interleave(num_key_value_groups, dim=1)
            value = value.repeat_interleave(num_key_value_groups, dim=1)
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=self.causal,
        )
