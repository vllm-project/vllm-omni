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


class FlashAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @classmethod
    def supports_attention_mask(cls) -> bool:
        return True

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

    @staticmethod
    def _unwrap_flash_output(out: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
        # FA3 may return (out, lse), FA2 returns out
        return out[0] if isinstance(out, tuple) else out

    def _forward_varlen_masked(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        from vllm_omni.diffusion.attention.backends.utils.fa import (
            _pad_input,
            _unpad_input,
            _upad_input,
            flash_attn_varlen_func,
        )

        assert attention_mask.ndim == 2, "attention_mask must be 2D, (batch_size, seq_len)"
        query_length = query.size(1)
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
        out_unpad = self._unwrap_flash_output(out_unpad)
        return _pad_input(out_unpad, indices_q, query.size(0), query_length)

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata = None,
    ) -> torch.Tensor:
        """CUDA/ROCm flash attention implementation."""
        from vllm_omni.diffusion.attention.backends.utils.fa import (
            HAS_FLASH_ATTN,
            flash_attn_func,
        )

        if not HAS_FLASH_ATTN:
            raise ImportError(
                "FlashAttentionBackend requires Flash Attention. "
                "Please install one of: fa3-fwd, flash-attention, or flash-attn. "
                "Otherwise, use SDPA backend by setting DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA"
            )

        attention_mask = attn_metadata.attn_mask if attn_metadata is not None else None

        if attention_mask is not None and torch.any(~attention_mask):
            return self._forward_varlen_masked(
                query,
                key,
                value,
                attention_mask,
            )

        out = flash_attn_func(
            query,
            key,
            value,
            causal=self.causal,
            softmax_scale=self.softmax_scale,
        )
        return self._unwrap_flash_output(out)

    def forward_xpu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata = None,
    ) -> torch.Tensor:
        """XPU flash attention implementation."""
        from vllm_omni.diffusion.attention.backends.utils.fa import (
            HAS_FLASH_ATTN,
            flash_attn_varlen_func,
        )

        if not HAS_FLASH_ATTN:
            raise ImportError(
                "FlashAttentionBackend requires Flash Attention. "
                "Please assure vllm-xpu-kernels properly installed. "
                "Otherwise, use SDPA backend by setting DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA"
            )

        attention_mask = attn_metadata.attn_mask if attn_metadata is not None else None

        if attention_mask is not None and torch.any(~attention_mask):
            return self._forward_varlen_masked(
                query,
                key,
                value,
                attention_mask,
            )

        batch_size, q_len = query.size()[:2]
        cu_seqlens = torch.arange(0, (batch_size + 1) * q_len, step=q_len, dtype=torch.int32, device=query.device)
        # b s ... -> (b s) ...
        query = query.flatten(0, 1)
        key = key.flatten(0, 1)
        value = value.flatten(0, 1)

        out = flash_attn_varlen_func(
            query,
            key,
            value,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=q_len,
            max_seqlen_k=q_len,
            causal=self.causal,
            softmax_scale=self.softmax_scale,
        )
        out = self._unwrap_flash_output(out)
        # (b s) h d -> b s h d
        return out.reshape(batch_size, q_len, *out.shape[1:])

    def forward_npu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata = None,
    ) -> torch.Tensor:
        """NPU attention implementation using mindiesd."""
        try:
            from mindiesd import attention_forward
        except ImportError:
            raise ImportError(
                "FlashAttentionBackend NPU implementation requires MindIE-SD. "
                "Please install MindIE-SD to enable NPU attention support. "
                "For installation details, see https://gitcode.com/Ascend/MindIE-SD"
                "Otherwise, use SDPA backend by setting DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA"
            )

        attention_mask = attn_metadata.attn_mask if attn_metadata else None
        output = attention_forward(
            query,
            key,
            value,
            attn_mask=attention_mask,
            opt_mode="manual",
            op_type="fused_attn_score",
            layout="BNSD",
        )
        return output
