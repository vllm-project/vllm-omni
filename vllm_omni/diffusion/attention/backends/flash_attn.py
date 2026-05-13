# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import partial

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)
from vllm_omni.diffusion.attention.backends.utils.piecewise_attn import (
    piecewise_attn,
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
    # Per-platform FP8 KV quantization support.
    # To enable FP8 on a new platform, add its OmniPlatformEnum value here
    # and handle kv_cache_dtype in the corresponding forward_{platform}().
    #
    # TODO(quant-backend): The FP8 quant path currently lives inside
    # FlashAttentionImpl gated by ``attn_metadata.extra["kv_cache_dtype"]``.
    # Eventually extract it into a dedicated FlashAttentionQuantBackend so
    # backend selection (not metadata) decides quant. Until then, model
    # authors can opt a specific Attention layer out via
    # ``Attention(disable_kv_quant=True)``.
    _supported_kv_cache_dtypes = {
        "npu": {"fp8"},
    }

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
        self.num_heads = num_heads
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.qkv_layout = qkv_layout
        if backend_kwargs:
            logger.warning("FlashAttentionImpl ignoring backend_kwargs: %s", list(backend_kwargs.keys()))

    @staticmethod
    def _unwrap_flash_output(out: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
        # FA3 may return (out, lse), FA2 returns out
        return out[0] if isinstance(out, tuple) else out

    @staticmethod
    def _flash_wrapper(q, k, v, *, attn_func, **kwargs):
        return FlashAttentionImpl._unwrap_flash_output(attn_func(q, k, v, **kwargs))

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

    def _forward_varlen_dense(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Common wrapper for calling flash_attn_varlen_func for XPU and CUDA in vLLM.

        NOTE: careful to keep the kwargs on these aligned and to pass everything as a keyword
        argument, because some of the args differ positionally at the moment.

        https://github.com/vllm-project/vllm/blob/v0.20.0/vllm/vllm_flash_attn/flash_attn_interface.py#L176
        https://github.com/vllm-project/vllm/blob/v0.20.0/vllm/_xpu_ops.py#L310
        """
        from vllm_omni.diffusion.attention.backends.utils.fa import (
            flash_attn_varlen_func,
        )

        batch_size, q_len = query.size()[:2]
        cu_seqlens = torch.arange(0, (batch_size + 1) * q_len, step=q_len, dtype=torch.int32, device=query.device)
        # b s ... -> (b s) ...
        query = query.flatten(0, 1)
        key = key.flatten(0, 1)
        value = value.flatten(0, 1)

        out = flash_attn_varlen_func(
            q=query,
            k=key,
            v=value,
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

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata = None,
    ) -> torch.Tensor:
        """CUDA/ROCm/MUSA flash attention implementation."""
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
        full_attn_spans = attn_metadata.full_attn_spans if attn_metadata is not None else None

        # Try piecewise attention
        if full_attn_spans is not None:
            logger.debug("Using piecewise Flash Attention for mixed causal/full mask")
            attn_func = partial(
                FlashAttentionImpl._flash_wrapper,
                attn_func=flash_attn_func,
            )

            return piecewise_attn(
                query,
                key,
                value,
                full_attn_spans,
                self.softmax_scale,
                attn_func,
            )

        if attention_mask is not None and torch.any(~attention_mask):
            return self._forward_varlen_masked(
                query,
                key,
                value,
                attention_mask,
            )

        if flash_attn_func is not None:
            out = flash_attn_func(
                query,
                key,
                value,
                causal=self.causal,
                softmax_scale=self.softmax_scale,
            )
            return self._unwrap_flash_output(out)

        return self._forward_varlen_dense(
            query,
            key,
            value,
        )

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

        return self._forward_varlen_dense(
            query,
            key,
            value,
        )

    def forward_npu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata = None,
    ) -> torch.Tensor:
        """NPU attention implementation using mindiesd."""

        kv_cache_dtype = attn_metadata.extra.get("kv_cache_dtype") if attn_metadata else None
        if kv_cache_dtype is not None:
            return self.forward_fa_quant_npu(query, key, value, attn_metadata)
        return self.forward_fa_npu(query, key, value, attn_metadata)

    def forward_fa_quant_npu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata = None,
    ) -> torch.Tensor:
        from vllm_omni.platforms.npu.quant.kv_quant_npu import fp8_rotate_quant_fa

        layout = self.qkv_layout or "BNSD"
        # Models pass (B, S, H, D); NPU fused op expects (B, N, S, D).
        out = fp8_rotate_quant_fa(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            layout=layout,
            softmax_scale=self.softmax_scale,
        )
        return out.transpose(1, 2)

    def forward_fa_npu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata = None,
    ) -> torch.Tensor:
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
        layout = self.qkv_layout or "BNSD"
        return attention_forward(
            query,
            key,
            value,
            attn_mask=attention_mask,
            opt_mode="manual",
            op_type="fused_attn_score",
            layout=layout,
        )
