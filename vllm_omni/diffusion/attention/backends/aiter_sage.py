# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""SageAttention on AMD RDNA4 (gfx1201) through aiter.

aiter's ``gfx1201_sage_attention`` computes INT8 QK^T and FP8 PV with FP32
accumulation; Q/K are quantized per 32-row block and V per channel inside the
kernel. Like ``SAGE_ATTN`` it is lossy, so it is opt-in and never a default.
The kernel covers non-causal BF16 attention with head size 128 and equal
Q/K/V shapes.
"""

from __future__ import annotations

from dataclasses import replace

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)
from vllm_omni.diffusion.attention.capabilities import (
    CapabilityResult,
    CompilationMode,
    ExecutionContext,
    ExecutionPathResult,
    MaskMode,
    PackingMode,
    ParallelStrategy,
)

logger = init_logger(__name__)

_NAME = "AITER_SAGE"
_HEAD_SIZE = 128
# Architectures aiter ships SageAttention kernels for; aiter rejects others.
_SUPPORTED_ARCHS = ("gfx1201",)
_PACKED_KEYS = ("cu_seqlens_q", "cu_seqlens_k", "max_seqlen_q", "max_seqlen_k")


def _gcn_arch() -> str:
    return torch.cuda.get_device_properties().gcnArchName.split(":")[0]


def _sage_kernel():
    # Registered by aiter with a fake implementation, so it is opaque to torch.compile.
    return torch.ops.aiter.gfx1201_sage_attention


def _valid_kv_length(attn_metadata: AttentionMetadata | None) -> int | None:
    if attn_metadata is None:
        return None
    valid = attn_metadata.extra.get("valid_kv_length")
    return valid if isinstance(valid, int) else None


class AiterSageAttentionBackend(AttentionBackend):
    # A contiguous valid K/V prefix is sliced instead of masked.
    supports_prefix_kv_slicing: bool = True
    supported_platforms = ("rocm",)

    @classmethod
    def validate_available(cls) -> None:
        arch = _gcn_arch()
        if arch not in _SUPPORTED_ARCHS:
            raise ValueError(
                f"{_NAME} has kernels for {', '.join(_SUPPORTED_ARCHS)} only, but this GPU is {arch}. "
                "Select TORCH_SDPA or FLASH_ATTN instead."
            )
        try:
            import aiter
        except ImportError as exc:
            raise ValueError(f"{_NAME} requires aiter; install ROCm/aiter or select TORCH_SDPA.") from exc
        if not hasattr(aiter, "gfx1201_sage_attention"):
            raise ValueError(
                f"{_NAME} requires an aiter build with gfx1201_sage_attention; update aiter or select TORCH_SDPA."
            )

    @classmethod
    def resolve_capabilities(cls, context: ExecutionContext) -> ExecutionPathResult:
        # Support depends on the actual inputs; see AiterSageAttentionImpl.
        return ExecutionPathResult.unmigrated(cls.get_name(), context, path="uninitialized")

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [_HEAD_SIZE]

    @staticmethod
    def get_name() -> str:
        return _NAME

    @staticmethod
    def get_impl_cls() -> type[AiterSageAttentionImpl]:
        return AiterSageAttentionImpl


class AiterSageAttentionImpl(AttentionImpl):
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
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_size = head_size
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout = extra_impl_args.get("dropout_p", 0.0)
        if backend_kwargs:
            logger.warning("AiterSageAttentionImpl ignoring backend_kwargs: %s", list(backend_kwargs))

    def _unsupported_reason(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
    ) -> str | None:
        if self.causal:
            return "causal attention is not supported."
        if self.dropout != 0.0:
            return f"dropout is not supported (dropout_p={self.dropout})."
        if attn_metadata is not None:
            if attn_metadata.attn_mask is not None:
                return "attn_mask is not supported; only a valid K/V prefix (valid_kv_length) is."
            packed = any(name in attn_metadata.extra for name in _PACKED_KEYS)
            if attn_metadata.packed_padding is not None or (packed and _valid_kv_length(attn_metadata) is None):
                return "packed multi-document inputs are not supported."
            if attn_metadata.full_attn_spans is not None:
                return "piecewise attention spans are not supported."
        if any(t.ndim != 4 for t in (query, key, value)):
            return "Q, K, and V must have rank 4 (batch, sequence, heads, head dimension)."
        if any(t.dtype != torch.bfloat16 for t in (query, key, value)):
            return "Q, K, and V must be BF16."
        if key.shape != query.shape or value.shape != query.shape:
            return "Q, K, and V must have equal shapes (no cross-attention or GQA)."
        if query.shape[-1] != _HEAD_SIZE:
            return f"head size must be {_HEAD_SIZE}, got {query.shape[-1]}."
        if any(size == 0 for size in query.shape):
            return "Q, K, and V dimensions must be nonzero."
        valid = _valid_kv_length(attn_metadata)
        if valid is not None and not 0 < valid <= key.shape[1]:
            return f"valid_kv_length must be within the K/V sequence, got {valid} for length {key.shape[1]}."
        return None

    def resolve_execution_path(
        self,
        context: ExecutionContext,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
    ) -> ExecutionPathResult:
        context = replace(
            context,
            kernel_variant="gfx1201" if context.platform == "rocm" else None,
            dtype=str(query.dtype).removeprefix("torch."),
            causal=self.causal,
            mask_mode=MaskMode.NONE,
            packing_mode=PackingMode.NONE,
        )
        result = ExecutionPathResult.unmigrated(_NAME, context, path="sage_dense")
        if context.platform != "rocm":
            return replace(result, support=CapabilityResult.unsupported(f"{_NAME} runs on ROCm only."))
        reason = self._unsupported_reason(query, key, value, attn_metadata)
        if reason:
            return replace(
                result,
                support=CapabilityResult.unsupported(f"{_NAME}: {reason} Select another backend."),
            )
        if (
            context.piecewise
            or context.paged_kv
            or context.kv_cache_dtype not in (None, "auto")
            or context.parallel_strategy is not ParallelStrategy.NONE
            or context.outer_boundaries
        ):
            return replace(
                result,
                support=CapabilityResult.unmigrated(
                    "Only dense single-device AITER_SAGE without paged KV, parallel, or outer boundaries is verified."
                ),
            )
        return replace(result, support=CapabilityResult.supported(), compilation_mode=CompilationMode.CUSTOM_OP)

    def forward_hip(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        reason = self._unsupported_reason(query, key, value, attn_metadata)
        if reason:
            raise ValueError(f"{_NAME}: {reason}")
        kernel = _sage_kernel()
        valid = _valid_kv_length(attn_metadata)
        if valid is None or valid == key.shape[1]:
            return kernel(query.contiguous(), key.contiguous(), value.contiguous(), self.softmax_scale)
        # Rows past the valid prefix are padding: they are never attended to as
        # keys, and their outputs are discarded, so they come back as zeros.
        output = torch.zeros(query.shape, dtype=query.dtype, device=query.device)
        output[:, :valid] = kernel(
            query[:, :valid].contiguous(),
            key[:, :valid].contiguous(),
            value[:, :valid].contiguous(),
            self.softmax_scale,
        )
        return output

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError(f"{_NAME} runs on ROCm only.")
