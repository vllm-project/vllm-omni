# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from dataclasses import replace
from typing import Any

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

_MISSING_AUTO_ROUND = (
    "SPARGE_ATTN requires auto-round-lib with the ARK XPU sparse attention kernel. "
    "Install it with `pip install auto-round-lib`."
)
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)
_SUPPORTED_HEAD_SIZES = (64, 128)


if not hasattr(torch.ops.vllm_omni, "sparge_attention"):

    @torch.library.custom_op("vllm_omni::sparge_attention", mutates_args=())
    def _sparge_attention_op(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        is_causal: bool,
        scale: float,
        topk: float,
        q_tile_override: int,
        sparse_q_block_tokens: int,
        sparse_k_block_tokens: int,
    ) -> torch.Tensor:
        import auto_round_kernel as ark

        output = ark.sparge_sage2_attn_meansim_topk_xpu(
            query,
            key,
            value,
            is_causal=is_causal,
            scale=scale,
            smooth_k=True,
            simthreshd1=-1.0,
            topk=topk,
            attention_sink=False,
            tensor_layout="NHD",
            q_tile_override=q_tile_override,
            sparse_q_block_tokens=sparse_q_block_tokens,
            sparse_k_block_tokens=sparse_k_block_tokens,
        )
        return output.contiguous()

    @_sparge_attention_op.register_fake
    def _sparge_attention_fake(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        is_causal: bool,
        scale: float,
        topk: float,
        q_tile_override: int,
        sparse_q_block_tokens: int,
        sparse_k_block_tokens: int,
    ) -> torch.Tensor:
        del key, is_causal, scale, topk, q_tile_override, sparse_q_block_tokens, sparse_k_block_tokens
        return query.new_empty(query.shape, dtype=value.dtype)


_sparge_attention_op = torch.ops.vllm_omni.sparge_attention


def _sparge_metadata_error(attn_metadata: AttentionMetadata | None) -> str | None:
    if attn_metadata is None:
        return None
    if attn_metadata.attn_mask is not None:
        return "attn_mask is not supported"
    if attn_metadata.joint_attn_mask is not None:
        return "joint_attn_mask is not supported"
    if attn_metadata.cross_query_mask is not None:
        return "cross-query masks are not supported"
    if attn_metadata.packed_padding is not None:
        return "packed padding metadata is not supported"
    if attn_metadata.full_attn_spans is not None or attn_metadata.query_ranges is not None:
        return "piecewise attention metadata is not supported"
    extra = attn_metadata.extra
    packed_keys = ("cu_seqlens_q", "cu_seqlens_k", "max_seqlen_q", "max_seqlen_k")
    if any(name in extra for name in packed_keys):
        return "packed variable-length attention is not supported"
    if extra.get("kv_cache_dtype") not in (None, "auto", "float"):
        return "quantized KV cache is not supported"
    return None


def _validate_sparge_metadata(attn_metadata: AttentionMetadata | None) -> None:
    reason = _sparge_metadata_error(attn_metadata)
    if reason is not None:
        raise ValueError(f"SPARGE_ATTN: {reason}. Select a compatible backend.")


def _validate_sparse_tiling(head_size: int, q_block_tokens: int, q_tile_override: int) -> None:
    if head_size == 128:
        if q_block_tokens == 64 and q_tile_override not in (0, 64):
            raise ValueError("SPARGE_ATTN with head_dim=128 and q_block_tokens=64 requires q_tile_override 0 or 64.")
        if q_block_tokens == 256 and q_tile_override != 256:
            raise ValueError("SPARGE_ATTN with head_dim=128 and q_block_tokens=256 requires q_tile_override=256.")
        if q_block_tokens not in (64, 256):
            raise ValueError("SPARGE_ATTN with head_dim=128 requires q_block_tokens 64 or 256.")
        return
    if q_block_tokens != 64:
        raise ValueError("SPARGE_ATTN with head_dim=64 requires q_block_tokens=64.")
    if q_tile_override not in (0, 128):
        raise ValueError("SPARGE_ATTN with head_dim=64 requires q_tile_override 0 or 128.")


class SpargeAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True
    supported_platforms = ("xpu",)

    @classmethod
    def validate_available(cls) -> None:
        try:
            import auto_round_kernel as ark
        except ImportError as exc:
            raise ImportError(_MISSING_AUTO_ROUND) from exc
        if not callable(getattr(ark, "sparge_sage2_attn_meansim_topk_xpu", None)):
            raise ImportError(_MISSING_AUTO_ROUND)

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return list(_SUPPORTED_HEAD_SIZES)

    @staticmethod
    def get_name() -> str:
        return "SPARGE_ATTN"

    @staticmethod
    def get_impl_cls() -> type["SpargeAttentionImpl"]:
        return SpargeAttentionImpl

    @classmethod
    def supports_cross_head_geometry(cls, q_heads: int, kv_heads: int) -> bool:
        return kv_heads > 0 and q_heads % kv_heads == 0

    @classmethod
    def resolve_capabilities(cls, context: ExecutionContext) -> ExecutionPathResult:
        return ExecutionPathResult.unmigrated(cls.get_name(), context, path="runtime_validation_dependent")


class SpargeAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        qkv_layout: str | None = None,
        backend_kwargs: dict[str, Any] | None = None,
        **extra_impl_args,
    ) -> None:
        del prefix, qkv_layout
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_size = head_size
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout = float(extra_impl_args.get("dropout_p", 0.0))

        if head_size not in _SUPPORTED_HEAD_SIZES:
            raise ValueError(f"SPARGE_ATTN supports head sizes {list(_SUPPORTED_HEAD_SIZES)}, got {head_size}.")
        if self.num_kv_heads <= 0 or self.num_heads % self.num_kv_heads:
            raise ValueError(
                "SPARGE_ATTN requires query heads to be a multiple of KV heads, "
                f"got q_heads={self.num_heads} and kv_heads={self.num_kv_heads}."
            )

        kwargs = backend_kwargs or {}
        default_q_block = 256 if head_size == 128 else 64
        self.topk = float(kwargs.get("sparge_topk", 0.5))
        self.q_block_tokens = int(kwargs.get("sparge_q_block_tokens", default_q_block))
        self.k_block_tokens = int(kwargs.get("sparge_k_block_tokens", 64))
        q_tile_override = kwargs.get("sparge_q_tile_override")
        if q_tile_override is None:
            if head_size == 64:
                q_tile_override = 128
            else:
                q_tile_override = 0 if self.q_block_tokens == 64 else 256
        self.q_tile_override = int(q_tile_override)

        if not 0.0 < self.topk <= 1.0:
            raise ValueError(f"SPARGE_ATTN top-k ratio must be in (0, 1], got {self.topk}.")
        if self.k_block_tokens != 64:
            raise ValueError(f"SPARGE_ATTN requires k_block_tokens=64, got {self.k_block_tokens}.")
        _validate_sparse_tiling(head_size, self.q_block_tokens, self.q_tile_override)

        known_kwargs = {
            "sparge_topk",
            "sparge_q_block_tokens",
            "sparge_k_block_tokens",
            "sparge_q_tile_override",
        }
        unknown_kwargs = sorted(set(kwargs) - known_kwargs)
        if unknown_kwargs:
            logger.warning("SpargeAttentionImpl ignoring unknown backend kwargs: %s", unknown_kwargs)

    def resolve_execution_path(
        self,
        context: ExecutionContext,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
    ) -> ExecutionPathResult:
        result = ExecutionPathResult.unmigrated(
            "SPARGE_ATTN", replace(context, kernel_variant=None), path="sparge_int8"
        )
        metadata_error = _sparge_metadata_error(attn_metadata)
        if metadata_error is not None:
            return replace(result, support=CapabilityResult.unsupported(f"SPARGE_ATTN: {metadata_error}"))
        if self.dropout != 0.0:
            return replace(
                result,
                support=CapabilityResult.unsupported(
                    f"SPARGE_ATTN: does not support dropout (dropout_p={self.dropout})."
                ),
            )
        if context.platform != "xpu":
            return replace(result, support=CapabilityResult.unsupported("SPARGE_ATTN runs on XPU only."))
        if any(tensor.ndim != 4 for tensor in (query, key, value)):
            reason = "Q, K, and V must have rank 4 (batch, sequence, heads, head dimension)."
        elif any(tensor.dtype not in _SUPPORTED_DTYPES for tensor in (query, key, value)):
            reason = "Q, K, and V must be float16 or bfloat16."
        elif query.dtype != key.dtype or query.dtype != value.dtype:
            reason = "Q, K, and V dtypes must match."
        elif query.device != key.device or query.device != value.device:
            reason = "Q, K, and V devices must match."
        elif key.shape != value.shape or query.shape[0] != key.shape[0] or query.shape[-1] != key.shape[-1]:
            reason = "K/V shapes, Q/K batch sizes, and Q/K head dimensions must match."
        elif query.shape[-1] not in _SUPPORTED_HEAD_SIZES:
            reason = f"head dimension must be one of {list(_SUPPORTED_HEAD_SIZES)}."
        elif key.shape[2] <= 0 or query.shape[2] % key.shape[2]:
            reason = "query heads must be a multiple of KV heads."
        elif any(size == 0 for tensor in (query, key, value) for size in tensor.shape):
            reason = "Q, K, and V dimensions must be nonzero."
        elif self.causal and query.shape[1] != key.shape[1]:
            reason = "causal attention requires equal query and K/V sequence lengths."
        else:
            reason = None
        if reason is not None:
            return replace(result, support=CapabilityResult.unsupported(f"SPARGE_ATTN: {reason}"))

        dtype = str(query.dtype).removeprefix("torch.")
        resolved_context = replace(
            context,
            kernel_variant=None,
            dtype=dtype,
            causal=self.causal,
            mask_mode=MaskMode.NONE,
            packing_mode=PackingMode.NONE,
        )
        result = ExecutionPathResult.unmigrated("SPARGE_ATTN", resolved_context, path="sparge_int8")
        if context.parallel_strategy in (ParallelStrategy.RING, ParallelStrategy.HYBRID_ULYSSES_RING):
            return replace(
                result,
                support=CapabilityResult.unsupported(
                    "SPARGE_ATTN does not support ring sequence parallelism: "
                    "the ring path bypasses the backend. Use Ulysses SP or disable ring SP."
                ),
            )
        if (
            context.mask_mode is not MaskMode.NONE
            or context.packing_mode is not PackingMode.NONE
            or context.piecewise
            or context.paged_kv
            or context.kv_cache_dtype not in (None, "auto", "float")
            or context.parallel_strategy is not ParallelStrategy.NONE
            or context.cross_parallel_layout is not None
            or context.outer_boundaries
        ):
            return replace(
                result,
                support=CapabilityResult.unmigrated(
                    "SPARGE_ATTN supports dense, non-packed XPU attention; this execution context is not verified."
                ),
            )
        return replace(result, support=CapabilityResult.supported(), compilation_mode=CompilationMode.CUSTOM_OP)

    def forward_xpu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        _validate_sparge_metadata(attn_metadata)
        if self.dropout != 0.0:
            raise ValueError(f"SPARGE_ATTN: does not support dropout (dropout_p={self.dropout}).")
        return _sparge_attention_op(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            self.causal,
            self.softmax_scale,
            self.topk,
            self.q_tile_override,
            self.q_block_tokens,
            self.k_block_tokens,
        )
