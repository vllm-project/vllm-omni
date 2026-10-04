# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import math
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
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)

try:
    from sageattn3 import sageattn3_blackwell  # noqa: F401
except ImportError:
    logger.warning(
        "SageAttention3Backend is not available. Install `sageattn3` from "
        "https://github.com/thu-ml/SageAttention/tree/main/sageattention3_blackwell"
    )
    raise ImportError


# Wrapping sageattn3_blackwell as a torch.library custom op keeps it opaque to
# torch.compile. Otherwise Dynamo graph-breaks on the raw pybind11 kernel and
# Inductor fails scheduling with KeyError: 'op5'. The hasattr guard keeps this
# idempotent across test re-imports that pop the module from sys.modules.
if not hasattr(torch.ops.vllm_omni, "sageattn3_blackwell"):

    @torch.library.custom_op("vllm_omni::sageattn3_blackwell", mutates_args=())
    def _sageattn3_blackwell_op(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        is_causal: bool,
    ) -> torch.Tensor:
        from sageattn3 import sageattn3_blackwell as _kernel

        # Sage3 centers K in place. Own that storage to honor mutates_args=(),
        # including when the caller aliases Q, K, and V.
        return _kernel(query, key.clone(), value, is_causal=is_causal).contiguous()

    @_sageattn3_blackwell_op.register_fake
    def _(query, key, value, is_causal):
        return torch.empty(query.shape, dtype=query.dtype, device=query.device)


_sageattn3_blackwell_op = torch.ops.vllm_omni.sageattn3_blackwell


def _sage3_kernel_variant(query: torch.Tensor) -> str | None:
    if query.device.type != "cuda" or not current_omni_platform.is_cuda():
        return None
    capability = current_omni_platform.get_device_capability(query.device.index)
    if capability is None:
        return None
    return f"sage3_sm{capability[0]}{capability[1]}"


def _validate_sage3_metadata(attn_metadata: AttentionMetadata | None) -> None:
    if attn_metadata is not None and attn_metadata.attn_mask is not None:
        raise ValueError("SAGE_ATTN_3 does not support attn_mask. Select a mask-capable backend.")


class SageAttention3Backend(AttentionBackend):
    accept_output_buffer: bool = True

    @classmethod
    def resolve_capabilities(cls, context: ExecutionContext) -> ExecutionPathResult:
        return ExecutionPathResult.unmigrated(cls.get_name(), replace(context, kernel_variant=None), path="unverified")

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 128, 256]

    @staticmethod
    def get_name() -> str:
        return "SAGE_ATTN_3"

    @staticmethod
    def get_impl_cls() -> type["SageAttention3Impl"]:
        return SageAttention3Impl


class SageAttention3Impl(AttentionImpl):
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
        self.dropout = extra_impl_args.get("dropout_p", 0.0)
        expected_scale = head_size**-0.5
        if not math.isclose(softmax_scale, expected_scale, rel_tol=1e-6):
            raise ValueError(
                "SAGE_ATTN_3 does not expose a custom softmax scale; "
                f"expected {expected_scale} for head_size={head_size}, got {softmax_scale}."
            )

    def resolve_execution_path(
        self,
        context: ExecutionContext,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
    ) -> ExecutionPathResult:
        _validate_sage3_metadata(attn_metadata)
        extra = attn_metadata.extra if attn_metadata is not None else {}
        packed = any(name in extra for name in ("cu_seqlens_q", "cu_seqlens_k", "max_seqlen_q", "max_seqlen_k"))
        if attn_metadata is not None and attn_metadata.packed_padding is not None:
            packing_mode = PackingMode.PACKED_PADDING
        else:
            packing_mode = PackingMode.MULTI_DOCUMENT if packed else PackingMode.NONE
        context = replace(
            context,
            kernel_variant=_sage3_kernel_variant(query),
            dtype=str(query.dtype).removeprefix("torch."),
            causal=self.causal,
            mask_mode=MaskMode.NONE,
            packing_mode=packing_mode,
            piecewise=attn_metadata is not None and attn_metadata.full_attn_spans is not None,
            kv_cache_dtype=extra.get("kv_cache_dtype"),
        )
        result = ExecutionPathResult.unmigrated("SAGE_ATTN_3", context, path="sage3_dense")
        if (
            context.platform != "cuda"
            or context.kernel_variant != "sage3_sm120"
            or context.dtype not in ("float16", "bfloat16")
            or context.packing_mode is not PackingMode.NONE
            or context.piecewise
            or context.paged_kv
            or context.kv_cache_dtype not in (None, "auto", "float")
            or context.parallel_strategy is not ParallelStrategy.NONE
            or context.outer_boundaries
            or self.dropout != 0.0
        ):
            return result
        if any(t.ndim != 4 for t in (query, key, value)):
            reason = "Q, K, and V must have rank 4."
        elif query.dtype != key.dtype or query.dtype != value.dtype:
            reason = "Q, K, and V dtypes must match."
        elif query.device != key.device or query.device != value.device:
            reason = "Q, K, and V devices must match."
        elif key.shape != value.shape or query.shape[0] != key.shape[0] or query.shape[-1] != key.shape[-1]:
            reason = "K/V shapes, Q/K batch sizes, and Q/K head dimensions must match."
        elif any(size == 0 for t in (query, key, value) for size in t.shape):
            reason = "Q, K, and V dimensions must be nonzero."
        else:
            reason = None
        if reason:
            return replace(result, support=CapabilityResult.unsupported("SAGE_ATTN_3: " + reason))
        # D=256 can dispatch to SDPA; only the tested FP4 paths are verified.
        if (
            query.shape[-1] not in (64, 128)
            or query.shape[2] != key.shape[2]
            or any(t.stride(-1) != 1 for t in (query, key, value))
            or (self.causal and query.shape[1] != key.shape[1])
            or not math.isclose(self.softmax_scale, query.shape[-1] ** -0.5, rel_tol=1e-6)
        ):
            return result
        return replace(result, support=CapabilityResult.supported(), compilation_mode=CompilationMode.CUSTOM_OP)

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        _validate_sage3_metadata(attn_metadata)
        query = query.transpose(1, 2).contiguous()
        key = key.transpose(1, 2).contiguous()
        value = value.transpose(1, 2).contiguous()

        if key.shape[1] != query.shape[1]:
            if query.shape[1] % key.shape[1] != 0:
                raise ValueError(
                    "GQA/MQA requires query heads to be a multiple of KV heads, "
                    f"got q_heads={query.shape[1]} and kv_heads={key.shape[1]}"
                )
            raise NotImplementedError(
                "SAGE_ATTN_3 was explicitly selected but does not support GQA/MQA "
                f"(q_heads={query.shape[1]}, kv_heads={key.shape[1]}). Select a GQA-capable backend."
            )

        output = _sageattn3_blackwell_op(query, key, value, self.causal)

        return output.transpose(1, 2).contiguous()
