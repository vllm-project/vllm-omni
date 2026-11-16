# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import inspect
import math
from dataclasses import replace
from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)
from vllm_omni.diffusion.attention.backends.utils.prefix_kv import slice_valid_kv_prefix
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
        per_block_mean: bool = True,
        q_chunk_size: int = 0,
    ) -> torch.Tensor:
        from sageattn3 import sageattn3_blackwell as _kernel

        kernel_kwargs: dict[str, Any] = {"is_causal": is_causal}
        if not per_block_mean or q_chunk_size > 0:
            accepts_per_block_mean, has_chunking = _inspect_sage3_kernel_caps(_kernel)
            if not per_block_mean:
                if not accepts_per_block_mean:
                    raise ValueError(
                        "SAGE_ATTN_3: sage3.per_block_mean=False requires a sageattn3 kernel "
                        "that accepts per_block_mean."
                    )
                kernel_kwargs["per_block_mean"] = False
            if q_chunk_size > 0:
                if not has_chunking:
                    raise ValueError(
                        "SAGE_ATTN_3: sage3.q_chunk_size requires a sageattn3 build with Q-chunking support "
                        "(e.g. thu-ml/SageAttention#394)."
                    )
                kernel_kwargs["q_chunk_size"] = q_chunk_size

        # Sage3 centers K in place. Own that storage to honor mutates_args=(),
        # including when the caller aliases Q, K, and V.
        return _kernel(query, key.clone(), value, **kernel_kwargs).contiguous()

    @_sageattn3_blackwell_op.register_fake
    def _(query, key, value, is_causal, per_block_mean=True, q_chunk_size=0):
        return torch.empty(query.shape, dtype=query.dtype, device=query.device)


_sageattn3_blackwell_op = torch.ops.vllm_omni.sageattn3_blackwell

_SAGE3_GROUP_SIZE = 128
_INT32_MAX = 2**31 - 1


def _parse_sage3_backend_kwargs(backend_kwargs: dict[str, Any] | None) -> tuple[bool, int | None]:
    if not backend_kwargs:
        return True, None
    kwargs = dict(backend_kwargs)
    per_block_mean = kwargs.pop("per_block_mean", True)
    if not isinstance(per_block_mean, bool):
        raise TypeError(f"SAGE_ATTN_3: per_block_mean must be a bool, got {type(per_block_mean)!r}.")
    q_chunk_size = kwargs.pop("q_chunk_size", None)
    if q_chunk_size is not None:
        if isinstance(q_chunk_size, bool) or not isinstance(q_chunk_size, int):
            raise TypeError(f"SAGE_ATTN_3: q_chunk_size must be an int or None, got {type(q_chunk_size)!r}.")
        if q_chunk_size <= 0 or q_chunk_size % _SAGE3_GROUP_SIZE != 0:
            raise ValueError(
                f"SAGE_ATTN_3: q_chunk_size must be a positive multiple of {_SAGE3_GROUP_SIZE}, got {q_chunk_size}."
            )
    if kwargs:
        logger.warning("SageAttention3Impl ignoring backend_kwargs: %s", sorted(kwargs.keys()))
    return per_block_mean, q_chunk_size


def _inspect_sage3_kernel_caps(kernel: Any | None = None) -> tuple[bool, bool]:
    if kernel is None:
        from sageattn3 import sageattn3_blackwell as _imported_kernel

        target: Any = _imported_kernel
    else:
        target = kernel

    try:
        params = inspect.signature(target).parameters
    except (TypeError, ValueError):
        return False, False
    kw_kinds = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    pbm_param = params.get("per_block_mean")
    accepts_per_block_mean = has_var_kw or (pbm_param is not None and pbm_param.kind in kw_kinds)
    q_chunk_param = params.get("q_chunk_size")
    has_chunking = q_chunk_param is not None and q_chunk_param.kind in kw_kinds
    return accepts_per_block_mean, has_chunking


def _delta_s_batch_stride(heads: int, q_tokens: int, kv_tokens: int) -> int:
    q_groups = (q_tokens + _SAGE3_GROUP_SIZE - 1) // _SAGE3_GROUP_SIZE
    padded_kv_tokens = ((kv_tokens + _SAGE3_GROUP_SIZE - 1) // _SAGE3_GROUP_SIZE) * _SAGE3_GROUP_SIZE
    return heads * q_groups * padded_kv_tokens


def _validate_sage3_delta_s_stride(
    heads: int,
    q_tokens: int,
    kv_tokens: int,
    head_dim: int,
    *,
    is_causal: bool,
    per_block_mean: bool,
    q_chunk_size: int | None,
    has_chunking: bool,
) -> None:
    if not per_block_mean or head_dim >= 256:
        return
    # Upstream PR #394 only chunks non-causal per_block_mean attention; causal calls
    # allocate the full delta_s TMA layout and overflow INT32 at large sequence lengths.
    if not is_causal and q_chunk_size is None and has_chunking:
        return
    effective_q = min(q_tokens, q_chunk_size) if (not is_causal and q_chunk_size is not None) else q_tokens
    stride = _delta_s_batch_stride(heads, effective_q, kv_tokens)
    if stride > _INT32_MAX:
        raise ValueError(
            "SAGE_ATTN_3: per_block_mean delta_s TMA batch stride "
            f"({stride}) exceeds INT32_MAX ({_INT32_MAX}) for heads={heads}, q_len={q_tokens}, kv_len={kv_tokens}. "
            "Install sageattn3 with Q-chunking support (thu-ml/SageAttention#394), set a smaller "
            "sage3.q_chunk_size, disable sage3.per_block_mean, or select SAGE_ATTN / CUDNN_ATTN / TORCH_SDPA."
        )


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
    supports_prefix_kv_slicing: bool = True

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
        backend_kwargs: dict[str, Any] | None = None,
        **extra_impl_args,
    ) -> None:
        self.causal = causal
        self.head_size = head_size
        self.softmax_scale = softmax_scale
        self.dropout = extra_impl_args.get("dropout_p", 0.0)
        expected_scale = head_size**-0.5
        if not math.isclose(softmax_scale, expected_scale, rel_tol=1e-6):
            raise ValueError(
                "SAGE_ATTN_3 does not expose a custom softmax scale; "
                f"expected {expected_scale} for head_size={head_size}, got {softmax_scale}."
            )
        self.per_block_mean, self.q_chunk_size = _parse_sage3_backend_kwargs(backend_kwargs)
        self._kernel_accepts_per_block_mean, self._kernel_has_chunking = _inspect_sage3_kernel_caps()
        if not self.per_block_mean and not self._kernel_accepts_per_block_mean:
            raise ValueError(
                "SAGE_ATTN_3: sage3.per_block_mean=False requires a sageattn3 kernel that accepts per_block_mean."
            )
        if self.q_chunk_size is not None and not self._kernel_has_chunking:
            raise ValueError(
                "SAGE_ATTN_3: sage3.q_chunk_size requires a sageattn3 build with Q-chunking support "
                "(e.g. thu-ml/SageAttention#394)."
            )

    def resolve_execution_path(
        self,
        context: ExecutionContext,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
    ) -> ExecutionPathResult:
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
        if attn_metadata is not None and attn_metadata.attn_mask is not None:
            return replace(result, support=CapabilityResult.unsupported("SAGE_ATTN_3: attn_mask is not supported"))
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
        elif query.shape[2] != key.shape[2]:
            reason = (
                "does not support GQA/MQA "
                f"(q_heads={query.shape[2]}, kv_heads={key.shape[2]}). Select a GQA-capable backend."
            )
        elif not math.isclose(self.softmax_scale, query.shape[-1] ** -0.5, rel_tol=1e-6):
            reason = f"softmax_scale {self.softmax_scale} does not match head dim {query.shape[-1]}."
        elif self.dropout != 0.0:
            reason = f"does not support dropout (dropout_p={self.dropout})."
        else:
            reason = None
        if reason:
            return replace(result, support=CapabilityResult.unsupported("SAGE_ATTN_3: " + reason))
        try:
            key, value = slice_valid_kv_prefix(key, value, attn_metadata)
            _validate_sage3_delta_s_stride(
                query.shape[2],
                query.shape[1],
                key.shape[1],
                query.shape[-1],
                is_causal=self.causal,
                per_block_mean=self.per_block_mean,
                q_chunk_size=self.q_chunk_size,
                has_chunking=self._kernel_has_chunking,
            )
        except ValueError as exc:
            message = str(exc)
            if not message.startswith("SAGE_ATTN_3:"):
                message = f"SAGE_ATTN_3: {message}"
            return replace(result, support=CapabilityResult.unsupported(message))
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
            or not self.per_block_mean
            or self.q_chunk_size is not None
        ):
            return result
        # D=256 can dispatch to SDPA; only the tested FP4 paths are verified.
        if (
            query.shape[-1] not in (64, 128)
            or any(t.stride(-1) != 1 for t in (query, key, value))
            or (self.causal and query.shape[1] != key.shape[1])
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
        if self.dropout != 0.0:
            raise ValueError(f"SAGE_ATTN_3: does not support dropout (dropout_p={self.dropout}).")
        # The constructor validates scale against head_size. An integer shape
        # guard preserves that contract under dynamic fullgraph compilation.
        if query.shape[-1] != self.head_size:
            raise ValueError(
                f"SAGE_ATTN_3: softmax_scale {self.softmax_scale} does not match head dim {query.shape[-1]}."
            )
        key, value = slice_valid_kv_prefix(key, value, attn_metadata)
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

        _validate_sage3_delta_s_stride(
            query.shape[1],
            query.shape[2],
            key.shape[2],
            query.shape[-1],
            is_causal=self.causal,
            per_block_mean=self.per_block_mean,
            q_chunk_size=self.q_chunk_size,
            has_chunking=self._kernel_has_chunking,
        )

        output = _sageattn3_blackwell_op(
            query,
            key,
            value,
            self.causal,
            self.per_block_mean,
            self.q_chunk_size if self.q_chunk_size is not None else 0,
        )

        return output.transpose(1, 2).contiguous()
