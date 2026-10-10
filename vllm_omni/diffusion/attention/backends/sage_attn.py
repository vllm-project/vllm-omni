# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import os
from collections.abc import Callable
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

_SAGE_ATTN_TENSOR_LAYOUT = os.environ.get("SAGE_ATTN_TENSOR_LAYOUT", "NHD").upper()
assert _SAGE_ATTN_TENSOR_LAYOUT in ("NHD", "HND"), (
    f"SAGE_ATTN_TENSOR_LAYOUT must be 'NHD' or 'HND', got '{_SAGE_ATTN_TENSOR_LAYOUT}'"
)

sageattn: Callable[..., torch.Tensor] | None = None

if current_omni_platform.is_xpu():
    try:
        import inspect

        from auto_round_kernel import ARK

        _ark = ARK()
        xpu_sageattn = _ark.sagev1
        _sagev1_params = inspect.signature(xpu_sageattn).parameters
        _sagev1_has_tensor_layout = "tensor_layout" in _sagev1_params
        _sagev1_scale_param = "sm_scale" if "sm_scale" in _sagev1_params else "scale"
    except ImportError:
        logger.warning(
            "XPU SageAttention (auto_round_kernel.ARK.sagev1) is not available. "
            "Install auto-round-lib for XPU sage attention support."
        )
        xpu_sageattn = None
        _sagev1_has_tensor_layout = False
        _sagev1_scale_param = "scale"
else:
    try:
        from sageattention import sageattn as _sageattn
        from sageattention import sageattn_varlen

        sageattn = _sageattn
    except ImportError:
        logger.warning(
            "SageAttentionBackend is not available. You may install sage-attention"
            " by pip install git+https://github.com/thu-ml/SageAttention.git"
        )
        sageattn = None
        sageattn_varlen = None

if not hasattr(torch.ops.vllm_omni, "sage_attention"):

    @torch.library.custom_op("vllm_omni::sage_attention", mutates_args=())
    def _sage_attention_op(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        is_causal: bool,
        sm_scale: float,
    ) -> torch.Tensor:
        # Keep architecture detection, quantization, and raw CUDA extension
        # calls outside Dynamo tracing, including under fullgraph=True.
        from sageattention import sageattn as kernel

        output = kernel(query, key, value, tensor_layout="NHD", is_causal=is_causal, sm_scale=sm_scale)
        # Sage can return a slice of a padded head dimension. Normalize its
        # strides to match the fake implementation on every kernel path.
        return output.contiguous()

    @_sage_attention_op.register_fake
    def _(query, key, value, is_causal, sm_scale):
        return torch.empty(query.shape, dtype=query.dtype, device=query.device)


_sage_attention_op = torch.ops.vllm_omni.sage_attention


def _validate_sage_metadata(attn_metadata: AttentionMetadata | None) -> None:
    if attn_metadata is not None and attn_metadata.attn_mask is not None:
        raise ValueError("SAGE_ATTN does not support attn_mask. Select a mask-capable backend.")


def _sage_cuda_kernel_variant(query: torch.Tensor) -> str | None:
    # Resolution is eager; never move this architecture query inside forward.
    if query.device.type != "cuda" or not current_omni_platform.is_cuda():
        return None
    capability = current_omni_platform.get_device_capability(query.device.index)
    if capability is None:
        return None
    return f"sage_sm{capability[0]}{capability[1]}"


class SageAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @classmethod
    def resolve_capabilities(cls, context: ExecutionContext) -> ExecutionPathResult:
        # The CUDA dispatcher chooses its kernel from the input device.
        return ExecutionPathResult.unmigrated(
            cls.get_name(), replace(context, kernel_variant=None), path="uninitialized"
        )

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @classmethod
    def supports_packed_mask_free(cls) -> bool:
        # forward_cuda dispatches sageattn_varlen
        # over the caller's packed cu_seqlens, so a packed sequence with padding
        # does not need a boolean attn_mask (the mask is what SAGE cannot take).
        return True

    @staticmethod
    def get_name() -> str:
        return "SAGE_ATTN"

    @staticmethod
    def get_impl_cls() -> type["SageAttentionImpl"]:
        return SageAttentionImpl


class SageAttentionImpl(AttentionImpl):
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
        self.dropout = extra_impl_args.get("dropout_p", 0.0)
        if backend_kwargs:
            logger.warning("SageAttentionImpl ignoring backend_kwargs: %s", list(backend_kwargs.keys()))

    def resolve_execution_path(
        self,
        context: ExecutionContext,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
    ) -> ExecutionPathResult:
        result = ExecutionPathResult.unmigrated("SAGE_ATTN", replace(context, kernel_variant=None), path="unverified")
        if attn_metadata is not None and attn_metadata.attn_mask is not None:
            return replace(result, support=CapabilityResult.unsupported("SAGE_ATTN: attn_mask is not supported"))
        if self.dropout != 0.0:
            return replace(
                result,
                support=CapabilityResult.unsupported(
                    f"SAGE_ATTN: does not support dropout (dropout_p={self.dropout})."
                ),
            )
        # XPU's ARK route is separate and has not been migrated.
        if context.platform != "cuda":
            return result
        extra = attn_metadata.extra if attn_metadata is not None else {}
        packed = any(name in extra for name in ("cu_seqlens_q", "cu_seqlens_k", "max_seqlen_q", "max_seqlen_k"))
        if attn_metadata is not None and attn_metadata.packed_padding is not None:
            packing_mode = PackingMode.PACKED_PADDING
        else:
            packing_mode = PackingMode.MULTI_DOCUMENT if packed else PackingMode.NONE
        context = replace(
            context,
            kernel_variant=_sage_cuda_kernel_variant(query),
            dtype=str(query.dtype).removeprefix("torch."),
            causal=self.causal,
            mask_mode=MaskMode.NONE,
            packing_mode=packing_mode,
            piecewise=attn_metadata is not None and attn_metadata.full_attn_spans is not None,
            kv_cache_dtype=extra.get("kv_cache_dtype"),
        )
        result = ExecutionPathResult.unmigrated("SAGE_ATTN", context, path="sage_dense")
        if sageattn is None:
            return replace(
                result,
                support=CapabilityResult.unsupported(
                    "SAGE_ATTN requires sageattention; install it or select another backend."
                ),
            )
        if any(t.ndim != 4 for t in (query, key, value)):
            reason = "Q, K, and V must have rank 4 (batch, sequence, heads, head dimension)."
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
            return replace(
                result,
                support=CapabilityResult.unsupported(
                    "SAGE_ATTN: " + reason + " Select compatible inputs or another backend."
                ),
            )
        if (
            context.kernel_variant != "sage_sm90"
            or context.dtype not in ("float16", "bfloat16")
            or context.packing_mode is not PackingMode.NONE
            or context.piecewise
            or context.paged_kv
            or context.kv_cache_dtype not in (None, "auto", "float")
            or context.parallel_strategy is not ParallelStrategy.NONE
            or context.outer_boundaries
        ):
            return replace(
                result,
                support=CapabilityResult.unmigrated(
                    "Only dense FP16/BF16 SageAttention on SM90 without packed, parallel, "
                    "or outer boundaries is migrated."
                ),
            )
        # These are the validated paths, not an exhaustive kernel support table.
        if (
            query.shape[-1] not in (32, 64, 96, 128)
            or any(t.stride(-1) != 1 for t in (query, key, value))
            or query.shape[2] != key.shape[2]
            or (self.causal and query.shape[1] != key.shape[1])
        ):
            return replace(
                result,
                support=CapabilityResult.unmigrated(
                    "Validated Sage paths use head sizes 32/64/96/128 with contiguous head elements, "
                    "equal head counts, and equal Q/K lengths if causal."
                ),
            )
        return replace(result, support=CapabilityResult.supported(), compilation_mode=CompilationMode.CUSTOM_OP)

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        _validate_sage_metadata(attn_metadata)
        if self.dropout != 0.0:
            raise ValueError(f"SAGE_ATTN: does not support dropout (dropout_p={self.dropout}).")
        packed = getattr(attn_metadata, "packed_padding", None) if attn_metadata is not None else None
        if packed is not None:
            if sageattn_varlen is None:
                raise ImportError(
                    "SAGE_ATTN requires sageattention. Install with: "
                    "pip install git+https://github.com/thu-ml/SageAttention.git"
                )
            # The packed buffer carries uninitialised padding beyond q_length
            # (the cuDNN path slices K/V itself; flash only reads the declared
            # ranges). Sage's Triton kernel validates the whole tensor, so slice
            # to the declared lengths, run, then scatter back into a zero-filled
            # buffer of the original shape.
            q3 = query.flatten(0, 1)
            k3 = key.flatten(0, 1)
            v3 = value.flatten(0, 1)
            n = int(packed.q_length)
            out = sageattn_varlen(
                q3[:n],
                k3[:n],
                v3[:n],
                packed.cu_seqlens_q,
                packed.cu_seqlens_k,
                n,
                int(packed.kv_length),
                is_causal=self.causal,
                sm_scale=self.softmax_scale,
            )
            if out.shape[0] == q3.shape[0]:
                return out.reshape_as(query)
            full = q3.new_zeros((q3.shape[0],) + tuple(out.shape[1:]))
            full[: out.shape[0]] = out
            return full.reshape_as(query)
        if sageattn is None:
            raise ImportError(
                "SAGE_ATTN requires sageattention. Install with: "
                "pip install git+https://github.com/thu-ml/SageAttention.git"
            )
        output = _sage_attention_op(
            query,
            key,
            value,
            self.causal,
            self.softmax_scale,
        )
        return output

    def forward_xpu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        _validate_sage_metadata(attn_metadata)
        if self.dropout != 0.0:
            raise ValueError(f"SAGE_ATTN: does not support dropout (dropout_p={self.dropout}).")
        if xpu_sageattn is None:
            raise ImportError("XPU SageAttention requires auto-round-lib. Install with: pip install auto-round-lib")
        orig_dtype = query.dtype
        q = query.to(torch.float16) if orig_dtype != torch.float16 else query
        k = key.to(torch.float16) if orig_dtype != torch.float16 else key
        v = value.to(torch.float16) if orig_dtype != torch.float16 else value

        if _sagev1_has_tensor_layout:
            if _SAGE_ATTN_TENSOR_LAYOUT == "HND":
                q = q.transpose(1, 2).contiguous()
                k = k.transpose(1, 2).contiguous()
                v = v.transpose(1, 2).contiguous()
            output = xpu_sageattn(
                q,
                k,
                v,
                tensor_layout=_SAGE_ATTN_TENSOR_LAYOUT,
                is_causal=self.causal,
                **{_sagev1_scale_param: self.softmax_scale},
            )
            if _SAGE_ATTN_TENSOR_LAYOUT == "HND":
                output = output.transpose(1, 2).contiguous()
        else:
            # No tensor_layout support: kernel expects HND [B, H, S, D]
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()
            output = xpu_sageattn(
                q,
                k,
                v,
                is_causal=self.causal,
                **{_sagev1_scale_param: self.softmax_scale},
            )
            output = output.transpose(1, 2).contiguous()

        if orig_dtype != torch.float16:
            output = output.to(orig_dtype)
        return output
