# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

import torch

from vllm_omni.platforms import current_omni_platform


class AttentionBackend(ABC):
    """Abstract class for diffusion attention backends."""

    accept_output_buffer: bool = False
    supports_piecewise_spans: bool = False
    # A backend that supports this capability can consume the opaque paged-KV
    # context prepared by the diffusion Worker data plane.  Keeping the
    # capability on the backend class prevents a paged request from silently
    # falling back to dense attention on an incompatible implementation.
    supports_paged_kv: bool = False
    # The backend can represent a contiguous valid K/V prefix by slicing the
    # tensors instead of materializing a padding mask. Models may use this to
    # avoid a slower masked-attention plan when tail padding is not semantic.
    supports_prefix_kv_slicing: bool = False

    @classmethod
    def supports_packed_mask_free(cls) -> bool:
        """Whether packed attention never reads attn_mask on this platform.

        When True, models that pack a [real, pad] two-document layout and
        carry cu_seqlens/max_seqlen in ``AttentionMetadata.extra`` may skip
        constructing the padding mask entirely. Backends whose mask-free
        behavior is platform-dependent must check current_omni_platform.
        """
        return False

    # ``OmniPlatformEnum`` values this backend runs on; None means unrestricted.
    # Platform resolution rejects an explicit selection outside this set, so a
    # hardware-specific backend fails before the model is built.
    supported_platforms: tuple[str, ...] | None = None

    @classmethod
    def validate_available(cls) -> None:
        """Raise if this backend's optional dependencies are missing.

        Called during platform resolution, i.e. before model construction, so a
        backend that probes its kernel package lazily still reports the problem
        while the user can still act on it.
        """
        return None

    @classmethod
    def supports_attention_mask(cls) -> bool:
        return False

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_impl_cls() -> type["AttentionImpl"]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_builder_cls():  # -> Type["AttentionMetadataBuilder"]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_supported_head_sizes() -> list[int]:
        """Get the list of supported head sizes for this backend."""
        raise NotImplementedError

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        supported_head_sizes = cls.get_supported_head_sizes()
        return (not supported_head_sizes) or head_size in supported_head_sizes

    @classmethod
    def indexes_kv_by_block_stride(cls) -> bool:
        """Whether this backend reads K/V pages by the runtime block stride.

        Returning ``True`` means the physical cache layout has ``num_blocks``
        as its outer stride, so native vLLM may safely use page-size padding
        when it unifies cache layouts across layers. Dense diffusion backends
        conservatively keep the default ``False``; a paged backend should
        override this only when its kernel actually follows that layout.
        """

        return False


@dataclass(frozen=True, slots=True)
class QueryRange:
    local_start: int
    local_end: int
    global_start: int


@dataclass(frozen=True, slots=True)
class VideoTokenLayout:
    """Where the video segment sits inside a packed multimodal sequence.

    A model that packs its sequence as ``[prefix | t*h*w video rows | padding]``
    publishes this so backends can recover spatiotemporal locality; the prefix
    holds everything that is not video (text, visual conditions, audio).
    Publishing it also asserts that any ``attn_mask`` masks only the trailing
    padding, so ``prefix_len + t*h*w`` is the used length of the sequence.

    Plain ints, so reading it never forces a device-to-host sync.
    """

    prefix_len: int
    latent_grid: tuple[int, int, int]


@dataclass
class AttentionMetadata:
    attn_mask: torch.Tensor | None = None
    joint_attn_mask: torch.Tensor | None = None
    # a joint mask for the joint query, key, and value, depends the joint_strategy
    joint_query: torch.Tensor | None = None
    # a replicated tensor among processes appended to the front or rear of query, depends the joint_strategy
    joint_key: torch.Tensor | None = None
    # a replicated tensor among processes appended to the front or rear of key, depends the joint_strategy
    joint_value: torch.Tensor | None = None
    # a replicated tensor among processes appended to the front or rear of value, depends the joint_strategy
    joint_strategy: str = "front"
    # the strategy to joint the query, key, and value, can be "front" or "rear"
    extra: dict[str, Any] = field(default_factory=dict)
    # Opaque backend-specific per-forward parameters (e.g. block masks, KV indices).
    # Backends MUST silently ignore unknown keys.
    #
    # Well-known optional keys (convention, not required on all forwards):
    #   "kv_cache_dtype": str | None — quantized KV dtype (e.g. "fp8"); backends
    #     decide whether/how to apply.
    #   "cu_seqlens_q" / "cu_seqlens_k": int32 CUDA tensors describing packed
    #     variable-length query/key sequences for FlashAttention.
    #   "max_seqlen_q" / "max_seqlen_k": maximum sequence lengths paired with
    #     the packed cu_seqlens tensors.
    #   "valid_kv_length": int — contiguous valid K/V prefix length for a
    #     backend that advertises supports_prefix_kv_slicing.
    #   "npu_attn_varlen": bool — model opt-in for the NPU packed varlen path
    #     (TND npu_fusion_attention driven by cu_seqlens, mask never read).
    #     Requires the [real, pad] two-document packing contract; see
    #     FlashAttentionImpl._forward_varlen_packed_npu.
    #   "laser_input_scale": float — model opt-in input pre-scale for the NPU
    #     ascend_laser_attention path. The kernel stores unscaled QK^T in an
    #     fp16 workspace, so outlier activations overflow 65504 into NaN rows;
    #     with this set (>1), q/k/v are divided by the factor before the op,
    #     the kernel scale_value is multiplied by its square, and the output
    #     is scaled back (exact for power-of-two factors). Absent means no
    #     pre-scaling. See FlashAttentionImpl._forward_prefix_kv_slice_npu.

    # Piecewise attention metadata (mixed causal/full masks).
    # full_attn_spans: per-sample [start, end) spans in global coordinates using full attention.
    full_attn_spans: list[list[tuple[int, int]]] | None = None
    query_ranges: tuple[QueryRange, ...] | None = None

    # Geometry of the video segment for backends that exploit spatiotemporal
    # locality (block-sparse selection, tiled masks). Dense backends ignore it.
    video_layout: VideoTokenLayout | None = None


T = TypeVar("T", bound=AttentionMetadata)


class AttentionImpl(ABC, Generic[T]):
    # Per-platform kv_cache_dtype support. Maps OmniPlatformEnum value
    # (e.g. "cuda", "npu") to the set of quantized dtypes that platform
    # handles.
    #
    # To add FP8 support for a new platform in a subclass:
    #   _supported_kv_cache_dtypes = {"cuda": {"fp8"}, "npu": {"fp8"}}
    _supported_kv_cache_dtypes: dict[str, set[str]] = {}

    @abstractmethod
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
        raise NotImplementedError

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: str | None, platform_key: str) -> bool:
        if kv_cache_dtype is None:
            return True
        return kv_cache_dtype in cls._supported_kv_cache_dtypes.get(platform_key, set())

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: T | None = None,
    ) -> torch.Tensor:
        """Dispatch to platform-specific forward implementation."""
        if current_omni_platform.is_rocm():
            return self.forward_hip(query, key, value, attn_metadata)
        elif current_omni_platform.is_cuda():
            return self.forward_cuda(query, key, value, attn_metadata)
        elif current_omni_platform.is_npu():
            return self.forward_npu(query, key, value, attn_metadata)
        elif current_omni_platform.is_xpu():
            return self.forward_xpu(query, key, value, attn_metadata)
        elif current_omni_platform.is_musa():
            return self.forward_musa(query, key, value, attn_metadata)
        else:
            raise NotImplementedError(f"No forward implementation for platform: {current_omni_platform}")

    def forward_paged(self, paged_kv_context: Any) -> torch.Tensor:
        """Execute one Worker-prepared paged-KV attention call.

        The context is intentionally opaque to the common attention layer.
        Backends opt in by setting ``supports_paged_kv`` on their backend
        class and implementing this method.  Dense callers continue to use
        ``forward`` unchanged.
        """

        del paged_kv_context
        raise NotImplementedError(f"{type(self).__name__} does not support paged KV attention")

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: T | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward_npu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: T | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward_xpu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: T | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward_hip(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: T | None = None,
    ) -> torch.Tensor:
        # By default, HIP ops are compatible with CUDA ops.
        return self.forward_cuda(query, key, value, attn_metadata)

    def forward_musa(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: T | None = None,
    ) -> torch.Tensor:
        # By default, MUSA ops are compatible with CUDA ops.
        return self.forward_cuda(query, key, value, attn_metadata)
