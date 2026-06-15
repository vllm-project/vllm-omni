# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

import torch

from vllm_omni.platforms import current_omni_platform


@dataclass(frozen=True)
class AttentionSegment:
    """Request-local semantic span used to split/gather dynamic attention.

    This describes model semantics, not cache layout. Cache-specific block
    tables and slot mappings stay on ``AttentionMetadata``.
    """

    request_id: str
    request_index: int
    role: str
    packed_start: int
    length: int
    request_start: int = 0
    branch: str = "positive"
    position_id: str | None = None

    @property
    def packed_end(self) -> int:
        return self.packed_start + self.length

    @property
    def request_end(self) -> int:
        return self.request_start + self.length


class AttentionBackend(ABC):
    """Abstract class for diffusion attention backends."""

    accept_output_buffer: bool = False

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


@dataclass
class AttentionMetadata:
    # Varlen execution contract. These fields are direct because attention
    # backends consume them on the hot path.
    q_cu_seqlens: torch.Tensor | None = None
    kv_cu_seqlens: torch.Tensor | None = None
    max_q_len: int | None = None
    max_kv_len: int | None = None
    q_segments: list[AttentionSegment] | None = None
    kv_segments: list[AttentionSegment] | None = None
    position: Any | None = None
    padded_tokens: int = 0

    # Future cache layout hooks, kept separate from AttentionSegment.
    # block_table: torch.Tensor | None = None
    # slot_mapping: torch.Tensor | None = None

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

    # Piecewise attention metadata (mixed causal/full masks).
    # full_attn_spans: per-sample [start, end) spans in global coordinates using full attention.
    full_attn_spans: list[list[tuple[int, int]]] | None = None

    @property
    def is_varlen(self) -> bool:
        return self.q_cu_seqlens is not None or self.kv_cu_seqlens is not None


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

    def get_kv_cache_spec(self, *args: Any, **kwargs: Any) -> None:
        # TODO(kv-cache): attention implementations do not register cache specs
        # for diffusion dynamic attention yet.
        return None

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
