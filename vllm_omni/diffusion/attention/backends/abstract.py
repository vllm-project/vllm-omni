# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import torch

from vllm_omni.platforms import current_omni_platform


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


T = TypeVar("T", bound=AttentionMetadata)


class AttentionImpl(ABC, Generic[T]):
    @abstractmethod
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
        raise NotImplementedError

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
