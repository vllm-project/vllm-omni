# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from typing import Any, Generic, TypeVar

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


@dataclass(frozen=True)
class PerForwardState:
    """Framework-injected, per-forward-pass attention state.

    Carried on :attr:`AttentionMetadata.per_forward` so any attention backend can
    read step state without reaching into the global ForwardContext, and so the
    function-plugin path can flatten it into a plain ``params`` dict.

    Fields live on the SAMPLE axis: one entry per sample, shape ``[num_samples]``
    (homogeneous batch -> all equal; heterogeneous batch -> mixed). Raw sequence
    length / packing is NOT here — that is the token axis, carried by q/k/v and
    cu_seqlens. Video *grid* geometry (frame / patch counts) IS provided, as the
    geometry fields below, so a kernel can map the flat sequence to coordinates.

    All fields are optional; backends MUST ignore fields they don't use. Adding
    fields later is non-breaking.
    """

    # Step state, written by the framework (the Attention bridge) from
    # ForwardContext. Per-sample tensors so the same shape covers homogeneous
    # and (future) heterogeneous/continuous batching without a contract change.
    denoise_step_idx: torch.Tensor | None = None
    total_denoise_steps: torch.Tensor | None = None
    # Raw, model-agnostic video geometry primitives, published generically by the
    # framework (a forward pre-hook on the transformer) — NOT by model code. Each
    # is a per-sample tensor of shape ``[num_samples, 3]``:
    #   latent_shape -> post-VAE, pre-patch grid ``(T, H, W)``
    #   patch_size   -> patch dims ``(p_t, p_h, p_w)``
    # They are enough for any kernel to derive its own layout; the default
    # resolver in the SPARSE_ATTN dispatcher turns them into the derived fields
    # below for plugins that just want frame / patch counts.
    latent_shape: torch.Tensor | None = None
    patch_size: torch.Tensor | None = None
    # Derived video geometry (post-patch). No longer written by the bridge;
    # produced on demand by the dispatcher's default resolver (or a plugin's
    # ``geometry_fn``) from the raw primitives above, so structure-aware kernels
    # can split the flat sequence into (frame, patch):
    # seq_len == total_latent_frames * patches_per_frame (the resolver only
    # fills them when that product matches the actual sequence length). Kept
    # optional for back-compat with plugins that read them straight from
    # ``params`` — note the resolver writes Python ints there, while values set
    # here are per-sample tensors; plugins should accept either.
    total_latent_frames: torch.Tensor | None = None
    patches_per_frame: torch.Tensor | None = None

    def to_dict(self, *, exclude_none: bool = False) -> dict[str, Any]:
        """Shallow dict of the fields (no deep-copy of tensors).

        With ``exclude_none=True`` only set fields are returned, so consumers
        can rely on ``params.get(key, default)`` semantics.
        """
        out = {f.name: getattr(self, f.name) for f in fields(self)}
        if exclude_none:
            return {k: v for k, v in out.items() if v is not None}
        return out


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

    # Piecewise attention metadata (mixed causal/full masks).
    # full_attn_spans: per-sample [start, end) spans in global coordinates using full attention.
    full_attn_spans: list[list[tuple[int, int]]] | None = None

    # Typed, framework-injected per-forward state.
    # Distinct from the opaque `extra` dict: `per_forward` holds canonical,
    # documented fields (step state) populated by the Attention wrapper from
    # ForwardContext. Backends ignore it if they don't need it.
    per_forward: PerForwardState | None = None


T = TypeVar("T", bound=AttentionMetadata)


class AttentionImpl(ABC, Generic[T]):
    # Whether this backend reads ``AttentionMetadata.per_forward`` (framework
    # step state / video geometry). Default ``False``: dense backends ignore it,
    # so the ``Attention`` layer skips building the per-forward tensors entirely
    # for them — avoiding per-layer, per-step GPU allocations on the default
    # (non-sparse) inference path. A backend that consumes per_forward (e.g. the
    # SPARSE_ATTN dispatcher) overrides this to ``True``.
    consumes_per_forward: bool = False

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
