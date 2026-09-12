# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SupportStatus(str, Enum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNMIGRATED = "unmigrated"


class CompilationMode(str, Enum):
    TRACEABLE = "traceable"
    CUSTOM_OP = "custom_op"
    EAGER_ONLY = "eager_only"


class PackingMode(str, Enum):
    NONE = "none"
    PACKED_PADDING = "packed_padding"
    MULTI_DOCUMENT = "multi_document"


class MaskMode(str, Enum):
    NONE = "none"
    PADDING = "padding"
    ARBITRARY = "arbitrary"
    UNKNOWN = "unknown"


class ParallelStrategy(str, Enum):
    NONE = "none"
    ULYSSES = "ulysses"
    RING = "ring"
    HYBRID_ULYSSES_RING = "hybrid_ulysses_ring"
    ALLGATHER_KV = "allgather_kv"


class OuterBoundary(str, Enum):
    HSDP = "hsdp"


@dataclass(frozen=True, slots=True)
class CapabilityResult:
    status: SupportStatus
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.status is SupportStatus.UNSUPPORTED and not self.reason:
            raise ValueError("UNSUPPORTED capability results require an actionable reason")

    @classmethod
    def supported(cls) -> CapabilityResult:
        return cls(SupportStatus.SUPPORTED)

    @classmethod
    def unsupported(cls, reason: str) -> CapabilityResult:
        return cls(SupportStatus.UNSUPPORTED, reason)

    @classmethod
    def unmigrated(cls, reason: str | None = None) -> CapabilityResult:
        return cls(SupportStatus.UNMIGRATED, reason)


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    platform: str
    kernel_variant: str | None = None
    dtype: str | None = None
    causal: bool | None = None
    mask_mode: MaskMode = MaskMode.NONE
    packing_mode: PackingMode = PackingMode.NONE
    piecewise: bool = False
    paged_kv: bool = False
    kv_cache_dtype: str | None = None
    parallel_strategy: ParallelStrategy = ParallelStrategy.NONE
    require_fullgraph: bool = False
    outer_boundaries: frozenset[OuterBoundary] = frozenset()


@dataclass(frozen=True, slots=True)
class ExecutionPathResult:
    backend: str
    path: str
    support: CapabilityResult
    compilation_mode: CompilationMode
    platform: str
    kernel_variant: str | None
    parallel_strategy: ParallelStrategy

    @classmethod
    def unmigrated(
        cls,
        backend: str,
        context: ExecutionContext,
        *,
        path: str = "unmigrated",
    ) -> ExecutionPathResult:
        return cls(
            backend=backend,
            path=path,
            support=CapabilityResult.unmigrated(),
            compilation_mode=CompilationMode.EAGER_ONLY,
            platform=context.platform,
            kernel_variant=context.kernel_variant,
            parallel_strategy=context.parallel_strategy,
        )

    def requested_support(self, context: ExecutionContext) -> CapabilityResult:
        """Validate request-level guarantees without changing path migration state."""
        if context.require_fullgraph and self.support.status is SupportStatus.UNMIGRATED:
            return CapabilityResult.unsupported(
                f"{self.backend} path {self.path!r} has no verified fullgraph declaration; "
                "use a verified path or disable the guaranteed-fullgraph request"
            )
        if self.support.status is not SupportStatus.SUPPORTED:
            return self.support
        if context.require_fullgraph and self.compilation_mode is CompilationMode.EAGER_ONLY:
            return CapabilityResult.unsupported(
                f"{self.backend} path {self.path!r} does not support guaranteed fullgraph compilation"
            )
        return CapabilityResult.supported()
