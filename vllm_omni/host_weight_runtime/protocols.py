# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Provider and consumer protocols for Host Weight Runtime extensions."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from types import TracebackType
from typing import TYPE_CHECKING, Protocol

import torch

from .identity import CanonicalJson, WeightArtifactIdentity
from .manifest import TensorKind

if TYPE_CHECKING:
    from .config import ValidationLevel
    from .lease import HostWeightLease
    from .outcomes import StoreResult


class ProductionSourceMode(str, Enum):
    FINALIZED_MODEL = "finalized_model"
    CHECKPOINT_STREAM = "checkpoint_stream"


class CoordinationScope(str, Enum):
    SINGLE_PROCESS = "single_process"
    GROUP_COHORT = "group_cohort"
    ARTIFACT_BUNDLE = "artifact_bundle"


class LookupPhase(str, Enum):
    PRE_LOAD_SAFE = "pre_load_safe"
    POST_LOAD_ONLY = "post_load_only"


@dataclass(frozen=True)
class TensorWriteSpec:
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    kind: TensorKind = TensorKind.PARAMETER
    role: str = "weight"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or not isinstance(self.role, str) or not self.role:
            raise ValueError("tensor write name and role must not be empty")
        if not isinstance(self.shape, tuple) or any(
            isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0 for dimension in self.shape
        ):
            raise ValueError(f"tensor {self.name!r} has a negative shape dimension")
        if not isinstance(self.dtype, torch.dtype):
            raise ValueError(f"tensor {self.name!r} must declare one torch dtype")
        if not isinstance(self.kind, TensorKind):
            raise ValueError(f"tensor {self.name!r} must use TensorKind")

    @property
    def numel(self) -> int:
        return math.prod(self.shape) if self.shape else 1

    @property
    def nbytes(self) -> int:
        return self.numel * torch.empty((), dtype=self.dtype).element_size()


class TensorFileWriter(Protocol):
    def write_tensor(self, name: str, tensor: torch.Tensor) -> None: ...

    def write_rows(self, name: str, start_row: int, tensor: torch.Tensor) -> None: ...

    def close(self) -> None: ...

    def __enter__(self) -> TensorFileWriter: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> bool | None: ...


class ArtifactWriter(Protocol):
    def open_tensor_file(
        self,
        relative_name: str,
        specs: tuple[TensorWriteSpec, ...],
    ) -> TensorFileWriter: ...


@dataclass(frozen=True)
class WeightProductionSpec:
    producer_id: str
    outputs: tuple[WeightArtifactIdentity, ...]
    source_mode: ProductionSourceMode
    coordination_scope: CoordinationScope = CoordinationScope.SINGLE_PROCESS
    lookup_phase: LookupPhase = LookupPhase.PRE_LOAD_SAFE

    def __post_init__(self) -> None:
        if not isinstance(self.producer_id, str) or not self.producer_id or not self.outputs:
            raise ValueError("producer_id and outputs must not be empty")
        if not isinstance(self.outputs, tuple) or any(
            not isinstance(output, WeightArtifactIdentity) for output in self.outputs
        ):
            raise ValueError("producer outputs must be an immutable tuple of artifact identities")
        if not isinstance(self.source_mode, ProductionSourceMode):
            raise ValueError("producer source mode must use ProductionSourceMode")
        if not isinstance(self.coordination_scope, CoordinationScope):
            raise ValueError("producer coordination scope must use CoordinationScope")
        if not isinstance(self.lookup_phase, LookupPhase):
            raise ValueError("producer lookup phase must use LookupPhase")
        if len({output.key for output in self.outputs}) != len(self.outputs):
            raise ValueError("producer outputs must have unique semantic identities")
        if any(output.producer.producer_id != self.producer_id for output in self.outputs):
            raise ValueError("producer outputs must name the declaring producer_id")


@dataclass(frozen=True)
class ProductionMetadata:
    producer_schema: str
    restorer_schema: str
    format_metadata: CanonicalJson = field(default_factory=CanonicalJson.empty)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.producer_schema, str)
            or not self.producer_schema
            or not isinstance(self.restorer_schema, str)
            or not self.restorer_schema
        ):
            raise ValueError("producer and restorer schema identifiers must not be empty")
        if not isinstance(self.format_metadata, CanonicalJson):
            raise ValueError("production format metadata must use CanonicalJson")


class WeightProducer(Protocol):
    """Synchronous V1 producer; callers do not preempt an active build."""

    @property
    def spec(self) -> WeightProductionSpec: ...

    def produce(self, writer: ArtifactWriter) -> ProductionMetadata:
        """Produce one exact artifact synchronously or raise."""
        ...


@dataclass(frozen=True)
class BuildRequest:
    identity: WeightArtifactIdentity


class WeightRestorePlan(Protocol):
    """A validated, one-shot model mutation plan."""

    def commit(self) -> None:
        """Mutate the target model exactly once or raise."""
        ...


class WeightRestorer(Protocol):
    """Validate a lease-to-model restore without mutating either input."""

    @property
    def schema(self) -> str: ...

    def plan_restore(self, model: object, lease: HostWeightLease) -> WeightRestorePlan:
        """Return a validated plan; all model mutation is deferred to commit()."""
        ...


class HostWeightStore(Protocol):
    """Exact artifact access whose deadlines bound lock acquisition only."""

    def lookup(
        self,
        identity: WeightArtifactIdentity,
        *,
        validation: ValidationLevel,
        deadline: float | None = None,
    ) -> StoreResult: ...

    def get_or_build(
        self,
        request: BuildRequest,
        producer: WeightProducer,
        *,
        validation: ValidationLevel,
        deadline: float,
    ) -> StoreResult: ...


@dataclass(frozen=True)
class RemoteArtifactDescriptor:
    identity: WeightArtifactIdentity
    provider_id: str
    manifest_sha256: str
    content_sha256: str


class RemoteArtifactProvider(Protocol):
    @property
    def provider_id(self) -> str: ...

    def probe(self, identity: WeightArtifactIdentity) -> RemoteArtifactDescriptor | None: ...

    def fetch(self, descriptor: RemoteArtifactDescriptor, writer: ArtifactWriter) -> ProductionMetadata: ...


__all__ = [
    "ArtifactWriter",
    "BuildRequest",
    "CoordinationScope",
    "HostWeightStore",
    "LookupPhase",
    "ProductionMetadata",
    "ProductionSourceMode",
    "RemoteArtifactDescriptor",
    "RemoteArtifactProvider",
    "TensorFileWriter",
    "TensorWriteSpec",
    "WeightProducer",
    "WeightProductionSpec",
    "WeightRestorePlan",
    "WeightRestorer",
]
