# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Typed semantic contracts shared by final-layout artifact representations."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Protocol

import torch

from vllm_omni.diffusion.models.host_weight_contract import (
    FINAL_LAYOUT_TENSOR_MODEL_CONTRACT_SCHEMA,
)
from vllm_omni.host_weight_runtime import (
    AdaptationIdentity,
    CanonicalJson,
    FailureCode,
    HostWeightError,
    HostWeightFailure,
    ProducerIdentity,
    ResolutionStage,
    TensorKind,
    WeightRepresentation,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .tensor_layout import RuntimeTensorTarget

FINAL_LAYOUT_TENSOR_RESTORER_SCHEMA = "diffusion-final-layout-tensor-restorer-v1"


def _require_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must not be empty")
    return value


def _require_positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


class FinalLayoutContractCode(str, Enum):
    SOURCE_CHANGED = "source_changed"
    SOURCE_COVERAGE_INVALID = "source_coverage_invalid"
    MODEL_CONTRACT_UNSUPPORTED = "model_contract_unsupported"
    OWNERSHIP_AMBIGUOUS = "ownership_ambiguous"
    TENSOR_UNSUPPORTED = "tensor_unsupported"
    DTYPE_UNSUPPORTED = "dtype_unsupported"
    IDENTITY_INCOMPATIBLE = "identity_incompatible"
    TENSOR_CONTRACT_CHANGED = "tensor_contract_changed"


class FinalLayoutContractError(ValueError):
    """Typed fail-closed error at a final-layout model integration boundary."""

    def __init__(self, code: FinalLayoutContractCode, message: str) -> None:
        if not isinstance(code, FinalLayoutContractCode):
            raise ValueError("final-layout contract errors require FinalLayoutContractCode")
        super().__init__(message)
        self.code = code


def final_layout_producer_error(error: FinalLayoutContractError) -> HostWeightError:
    """Translate representation failures into the runtime's stable taxonomy."""
    if error.code is FinalLayoutContractCode.SOURCE_CHANGED:
        stage = ResolutionStage.CANONICAL_LOADING
        code = FailureCode.CANONICAL_SOURCE_FAILED
    elif error.code in {
        FinalLayoutContractCode.MODEL_CONTRACT_UNSUPPORTED,
        FinalLayoutContractCode.TENSOR_UNSUPPORTED,
        FinalLayoutContractCode.DTYPE_UNSUPPORTED,
    }:
        stage = ResolutionStage.PRODUCTION
        code = FailureCode.PRODUCER_UNSUPPORTED
    else:
        stage = ResolutionStage.PRODUCTION
        code = FailureCode.PRODUCER_FAILED
    return HostWeightError(
        HostWeightFailure(
            stage=stage,
            code=code,
            retryable=False,
            message=str(error),
            details=CanonicalJson.from_value({"final_layout_code": error.code.value}),
        )
    )


@dataclass(frozen=True)
class ImplementationIdentity:
    """Explicit stable identity for one semantic implementation boundary."""

    implementation_id: str
    version: str
    fingerprint: str

    def __post_init__(self) -> None:
        _require_text("implementation_id", self.implementation_id)
        _require_text("implementation version", self.version)
        _require_text("implementation fingerprint", self.fingerprint)

    def to_dict(self) -> dict[str, str]:
        return {
            "implementation_id": self.implementation_id,
            "version": self.version,
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True)
class FinalLayoutLoaderIdentity:
    """Loader ABI and exact configuration facts that may change final bytes."""

    implementation: ImplementationIdentity
    model_config_fingerprint: str
    weight_transform_fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.implementation, ImplementationIdentity):
            raise ValueError("loader implementation must use ImplementationIdentity")
        _require_text("model config fingerprint", self.model_config_fingerprint)
        _require_text("weight transform fingerprint", self.weight_transform_fingerprint)

    def to_dict(self) -> dict[str, object]:
        return {
            "implementation": self.implementation.to_dict(),
            "model_config_fingerprint": self.model_config_fingerprint,
            "weight_transform_fingerprint": self.weight_transform_fingerprint,
        }


@dataclass(frozen=True)
class FinalLayoutParallelIdentity:
    """Semantic parallel layout with DP and SP rank intentionally absent."""

    tensor_parallel_size: int = 1
    tensor_parallel_rank: int = 0
    sequence_parallel_size: int = 1
    ulysses_degree: int = 1
    ring_degree: int = 1
    allgather_degree: int = 1
    ulysses_mode: str = "strict"
    pipeline_parallel_size: int = 1
    cfg_parallel_size: int = 1
    use_hsdp: bool = False
    enable_expert_parallel: bool = False

    def __post_init__(self) -> None:
        tp_size = _require_positive_int("tensor_parallel_size", self.tensor_parallel_size)
        if isinstance(self.tensor_parallel_rank, bool) or not isinstance(self.tensor_parallel_rank, int):
            raise ValueError("tensor_parallel_rank must be an integer")
        if not 0 <= self.tensor_parallel_rank < tp_size:
            raise ValueError("tensor_parallel_rank must be within tensor_parallel_size")
        for name, value in (
            ("sequence_parallel_size", self.sequence_parallel_size),
            ("ulysses_degree", self.ulysses_degree),
            ("ring_degree", self.ring_degree),
            ("allgather_degree", self.allgather_degree),
            ("pipeline_parallel_size", self.pipeline_parallel_size),
            ("cfg_parallel_size", self.cfg_parallel_size),
        ):
            _require_positive_int(name, value)
        _require_text("ulysses_mode", self.ulysses_mode)
        if not isinstance(self.use_hsdp, bool) or not isinstance(self.enable_expert_parallel, bool):
            raise ValueError("HSDP and expert-parallel flags must be booleans")

    @property
    def sequence_parallel_backend(self) -> str:
        if self.sequence_parallel_size == 1:
            return "none"
        active = [
            name
            for name, degree in (
                ("ulysses", self.ulysses_degree),
                ("ring", self.ring_degree),
                ("allgather", self.allgather_degree),
            )
            if degree > 1
        ]
        return "+".join(active) if active else "replicated-sp"

    def to_dict(self) -> dict[str, object]:
        return {
            "tensor_parallel_size": self.tensor_parallel_size,
            "tensor_parallel_rank": self.tensor_parallel_rank,
            "sequence_parallel_size": self.sequence_parallel_size,
            "sequence_parallel_backend": self.sequence_parallel_backend,
            "ulysses_degree": self.ulysses_degree,
            "ring_degree": self.ring_degree,
            "allgather_degree": self.allgather_degree,
            "ulysses_mode": self.ulysses_mode,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "cfg_parallel_size": self.cfg_parallel_size,
            "use_hsdp": self.use_hsdp,
            "enable_expert_parallel": self.enable_expert_parallel,
        }


@dataclass(frozen=True)
class FinalLayoutRequest:
    """Representation-independent semantic request assembled by the loader."""

    model_id: str
    loader: FinalLayoutLoaderIdentity
    parallel: FinalLayoutParallelIdentity = field(default_factory=FinalLayoutParallelIdentity)
    load_format: str = "default"
    adaptation: AdaptationIdentity = field(default_factory=AdaptationIdentity)

    def __post_init__(self) -> None:
        _require_text("canonical model ID or path", self.model_id)
        if not isinstance(self.loader, FinalLayoutLoaderIdentity):
            raise ValueError("loader must use FinalLayoutLoaderIdentity")
        if not isinstance(self.parallel, FinalLayoutParallelIdentity):
            raise ValueError("parallel must use FinalLayoutParallelIdentity")
        _require_text("load_format", self.load_format)
        if not isinstance(self.adaptation, AdaptationIdentity):
            raise ValueError("adaptation must use AdaptationIdentity")


def implementation_abi_fingerprint(abi: CanonicalJson) -> str:
    if not isinstance(abi, CanonicalJson):
        raise ValueError("implementation ABI must use CanonicalJson")
    return hashlib.sha256(abi.encoded).hexdigest()


@dataclass(frozen=True)
class FinalLayoutArtifactSpec:
    """Concrete representation, layout, producer, and restoration ABI."""

    representation: WeightRepresentation
    producer: ProducerIdentity
    implementation_abi: CanonicalJson
    layout_name: str = "diffusion-final-module-layout-v1"
    component_name: str = "diffusion-dit"
    component_ownership: str = "complete-final-layout-tensors"
    model_contract_schema: str = FINAL_LAYOUT_TENSOR_MODEL_CONTRACT_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.representation, WeightRepresentation):
            raise ValueError("artifact representation must use WeightRepresentation")
        if not isinstance(self.producer, ProducerIdentity):
            raise ValueError("artifact producer must use ProducerIdentity")
        if not isinstance(self.implementation_abi, CanonicalJson):
            raise ValueError("artifact implementation ABI must use CanonicalJson")
        for name, value in (
            ("layout_name", self.layout_name),
            ("component_name", self.component_name),
            ("component_ownership", self.component_ownership),
            ("model_contract_schema", self.model_contract_schema),
        ):
            _require_text(name, value)
        abi = self.implementation_abi.to_value()
        required_abi_fields = {
            "artifact_identity",
            "model_contract",
            "producer",
            "representation_policy",
            "restorer",
            "source_identity",
            "tensor_contract",
        }
        if not isinstance(abi, dict) or not required_abi_fields <= set(abi):
            raise ValueError(
                "artifact implementation ABI must version identity, source, tensor, policy, producer, "
                "restorer, and model contracts"
            )
        for name in required_abi_fields:
            _require_text(f"implementation ABI {name}", abi[name])
        if self.producer.implementation_fingerprint != implementation_abi_fingerprint(self.implementation_abi):
            raise ValueError("producer implementation fingerprint differs from its explicit ABI descriptor")


class FinalLayoutTensorPolicy(Protocol):
    """Representation-specific validation over shared final-layout mechanics."""

    @property
    def spec(self) -> FinalLayoutArtifactSpec: ...

    def validate_request(self, request: FinalLayoutRequest) -> None: ...

    def tensor_role(self, name: str, tensor: torch.Tensor, kind: TensorKind) -> str: ...

    def validate_target(self, target: RuntimeTensorTarget) -> None: ...

    def validate_collection(self, targets: Sequence[RuntimeTensorTarget]) -> None: ...

    def build_format_metadata(
        self,
        *,
        component_names: tuple[str, ...],
        tensor_contract_digest: str,
        tensor_count: int,
    ) -> CanonicalJson: ...

    def validate_format_metadata(
        self,
        metadata: CanonicalJson,
        *,
        component_names: tuple[str, ...],
        tensor_contract_digest: str,
        tensor_count: int,
    ) -> None: ...


__all__ = [
    "FinalLayoutArtifactSpec",
    "FinalLayoutContractCode",
    "FinalLayoutContractError",
    "FinalLayoutLoaderIdentity",
    "FinalLayoutParallelIdentity",
    "FinalLayoutRequest",
    "FinalLayoutTensorPolicy",
    "FINAL_LAYOUT_TENSOR_RESTORER_SCHEMA",
    "ImplementationIdentity",
    "final_layout_producer_error",
    "implementation_abi_fingerprint",
]
