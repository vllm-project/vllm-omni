# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exact pre-load identity assembly for policy-defined final-layout artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from torch import nn

from vllm_omni.diffusion.models.host_weight_contract import FinalLayoutModelContract
from vllm_omni.host_weight_runtime import (
    CanonicalJson,
    ComponentIdentity,
    RuntimeWeightLayout,
    WeightArtifactIdentity,
    WeightRepresentation,
)

from .contracts import (
    FinalLayoutArtifactSpec,
    FinalLayoutContractCode,
    FinalLayoutContractError,
    FinalLayoutRequest,
    FinalLayoutTensorPolicy,
)
from .source_identity import (
    FinalLayoutSourceContext,
    NodeSourceDigestCache,
    PreparedWeightSource,
    resolve_final_layout_source_identity,
)
from .tensor_layout import (
    RuntimeTensorTarget,
    collect_final_layout_targets,
    tensor_contract_sha256,
    validate_final_layout_model_contract,
)


@dataclass(frozen=True)
class FinalLayoutIdentityContext:
    """Exact semantic identity plus process-local transaction guards."""

    identity: WeightArtifactIdentity
    spec: FinalLayoutArtifactSpec
    policy: FinalLayoutTensorPolicy
    request: FinalLayoutRequest
    tensor_names: frozenset[str]
    dit_names: tuple[str, ...]
    model_contracts: tuple[FinalLayoutModelContract, ...]
    source: FinalLayoutSourceContext
    pipeline_type: type[nn.Module]
    dit_types: tuple[type[nn.Module], ...]

    def sources_unchanged(self) -> bool:
        return self.source.sources_unchanged()

    def ensure_sources_unchanged(self) -> None:
        self.source.ensure_sources_unchanged()


def _require_metadata_object(name: str, metadata: CanonicalJson) -> Mapping[str, object]:
    value = metadata.to_value()
    if not isinstance(value, dict):
        raise ValueError(f"{name} must encode one JSON object")
    return value


def _model_contract_values(contracts: Sequence[FinalLayoutModelContract]) -> list[dict[str, str]]:
    return [contract.to_dict() for contract in contracts]


def build_final_layout_identity(
    pipeline: nn.Module,
    *,
    dit_modules: Sequence[tuple[str, nn.Module]],
    prepared_sources: Sequence[PreparedWeightSource],
    request: FinalLayoutRequest,
    policy: FinalLayoutTensorPolicy,
    source_digest_cache: NodeSourceDigestCache | None = None,
) -> FinalLayoutIdentityContext:
    """Build exact source, representation, layout, and implementation identity."""
    if not isinstance(request, FinalLayoutRequest):
        raise TypeError("final-layout identity requires FinalLayoutRequest")
    policy.validate_request(request)
    spec = policy.spec
    bindings = validate_final_layout_model_contract(
        dit_modules,
        expected_schema=spec.model_contract_schema,
    )
    structural_targets = collect_final_layout_targets(
        pipeline,
        dit_modules,
        policy=policy,
        require_materialized=False,
    )
    target_names = frozenset(target.name for target in structural_targets)
    source = resolve_final_layout_source_identity(
        prepared_sources,
        model_id=request.model_id,
        target_names=target_names,
        digest_cache=source_digest_cache,
    )
    dit_names = tuple(name for name, _ in dit_modules)
    model_contracts = tuple(binding.contract for binding in bindings)

    representation = WeightRepresentation(
        name=spec.representation.name,
        dtype=spec.representation.dtype,
        metadata=CanonicalJson.from_value(
            {
                "artifact_policy": _require_metadata_object(
                    "artifact representation metadata",
                    spec.representation.metadata,
                ),
                "load_format": request.load_format,
                "loader": request.loader.to_dict(),
            }
        ),
    )
    identity = WeightArtifactIdentity(
        schema_version=1,
        source=source.identity,
        component=ComponentIdentity(
            name=spec.component_name,
            ownership=spec.component_ownership,
            metadata=CanonicalJson.from_value(
                {
                    "component_names": list(dit_names),
                    "model_contracts": _model_contract_values(model_contracts),
                }
            ),
        ),
        representation=representation,
        layout=RuntimeWeightLayout(
            name=spec.layout_name,
            tensor_parallel_size=request.parallel.tensor_parallel_size,
            tensor_parallel_rank=request.parallel.tensor_parallel_rank,
            sequence_parallel_size=request.parallel.sequence_parallel_size,
            sequence_parallel_backend=request.parallel.sequence_parallel_backend,
            metadata=CanonicalJson.from_value(
                {
                    "model_contract_schema": spec.model_contract_schema,
                    "parallel": request.parallel.to_dict(),
                    "tensor_contract_sha256": tensor_contract_sha256(structural_targets),
                }
            ),
        ),
        adaptation=request.adaptation,
        producer=spec.producer,
    )
    return FinalLayoutIdentityContext(
        identity=identity,
        spec=spec,
        policy=policy,
        request=request,
        tensor_names=target_names,
        dit_names=dit_names,
        model_contracts=model_contracts,
        source=source,
        pipeline_type=type(pipeline),
        dit_types=tuple(type(module) for _, module in dit_modules),
    )


def validate_final_layout_identity(
    context: FinalLayoutIdentityContext,
    targets: Sequence[RuntimeTensorTarget],
) -> str:
    """Validate a context against current targets and return their digest."""
    identity = context.identity
    spec = context.spec
    request = context.request
    digest = tensor_contract_sha256(targets)
    if identity.source != context.source.identity or identity.adaptation != request.adaptation:
        raise FinalLayoutContractError(
            FinalLayoutContractCode.IDENTITY_INCOMPATIBLE,
            "identity source or adaptation differs from the exact context",
        )
    if identity.producer != spec.producer:
        raise FinalLayoutContractError(
            FinalLayoutContractCode.IDENTITY_INCOMPATIBLE,
            "identity producer differs from the artifact specification",
        )
    if identity.representation.name != spec.representation.name or identity.representation.dtype != (
        spec.representation.dtype
    ):
        raise FinalLayoutContractError(
            FinalLayoutContractCode.IDENTITY_INCOMPATIBLE,
            "identity representation differs from the artifact specification",
        )
    if identity.component.name != spec.component_name or identity.component.ownership != spec.component_ownership:
        raise FinalLayoutContractError(
            FinalLayoutContractCode.IDENTITY_INCOMPATIBLE,
            "identity component ownership differs from the artifact specification",
        )
    if identity.layout.name != spec.layout_name:
        raise FinalLayoutContractError(
            FinalLayoutContractCode.IDENTITY_INCOMPATIBLE,
            "identity runtime layout differs from the artifact specification",
        )

    component = _require_metadata_object("component metadata", identity.component.metadata)
    representation = _require_metadata_object("representation metadata", identity.representation.metadata)
    layout = _require_metadata_object("layout metadata", identity.layout.metadata)
    if representation != {
        "artifact_policy": _require_metadata_object(
            "artifact representation metadata",
            spec.representation.metadata,
        ),
        "load_format": request.load_format,
        "loader": request.loader.to_dict(),
    }:
        raise FinalLayoutContractError(
            FinalLayoutContractCode.IDENTITY_INCOMPATIBLE,
            "identity loader or representation policy differs from the exact context",
        )
    if component.get("component_names") != list(context.dit_names) or component.get("model_contracts") != (
        _model_contract_values(context.model_contracts)
    ):
        raise FinalLayoutContractError(
            FinalLayoutContractCode.IDENTITY_INCOMPATIBLE,
            "identity model ownership differs from the exact context",
        )
    if layout.get("model_contract_schema") != spec.model_contract_schema:
        raise FinalLayoutContractError(
            FinalLayoutContractCode.IDENTITY_INCOMPATIBLE,
            "identity model restore schema differs from the artifact specification",
        )
    if (
        identity.layout.tensor_parallel_size != request.parallel.tensor_parallel_size
        or identity.layout.tensor_parallel_rank != request.parallel.tensor_parallel_rank
        or identity.layout.sequence_parallel_size != request.parallel.sequence_parallel_size
        or identity.layout.sequence_parallel_backend != request.parallel.sequence_parallel_backend
        or layout.get("parallel") != request.parallel.to_dict()
    ):
        raise FinalLayoutContractError(
            FinalLayoutContractCode.IDENTITY_INCOMPATIBLE,
            "identity parallel semantics differ from the exact context",
        )
    if layout.get("tensor_contract_sha256") != digest:
        raise FinalLayoutContractError(
            FinalLayoutContractCode.TENSOR_CONTRACT_CHANGED,
            "identity tensor contract differs from the current model",
        )
    if frozenset(target.name for target in targets) != context.tensor_names:
        raise FinalLayoutContractError(
            FinalLayoutContractCode.TENSOR_CONTRACT_CHANGED,
            "current tensor ownership differs from the exact identity context",
        )
    return digest


__all__ = [
    "FinalLayoutIdentityContext",
    "build_final_layout_identity",
    "validate_final_layout_identity",
]
