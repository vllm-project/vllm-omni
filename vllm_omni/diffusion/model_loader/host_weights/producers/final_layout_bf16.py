# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Concrete BF16-with-preserved-FP32 final-layout artifact policy and producer."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import ExitStack
from typing import TYPE_CHECKING

import torch
from safetensors import safe_open
from torch import nn

from vllm_omni.diffusion.models.host_weight_contract import (
    FINAL_LAYOUT_TENSOR_MODEL_CONTRACT_SCHEMA,
)
from vllm_omni.host_weight_runtime import (
    ArtifactWriter,
    CanonicalJson,
    CoordinationScope,
    LookupPhase,
    ProducerIdentity,
    ProductionMetadata,
    ProductionSourceMode,
    TensorKind,
    TensorWriteSpec,
    WeightProductionSpec,
    WeightRepresentation,
)

from ..contracts import (
    FINAL_LAYOUT_TENSOR_RESTORER_SCHEMA,
    FinalLayoutArtifactSpec,
    FinalLayoutContractCode,
    FinalLayoutContractError,
    FinalLayoutRequest,
    final_layout_producer_error,
    implementation_abi_fingerprint,
)
from ..identity_adapter import (
    FinalLayoutIdentityContext,
    validate_final_layout_identity,
)
from ..tensor_layout import (
    RuntimeTensorTarget,
    collect_final_layout_targets,
    validate_final_layout_model_contract,
)

if TYPE_CHECKING:
    from vllm_omni.diffusion.model_loader.host_weight_plan import HostWeightPlan

FINAL_LAYOUT_BF16_PRODUCER_ID = "vllm-omni.diffusion.final-layout-bf16"
FINAL_LAYOUT_BF16_VERSION = "1"
FINAL_LAYOUT_BF16_REPRESENTATION = "diffusion-final-layout-bf16"
FINAL_LAYOUT_BF16_MANIFEST_SCHEMA = "diffusion-final-layout-bf16-manifest-v1"
DEFAULT_SHARD_SIZE_BYTES = 5 * 1024**3

_BF16_PARAMETER_DTYPES = {torch.bfloat16, torch.float32}
_BF16_BUFFER_DTYPES = {
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
}
for _dtype_name in (
    "float8_e4m3fn",
    "float8_e5m2",
    "float8_e4m3fnuz",
    "float8_e5m2fnuz",
):
    if (_dtype := getattr(torch, _dtype_name, None)) is not None:
        _BF16_BUFFER_DTYPES.add(_dtype)

_BF16_IMPLEMENTATION_ABI = CanonicalJson.from_value(
    {
        "artifact_identity": "diffusion-final-layout-identity-v1",
        "model_contract": FINAL_LAYOUT_TENSOR_MODEL_CONTRACT_SCHEMA,
        "producer": "finalized-bf16-tensor-writer-v1",
        "representation_policy": "bf16-with-preserved-fp32-v1",
        "restorer": "exact-final-layout-tensor-rebind-v1",
        "source_identity": "prepared-diffusion-weight-source-v2",
        "tensor_contract": "complete-strided-tensor-ownership-v1",
    }
)

FINAL_LAYOUT_BF16_SPEC = FinalLayoutArtifactSpec(
    representation=WeightRepresentation(
        name=FINAL_LAYOUT_BF16_REPRESENTATION,
        dtype=str(torch.bfloat16),
        metadata=CanonicalJson.from_value(
            {
                "format": FINAL_LAYOUT_BF16_REPRESENTATION,
                "mixed_precision_policy": "bf16-with-preserved-fp32",
            }
        ),
    ),
    producer=ProducerIdentity(
        producer_id=FINAL_LAYOUT_BF16_PRODUCER_ID,
        version=FINAL_LAYOUT_BF16_VERSION,
        implementation_fingerprint=implementation_abi_fingerprint(_BF16_IMPLEMENTATION_ABI),
        manifest_schema=FINAL_LAYOUT_BF16_MANIFEST_SCHEMA,
        restorer_schema=FINAL_LAYOUT_TENSOR_RESTORER_SCHEMA,
    ),
    implementation_abi=_BF16_IMPLEMENTATION_ABI,
)


class FinalLayoutBF16Policy:
    """Concrete compatibility and metadata policy for finalized BF16 tensors."""

    @property
    def spec(self) -> FinalLayoutArtifactSpec:
        return FINAL_LAYOUT_BF16_SPEC

    def validate_request(self, request: FinalLayoutRequest) -> None:
        if request.load_format != "default":
            raise ValueError(f"final-layout BF16 artifacts require load_format='default', got {request.load_format!r}")
        if request.adaptation.kind != "base" or request.adaptation.fingerprint is not None:
            raise ValueError("merged or static LoRA requires a representation-specific producer")
        parallel = request.parallel
        if parallel.pipeline_parallel_size != 1:
            raise ValueError("pipeline-parallel ownership is not supported by the final-layout BF16 producer")
        if parallel.cfg_parallel_size != 1:
            raise ValueError("CFG-parallel ownership is not supported by the final-layout BF16 producer")
        if parallel.use_hsdp:
            raise ValueError("HSDP/DTensor layouts are not supported by the final-layout BF16 producer")
        if parallel.enable_expert_parallel:
            raise ValueError("expert-parallel ownership is not supported by the final-layout BF16 producer")

    def tensor_role(self, name: str, tensor: torch.Tensor, kind: TensorKind) -> str:
        del name, tensor
        if kind is TensorKind.PARAMETER:
            return "weight"
        return "persistent_buffer"

    def validate_target(self, target: RuntimeTensorTarget) -> None:
        if target.kind is TensorKind.PARAMETER and target.tensor.dtype not in _BF16_PARAMETER_DTYPES:
            raise FinalLayoutContractError(
                FinalLayoutContractCode.DTYPE_UNSUPPORTED,
                f"{target.name!r} must be BF16 or an explicitly preserved FP32 parameter, got {target.tensor.dtype}",
            )
        if target.kind is TensorKind.BUFFER and target.tensor.dtype not in _BF16_BUFFER_DTYPES:
            raise FinalLayoutContractError(
                FinalLayoutContractCode.DTYPE_UNSUPPORTED,
                f"{target.name!r} uses unsupported buffer dtype {target.tensor.dtype}",
            )

    def validate_collection(self, targets: Sequence[RuntimeTensorTarget]) -> None:
        if not any(target.kind is TensorKind.PARAMETER and target.tensor.dtype is torch.bfloat16 for target in targets):
            raise FinalLayoutContractError(
                FinalLayoutContractCode.DTYPE_UNSUPPORTED,
                "final-layout BF16 representation requires at least one BF16 parameter",
            )

    def build_format_metadata(
        self,
        *,
        component_names: tuple[str, ...],
        tensor_contract_digest: str,
        tensor_count: int,
    ) -> CanonicalJson:
        return CanonicalJson.from_value(
            {
                "component_names": list(component_names),
                "format": FINAL_LAYOUT_BF16_REPRESENTATION,
                "mixed_precision_policy": "bf16-with-preserved-fp32",
                "tensor_contract_sha256": tensor_contract_digest,
                "tensor_count": tensor_count,
                "tensor_layout": "contiguous",
            }
        )

    def validate_format_metadata(
        self,
        metadata: CanonicalJson,
        *,
        component_names: tuple[str, ...],
        tensor_contract_digest: str,
        tensor_count: int,
    ) -> None:
        expected = self.build_format_metadata(
            component_names=component_names,
            tensor_contract_digest=tensor_contract_digest,
            tensor_count=tensor_count,
        )
        if metadata != expected:
            raise ValueError("lease format metadata differs from the BF16 artifact policy")


FINAL_LAYOUT_BF16_POLICY = FinalLayoutBF16Policy()


def _split_shards(
    specs: Sequence[TensorWriteSpec],
    max_shard_bytes: int,
) -> tuple[tuple[TensorWriteSpec, ...], ...]:
    if max_shard_bytes <= 0:
        raise ValueError("final-layout BF16 shard size must be positive")
    shards: list[tuple[TensorWriteSpec, ...]] = []
    current: list[TensorWriteSpec] = []
    current_bytes = 0
    for spec in specs:
        if current and current_bytes + spec.nbytes > max_shard_bytes:
            shards.append(tuple(current))
            current = []
            current_bytes = 0
        current.append(spec)
        current_bytes += spec.nbytes
    if current:
        shards.append(tuple(current))
    return tuple(shards)


def _write_shards(
    writer: ArtifactWriter,
    specs: Sequence[TensorWriteSpec],
    *,
    max_shard_bytes: int,
    tensor_for: Callable[[str], torch.Tensor],
) -> None:
    shards = _split_shards(specs, max_shard_bytes)
    shard_count = len(shards)
    for index, shard in enumerate(shards, start=1):
        file_name = f"model-{index:05d}-of-{shard_count:05d}.safetensors"
        with writer.open_tensor_file(file_name, shard, ordered=True) as output:
            for spec in shard:
                output.write_tensor(spec.name, tensor_for(spec.name))


def _tensor_specs(records: Sequence[RuntimeTensorTarget]) -> tuple[TensorWriteSpec, ...]:
    return tuple(
        TensorWriteSpec(
            name=record.name,
            shape=tuple(record.tensor.shape),
            dtype=record.tensor.dtype,
            kind=record.kind,
            role=record.role,
        )
        for record in records
    )


def _production_metadata(
    context: FinalLayoutIdentityContext,
    contract_digest: str,
    tensor_count: int,
) -> ProductionMetadata:
    return ProductionMetadata(
        producer_schema=FINAL_LAYOUT_BF16_MANIFEST_SCHEMA,
        restorer_schema=FINAL_LAYOUT_TENSOR_RESTORER_SCHEMA,
        format_metadata=FINAL_LAYOUT_BF16_POLICY.build_format_metadata(
            component_names=context.dit_names,
            tensor_contract_digest=contract_digest,
            tensor_count=tensor_count,
        ),
    )


class FinalLayoutBF16Producer:
    """Publish finalized BF16-policy tensors through a store-scoped writer."""

    def __init__(
        self,
        context: FinalLayoutIdentityContext,
        pipeline: nn.Module,
        dit_modules: Sequence[tuple[str, nn.Module]],
        *,
        max_shard_bytes: int = DEFAULT_SHARD_SIZE_BYTES,
    ) -> None:
        if max_shard_bytes <= 0:
            raise ValueError("final-layout BF16 shard size must be positive")
        if context.spec != FINAL_LAYOUT_BF16_SPEC:
            raise ValueError("producer requires the exact BF16 artifact specification")
        self._context = context
        self._pipeline = pipeline
        self._dit_modules = tuple(dit_modules)
        self._max_shard_bytes = max_shard_bytes
        if tuple(name for name, _ in self._dit_modules) != context.dit_names:
            raise ValueError("producer DiT components differ from the exact identity context")
        if type(pipeline) is not context.pipeline_type or tuple(type(module) for _, module in self._dit_modules) != (
            context.dit_types
        ):
            raise ValueError("producer model implementation differs from the exact identity context")
        self._spec = WeightProductionSpec(
            producer_id=FINAL_LAYOUT_BF16_PRODUCER_ID,
            outputs=(context.identity,),
            source_mode=ProductionSourceMode.FINALIZED_MODEL,
            coordination_scope=CoordinationScope.SINGLE_PROCESS,
            lookup_phase=LookupPhase.POST_LOAD_ONLY,
        )

    @property
    def spec(self) -> WeightProductionSpec:
        return self._spec

    def produce(self, writer: ArtifactWriter) -> ProductionMetadata:
        try:
            return self._produce(writer)
        except FinalLayoutContractError as exc:
            raise final_layout_producer_error(exc) from exc

    def _produce(self, writer: ArtifactWriter) -> ProductionMetadata:
        self._context.ensure_sources_unchanged()
        bindings = validate_final_layout_model_contract(
            self._dit_modules,
            expected_schema=self._context.spec.model_contract_schema,
        )
        records = collect_final_layout_targets(
            self._pipeline,
            self._dit_modules,
            policy=FINAL_LAYOUT_BF16_POLICY,
            require_materialized=True,
        )
        contract_digest = validate_final_layout_identity(self._context, records)
        for binding in bindings:
            binding.validator()

        tensors = {record.name: record.tensor.detach() for record in records}
        _write_shards(
            writer,
            _tensor_specs(records),
            max_shard_bytes=self._max_shard_bytes,
            tensor_for=tensors.__getitem__,
        )

        self._context.ensure_sources_unchanged()
        return _production_metadata(self._context, contract_digest, len(records))


class CheckpointPlanBF16Producer:
    """Stream one validated checkpoint plan directly into an HWR artifact."""

    def __init__(
        self,
        context: FinalLayoutIdentityContext,
        pipeline: nn.Module,
        dit_modules: Sequence[tuple[str, nn.Module]],
        plan: HostWeightPlan,
        *,
        max_shard_bytes: int = DEFAULT_SHARD_SIZE_BYTES,
    ) -> None:
        self._context = context
        self._plan = plan
        self._max_shard_bytes = max_shard_bytes
        records = collect_final_layout_targets(
            pipeline,
            dit_modules,
            policy=FINAL_LAYOUT_BF16_POLICY,
            require_materialized=False,
        )
        self._contract_digest = validate_final_layout_identity(context, records)
        self._tensor_specs = _tensor_specs(records)
        self._spec = WeightProductionSpec(
            producer_id=FINAL_LAYOUT_BF16_PRODUCER_ID,
            outputs=(context.identity,),
            source_mode=ProductionSourceMode.CHECKPOINT_STREAM,
            coordination_scope=CoordinationScope.SINGLE_PROCESS,
            lookup_phase=LookupPhase.PRE_LOAD_SAFE,
        )

    @property
    def spec(self) -> WeightProductionSpec:
        return self._spec

    def produce(self, writer: ArtifactWriter) -> ProductionMetadata:
        try:
            return self._produce(writer)
        except FinalLayoutContractError as exc:
            raise final_layout_producer_error(exc) from exc

    def _produce(self, writer: ArtifactWriter) -> ProductionMetadata:
        self._context.ensure_sources_unchanged()

        with ExitStack() as stack:
            handles = {
                file_path: stack.enter_context(safe_open(file_path, framework="pt", device="cpu"))
                for file_path in sorted({binding.file_path for binding in self._plan.bindings.values()})
            }

            def checkpoint_tensor(name: str) -> torch.Tensor:
                binding = self._plan.bindings[name]
                tensor = handles[binding.file_path].get_tensor(binding.checkpoint_key)
                if binding.transform is not None:
                    tensor = binding.transform(tensor)
                return tensor.detach()

            _write_shards(
                writer,
                self._tensor_specs,
                max_shard_bytes=self._max_shard_bytes,
                tensor_for=checkpoint_tensor,
            )

        self._context.ensure_sources_unchanged()
        return _production_metadata(self._context, self._contract_digest, len(self._tensor_specs))


__all__ = [
    "DEFAULT_SHARD_SIZE_BYTES",
    "FINAL_LAYOUT_BF16_MANIFEST_SCHEMA",
    "FINAL_LAYOUT_BF16_POLICY",
    "FINAL_LAYOUT_BF16_PRODUCER_ID",
    "FINAL_LAYOUT_BF16_REPRESENTATION",
    "FINAL_LAYOUT_BF16_SPEC",
    "FINAL_LAYOUT_BF16_VERSION",
    "CheckpointPlanBF16Producer",
    "FinalLayoutBF16Policy",
    "FinalLayoutBF16Producer",
]
