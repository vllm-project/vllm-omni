# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Representation-neutral ownership discovery for final-layout tensors."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from itertools import chain

import torch
from torch import nn

from vllm_omni.diffusion.models.host_weight_contract import FinalLayoutModelContract
from vllm_omni.host_weight_runtime import TensorKind
from vllm_omni.host_weight_runtime.identity import canonical_json

from .contracts import (
    FinalLayoutContractCode,
    FinalLayoutContractError,
    FinalLayoutTensorPolicy,
)


@dataclass(frozen=True)
class ModelRestoreBinding:
    """Model-declared restore ABI and its post-commit validator."""

    name: str
    contract: FinalLayoutModelContract
    validator: Callable[[], None]


@dataclass(frozen=True)
class RuntimeTensorTarget:
    """One complete final-layout tensor owned by a DiT component."""

    name: str
    tensor: torch.Tensor
    kind: TensorKind
    role: str

    @property
    def nbytes(self) -> int:
        return self.tensor.numel() * self.tensor.element_size()


def validate_final_layout_model_contract(
    dit_modules: Sequence[tuple[str, nn.Module]],
    *,
    expected_schema: str,
) -> tuple[ModelRestoreBinding, ...]:
    """Require an explicit tensor-complete restore ABI from every DiT."""
    if not dit_modules:
        raise FinalLayoutContractError(
            FinalLayoutContractCode.MODEL_CONTRACT_UNSUPPORTED,
            "final-layout artifacts require at least one DiT component",
        )

    bindings: list[ModelRestoreBinding] = []
    seen_names: set[str] = set()
    for name, module in dit_modules:
        if not name or name in seen_names:
            raise FinalLayoutContractError(
                FinalLayoutContractCode.OWNERSHIP_AMBIGUOUS,
                f"DiT component name {name!r} is empty or duplicated",
            )
        seen_names.add(name)
        contract = getattr(module, "host_weight_restore_contract", None)
        if not isinstance(contract, FinalLayoutModelContract) or contract.schema != expected_schema:
            raise FinalLayoutContractError(
                FinalLayoutContractCode.MODEL_CONTRACT_UNSUPPORTED,
                f"DiT component {name!r} does not declare restore schema {expected_schema!r}",
            )
        validator = getattr(module, "validate_restored_host_weights", None)
        if not callable(validator):
            raise FinalLayoutContractError(
                FinalLayoutContractCode.MODEL_CONTRACT_UNSUPPORTED,
                f"DiT component {name!r} does not implement validate_restored_host_weights()",
            )
        bindings.append(ModelRestoreBinding(name, contract, validator))
    return tuple(bindings)


def _named_parameters_with_duplicates(module: nn.Module) -> Iterator[tuple[str, nn.Parameter]]:
    try:
        return module.named_parameters(remove_duplicate=False)
    except TypeError:  # pragma: no cover - compatibility with old torch
        return module.named_parameters()


def _named_buffers_with_duplicates(module: nn.Module) -> Iterator[tuple[str, torch.Tensor]]:
    try:
        return module.named_buffers(remove_duplicate=False)
    except TypeError:  # pragma: no cover - compatibility with old torch
        return module.named_buffers()


def _is_persistent_buffer(module: nn.Module, local_name: str) -> bool:
    parent_path, _, leaf_name = local_name.rpartition(".")
    owner = module.get_submodule(parent_path)
    return leaf_name not in owner._non_persistent_buffers_set


def _resolve_pipeline_tensor(pipeline: nn.Module, runtime_name: str) -> torch.Tensor | None:
    parent_path, _, leaf_name = runtime_name.rpartition(".")
    parent = pipeline.get_submodule(parent_path)
    tensor = parent._parameters.get(leaf_name)
    if tensor is None:
        tensor = parent._buffers.get(leaf_name)
    return tensor


def collect_final_layout_targets(
    pipeline: nn.Module,
    dit_modules: Sequence[tuple[str, nn.Module]],
    *,
    policy: FinalLayoutTensorPolicy,
    require_materialized: bool,
) -> tuple[RuntimeTensorTarget, ...]:
    """Return complete, alias-free targets validated by one representation policy."""
    validate_final_layout_model_contract(
        dit_modules,
        expected_schema=policy.spec.model_contract_schema,
    )
    records: dict[str, RuntimeTensorTarget] = {}
    for dit_name, dit_module in dit_modules:
        for local_name, tensor in _named_parameters_with_duplicates(dit_module):
            runtime_name = f"{dit_name}.{local_name}"
            candidate = RuntimeTensorTarget(
                runtime_name,
                tensor,
                TensorKind.PARAMETER,
                policy.tensor_role(runtime_name, tensor, TensorKind.PARAMETER),
            )
            existing = records.get(runtime_name)
            if existing is not None and existing.tensor is not tensor:
                raise FinalLayoutContractError(
                    FinalLayoutContractCode.OWNERSHIP_AMBIGUOUS,
                    f"multiple DiT tensors resolve to {runtime_name!r}",
                )
            records[runtime_name] = candidate
        for local_name, tensor in _named_buffers_with_duplicates(dit_module):
            if not _is_persistent_buffer(dit_module, local_name):
                continue
            runtime_name = f"{dit_name}.{local_name}"
            candidate = RuntimeTensorTarget(
                runtime_name,
                tensor,
                TensorKind.BUFFER,
                policy.tensor_role(runtime_name, tensor, TensorKind.BUFFER),
            )
            existing = records.get(runtime_name)
            if existing is not None and existing.tensor is not tensor:
                raise FinalLayoutContractError(
                    FinalLayoutContractCode.OWNERSHIP_AMBIGUOUS,
                    f"multiple DiT tensors resolve to {runtime_name!r}",
                )
            records[runtime_name] = candidate

    if not records:
        raise FinalLayoutContractError(
            FinalLayoutContractCode.OWNERSHIP_AMBIGUOUS,
            "no DiT parameters or persistent buffers were discovered",
        )

    object_owners: dict[int, str] = {}
    storage_owners: dict[tuple[int, int], str] = {}
    for record in records.values():
        tensor = record.tensor
        if not record.role:
            raise FinalLayoutContractError(
                FinalLayoutContractCode.TENSOR_UNSUPPORTED,
                f"{record.name!r} has an empty semantic role",
            )
        try:
            pipeline_tensor = _resolve_pipeline_tensor(pipeline, record.name)
        except AttributeError as exc:
            raise FinalLayoutContractError(
                FinalLayoutContractCode.OWNERSHIP_AMBIGUOUS,
                f"DiT tensor {record.name!r} is not owned by the pipeline",
            ) from exc
        if pipeline_tensor is not tensor:
            raise FinalLayoutContractError(
                FinalLayoutContractCode.OWNERSHIP_AMBIGUOUS,
                f"DiT tensor {record.name!r} does not resolve to the discovered object",
            )

        object_owner = object_owners.setdefault(id(tensor), record.name)
        if object_owner != record.name:
            raise FinalLayoutContractError(
                FinalLayoutContractCode.TENSOR_UNSUPPORTED,
                f"{record.name!r} aliases tensor object {object_owner!r}",
            )
        if hasattr(tensor, "to_local"):
            raise FinalLayoutContractError(
                FinalLayoutContractCode.TENSOR_UNSUPPORTED,
                f"{record.name!r} is a distributed tensor",
            )
        if tensor.layout != torch.strided:
            raise FinalLayoutContractError(
                FinalLayoutContractCode.TENSOR_UNSUPPORTED,
                f"{record.name!r} uses unsupported layout {tensor.layout}",
            )
        if tensor.device.type not in {"cpu", "meta"}:
            raise FinalLayoutContractError(
                FinalLayoutContractCode.TENSOR_UNSUPPORTED,
                f"{record.name!r} must be a CPU or meta tensor, got {tensor.device}",
            )
        if not tensor.is_contiguous():
            raise FinalLayoutContractError(
                FinalLayoutContractCode.TENSOR_UNSUPPORTED,
                f"{record.name!r} is non-contiguous with stride {tensor.stride()}",
            )
        policy.validate_target(record)

        if not require_materialized:
            continue
        if tensor.device.type != "cpu" or tensor.is_meta:
            raise FinalLayoutContractError(
                FinalLayoutContractCode.TENSOR_UNSUPPORTED,
                f"{record.name!r} must be a materialized CPU tensor, got {tensor.device}",
            )
        storage = tensor.untyped_storage()
        if tensor.storage_offset() != 0 or storage.nbytes() != record.nbytes:
            raise FinalLayoutContractError(
                FinalLayoutContractCode.TENSOR_UNSUPPORTED,
                f"{record.name!r} is a view into a larger storage",
            )
        if record.nbytes:
            storage_id = (storage.data_ptr(), storage.nbytes())
            storage_owner = storage_owners.setdefault(storage_id, record.name)
            if storage_owner != record.name:
                raise FinalLayoutContractError(
                    FinalLayoutContractCode.TENSOR_UNSUPPORTED,
                    f"{record.name!r} shares storage with {storage_owner!r}",
                )

    for pipeline_name, tensor in chain(
        _named_parameters_with_duplicates(pipeline),
        _named_buffers_with_duplicates(pipeline),
    ):
        owner = object_owners.get(id(tensor))
        if owner is not None and owner != pipeline_name:
            raise FinalLayoutContractError(
                FinalLayoutContractCode.TENSOR_UNSUPPORTED,
                f"cached tensor {owner!r} aliases pipeline tensor {pipeline_name!r}",
            )
        if not require_materialized or tensor.device.type != "cpu" or tensor.is_meta or tensor.numel() == 0:
            continue
        storage = tensor.untyped_storage()
        pipeline_storage_owner = storage_owners.get((storage.data_ptr(), storage.nbytes()))
        if pipeline_storage_owner is not None and pipeline_storage_owner != pipeline_name:
            raise FinalLayoutContractError(
                FinalLayoutContractCode.TENSOR_UNSUPPORTED,
                f"cached tensor {pipeline_storage_owner!r} shares storage with pipeline tensor {pipeline_name!r}",
            )

    ordered = tuple(records[name] for name in sorted(records))
    policy.validate_collection(ordered)
    return ordered


def tensor_contract_sha256(records: Sequence[RuntimeTensorTarget]) -> str:
    """Hash exact structural ownership and representation-specific tensor roles."""
    contract = [
        {
            "dtype": str(record.tensor.dtype),
            "kind": record.kind.value,
            "name": record.name,
            "role": record.role,
            "shape": list(record.tensor.shape),
            "stride": list(record.tensor.stride()),
        }
        for record in records
    ]
    return hashlib.sha256(canonical_json(contract)).hexdigest()


__all__ = [
    "ModelRestoreBinding",
    "RuntimeTensorTarget",
    "collect_final_layout_targets",
    "tensor_contract_sha256",
    "validate_final_layout_model_contract",
]
