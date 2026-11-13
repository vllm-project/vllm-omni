# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Generic validation-first restorer for exact final-layout tensor artifacts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

from vllm_omni.host_weight_runtime import HostWeightLease, TensorKind, WeightRestorePlan

from .identity_adapter import (
    FinalLayoutIdentityContext,
    validate_final_layout_identity,
)
from .tensor_layout import collect_final_layout_targets, validate_final_layout_model_contract


@dataclass(frozen=True)
class _Replacement:
    name: str
    parent: nn.Module
    leaf_name: str
    target: torch.Tensor
    source: torch.Tensor
    kind: TensorKind

    def current_target(self) -> torch.Tensor | None:
        if self.kind is TensorKind.PARAMETER:
            return self.parent._parameters.get(self.leaf_name)
        return self.parent._buffers.get(self.leaf_name)


class FinalLayoutTensorRestorePlan(WeightRestorePlan):
    """One-shot exact tensor rebinding validated before the first mutation."""

    def __init__(
        self,
        lease: HostWeightLease,
        replacements: tuple[_Replacement, ...],
        validators: tuple[Callable[[], None], ...],
        source_guard: Callable[[], None],
    ) -> None:
        self._lease = lease
        self._replacements = replacements
        self._validators = validators
        self._source_guard = source_guard
        self._committed = False

    def commit(self) -> None:
        if self._committed:
            raise RuntimeError("final-layout tensor restore plan was already committed")
        if self._lease.closed:
            raise RuntimeError("cannot commit a final-layout tensor restore from a closed lease")
        self._source_guard()

        for replacement in self._replacements:
            current = replacement.current_target()
            if current is not replacement.target:
                raise RuntimeError(f"restore target {replacement.name!r} changed after preflight")
            if (
                tuple(current.shape) != tuple(replacement.source.shape)
                or current.dtype != replacement.source.dtype
                or tuple(current.stride()) != tuple(replacement.source.stride())
                or not replacement.source.is_contiguous()
            ):
                raise RuntimeError(f"restore target {replacement.name!r} changed layout after preflight")

        # Mark first: assignment or model-validator failure makes the plan
        # permanently non-retryable and the partially restored model disposable.
        self._committed = True
        for replacement in self._replacements:
            if replacement.target.is_meta:
                if replacement.kind is TensorKind.PARAMETER:
                    replacement.parent._parameters[replacement.leaf_name] = nn.Parameter(
                        replacement.source,
                        requires_grad=replacement.target.requires_grad,
                    )
                else:
                    replacement.parent._buffers[replacement.leaf_name] = replacement.source
            else:
                replacement.target.data = replacement.source
        for validator in self._validators:
            validator()


class FinalLayoutTensorRestorer:
    """Validate and plan restoration for one exact policy-defined lease."""

    def __init__(self, context: FinalLayoutIdentityContext) -> None:
        if not context.dit_names or any(not name for name in context.dit_names):
            raise ValueError("final-layout restorer requires DiT component names")
        self._context = context

    @property
    def schema(self) -> str:
        return self._context.spec.producer.restorer_schema

    def plan_restore(self, model: object, lease: HostWeightLease) -> FinalLayoutTensorRestorePlan:
        if not isinstance(model, nn.Module):
            raise TypeError("final-layout restoration requires an nn.Module pipeline")
        if type(model) is not self._context.pipeline_type:
            raise ValueError("restore model implementation differs from the exact identity context")
        if lease.closed:
            raise ValueError("cannot restore from a closed HostWeightLease")
        if lease.identity.canonical_bytes != self._context.identity.canonical_bytes:
            raise ValueError("lease semantic identity differs from the exact restore request")
        if lease.manifest.identity.canonical_bytes != self._context.identity.canonical_bytes:
            raise ValueError("lease manifest identity differs from the exact restore request")
        if lease.identity.producer.restorer_schema != self.schema or lease.manifest.restorer_schema != self.schema:
            raise ValueError("lease restorer schema is incompatible with the exact restore request")
        self._context.ensure_sources_unchanged()

        try:
            dit_modules = tuple((name, model.get_submodule(name)) for name in self._context.dit_names)
        except AttributeError as exc:
            raise ValueError("one or more lease-owned DiT modules do not exist in the pipeline") from exc
        if tuple(type(module) for _, module in dit_modules) != self._context.dit_types:
            raise ValueError("restore DiT implementation differs from the exact identity context")
        bindings = validate_final_layout_model_contract(
            dit_modules,
            expected_schema=self._context.spec.model_contract_schema,
        )
        targets = collect_final_layout_targets(
            model,
            dit_modules,
            policy=self._context.policy,
            require_materialized=False,
        )
        contract_digest = validate_final_layout_identity(self._context, targets)

        manifest_entries = {entry.name: entry for entry in lease.manifest.tensors}
        target_names = {target.name for target in targets}
        if set(manifest_entries) != target_names or target_names != self._context.tensor_names:
            missing = sorted(target_names - set(manifest_entries))
            unexpected = sorted(set(manifest_entries) - target_names)
            raise ValueError(
                "lease tensor coverage differs from the structurally initialized DiT: "
                f"missing={missing[:5]}, unexpected={unexpected[:5]}"
            )
        self._context.policy.validate_format_metadata(
            lease.manifest.format_metadata,
            component_names=self._context.dit_names,
            tensor_contract_digest=contract_digest,
            tensor_count=len(manifest_entries),
        )

        replacements: list[_Replacement] = []
        source_storages: dict[tuple[int, int], str] = {}
        for target in targets:
            entry = manifest_entries[target.name]
            source = lease.tensors[target.name]
            if entry.kind is not target.kind or entry.role != target.role:
                raise ValueError(f"lease tensor kind or role differs for {target.name!r}")
            if (
                tuple(source.shape) != tuple(target.tensor.shape)
                or source.dtype != target.tensor.dtype
                or tuple(source.stride()) != tuple(target.tensor.stride())
                or not source.is_contiguous()
                or source.device.type != "cpu"
            ):
                raise ValueError(f"lease tensor layout differs for {target.name!r}")
            if source.numel():
                storage = source.untyped_storage()
                storage_id = (storage.data_ptr(), storage.nbytes())
                owner = source_storages.setdefault(storage_id, target.name)
                if owner != target.name:
                    raise ValueError(f"lease tensor {target.name!r} aliases {owner!r}")
            parent_path, _, leaf_name = target.name.rpartition(".")
            replacements.append(
                _Replacement(
                    name=target.name,
                    parent=model.get_submodule(parent_path),
                    leaf_name=leaf_name,
                    target=target.tensor,
                    source=source,
                    kind=target.kind,
                )
            )

        self._context.ensure_sources_unchanged()
        return FinalLayoutTensorRestorePlan(
            lease,
            tuple(replacements),
            tuple(binding.validator for binding in bindings),
            self._context.ensure_sources_unchanged,
        )


__all__ = [
    "FinalLayoutTensorRestorePlan",
    "FinalLayoutTensorRestorer",
]
