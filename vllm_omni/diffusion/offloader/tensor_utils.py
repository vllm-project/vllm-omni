# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared tensor utilities for distributed layerwise offload.

These helpers are used by both DistributedLayerwiseOffloadHook and
DistributedLayerwiseOffloadBackend, and can be reused by other
offload backends.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from itertools import chain
from typing import Any

import torch
from torch import nn
from torch.distributed.tensor import DTensor


@dataclass(frozen=True)
class TensorStorageSpec:
    """One tensor's target and physical-storage layout."""

    name: str
    target: torch.Tensor
    value: torch.Tensor
    storage_numel: int
    stride: tuple[int, ...]


@dataclass(frozen=True)
class ModuleTensors:
    """Registered bindings owned by the block transport."""

    parameters: dict[str, nn.Parameter]
    buffers: dict[str, torch.Tensor]


@dataclass(frozen=True)
class OffloadTensorInventory:
    """Partition independently staged CPU modules from DLO-owned tensors."""

    _independent_ids: frozenset[int] = frozenset()

    @property
    def has_independent(self) -> bool:
        return bool(self._independent_ids)

    @classmethod
    def resolve(
        cls,
        pipeline: nn.Module,
        paths: frozenset[str],
    ) -> OffloadTensorInventory:
        if not paths:
            return cls()
        ordered = sorted(paths)
        owners: dict[int, str] = {}
        for index, path in enumerate(ordered):
            nested = any(path.startswith(p + ".") for p in ordered[:index])
            if not path or nested:
                raise ValueError("Independent offload module roots must not overlap")
            try:
                root = pipeline.get_submodule(path)
            except AttributeError as exc:
                raise ValueError(f"Independent root not found: {path!r}") from exc
            tensors = [*root.parameters(), *root.buffers()]
            cpu_only = all(t.device.type == "cpu" for t in tensors)
            valid = cpu_only and all(map(is_materialized_tensor, tensors))
            if not tensors or not valid:
                raise ValueError(f"Independent root needs real CPU weights: {path!r}")
            for tensor in tensors:
                previous = owners.setdefault(id(tensor), path)
                if previous != path:
                    raise ValueError("Independent roots must not share tensors")

        storage_owners: dict[tuple[Any, ...], str | None] = {}
        bindings = chain(
            pipeline.named_parameters(remove_duplicate=False),
            pipeline.named_buffers(remove_duplicate=False),
        )
        for name, tensor in bindings:
            matches = (p for p in ordered if name.startswith(p + "."))
            owner = next(matches, None)
            if owners.get(id(tensor)) != owner:
                raise ValueError(f"Tensor {name!r} aliases different offload owners")
            local = tensor.to_local() if is_dtensor(tensor) else tensor
            if local.is_meta or not local.numel():
                continue
            storage_key = (local.device, local.untyped_storage().data_ptr())
            if storage_key in storage_owners and storage_owners[storage_key] != owner:
                raise ValueError(f"Storage crosses offload owners: {name!r}")
            storage_owners[storage_key] = owner
        return cls(frozenset(owners))

    def owned(self, module: nn.Module) -> ModuleTensors:
        params = dict(module.named_parameters())
        buffers = dict(module.named_buffers())
        excluded = self._independent_ids
        return ModuleTensors(
            {name: t for name, t in params.items() if id(t) not in excluded},
            {name: t for name, t in buffers.items() if id(t) not in excluded},
        )


def dtype_size(dtype: torch.dtype) -> int:
    """Return element size in bytes for a torch.dtype."""
    return torch.empty(1, dtype=dtype).element_size()


def group_named_tensors_by_dtype(
    params: Mapping[str, torch.Tensor],
    buffers: Mapping[str, torch.Tensor],
) -> dict[torch.dtype, list[tuple[str, torch.Tensor]]]:
    """Group parameters and buffers by dtype while preserving their order."""
    grouped: dict[torch.dtype, list[tuple[str, torch.Tensor]]] = {}
    for name, tensor in chain(params.items(), buffers.items()):
        grouped.setdefault(tensor.dtype, []).append((name, tensor))
    return grouped


def physical_storage_numel(tensor: torch.Tensor) -> int:
    """Return the storage span needed to preserve ``tensor``'s stride."""
    if tensor.numel() == 0:
        return 0
    return 1 + sum((size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride()))


def flatten_physical_storage(
    tensor: torch.Tensor,
    storage_numel: int | None = None,
) -> torch.Tensor:
    """Pack a possibly strided tensor in physical storage order."""
    if tensor.is_contiguous():
        return tensor.flatten()
    storage_numel = physical_storage_numel(tensor) if storage_numel is None else storage_numel
    storage = torch.zeros(storage_numel, dtype=tensor.dtype, device=tensor.device)
    torch.as_strided(storage, size=tensor.shape, stride=tensor.stride()).copy_(tensor)
    return storage


def describe_tensor_storage(
    named_tensors: Iterable[tuple[str, torch.Tensor]],
    transforms: Mapping[int, Callable[[torch.Tensor], torch.Tensor]] | None = None,
) -> list[TensorStorageSpec]:
    """Resolve local tensors and their transport-preserving layouts."""
    specs: list[TensorStorageSpec] = []
    for name, target in named_tensors:
        value = target.to_local() if hasattr(target, "to_local") else target
        transform = (transforms or {}).get(id(target))
        if callable(transform):
            value = transform(value)
        specs.append(
            TensorStorageSpec(
                name=name,
                target=target,
                value=value,
                storage_numel=physical_storage_numel(value),
                stride=value.stride(),
            )
        )
    return specs


def tensor_storage_metadata(spec: TensorStorageSpec, offset: int, *, include_device: bool = False) -> dict[str, Any]:
    """Build reconstruction metadata shared by offload transports."""
    metadata: dict[str, Any] = {
        "name": spec.name,
        "offset": offset,
        "numel": spec.storage_numel,
        "shape": spec.value.shape,
        "stride": spec.stride,
    }
    if include_device:
        metadata["device"] = spec.value.device
    return metadata


def is_dtensor(t: torch.Tensor) -> bool:
    """Check if tensor is a DTensor."""
    return isinstance(t, DTensor)


def set_tensor_storage(target: torch.Tensor, value: torch.Tensor) -> None:
    """Replace target's underlying storage with value (zero-copy)."""
    if is_dtensor(target):
        target._local_tensor = value
    else:
        target.data = value


def make_offload_placeholder(tensor: torch.Tensor) -> torch.Tensor:
    """Create a zero-element placeholder to free GPU memory."""
    if is_dtensor(tensor):
        local_shape = tuple(tensor.to_local().shape)
        return torch.empty(local_shape, device="meta", dtype=tensor.dtype)
    return torch.empty((0,), device=tensor.device, dtype=tensor.dtype)


def clear_tensor_storage(tensors: Iterable[torch.Tensor]) -> None:
    """Atomically replace tensor storage with offload placeholders."""
    targets = list(tensors)
    originals = [tensor.to_local() if is_dtensor(tensor) else tensor.data for tensor in targets]
    placeholders = [make_offload_placeholder(tensor) for tensor in targets]
    committed = 0
    try:
        for target, placeholder in zip(targets, placeholders, strict=True):
            set_tensor_storage(target, placeholder)
            committed += 1
    except BaseException:
        rollback_error: BaseException | None = None
        for target, original in zip(targets[:committed], originals[:committed], strict=True):
            try:
                set_tensor_storage(target, original)
            except BaseException as exc:
                rollback_error = rollback_error or exc
        if rollback_error is not None:
            raise RuntimeError("Tensor storage clear failed and rollback was incomplete") from rollback_error
        raise


def clear_block_storage(
    params: Mapping[str, torch.Tensor],
    buffers: Mapping[str, torch.Tensor],
    ready_event: Any | None,
) -> None:
    """Wait for an outstanding transfer, then release one block's residency."""
    if ready_event is not None:
        from vllm_omni.platforms import current_omni_platform

        current_omni_platform.current_stream().wait_event(ready_event)
    clear_tensor_storage(chain(params.values(), buffers.values()))


def is_materialized_tensor(t: torch.Tensor) -> bool:
    """Check if tensor holds real data (not meta or empty placeholder)."""
    if is_dtensor(t):
        local_t = t.to_local()
        return not local_t.is_meta
    return not t.is_meta and t.data.numel() > 0


def materialization_probe(
    params: Mapping[str, torch.Tensor],
    buffers: Mapping[str, torch.Tensor],
) -> torch.Tensor | None:
    """Pick one non-empty tensor that tracks a block's atomic storage state."""
    return next((tensor for tensor in chain(params.values(), buffers.values()) if tensor.numel()), None)


def module_materialization_probe(module: nn.Module) -> torch.Tensor | None:
    """Capture a probe before another hook releases the module's storage."""
    return materialization_probe(dict(module.named_parameters()), dict(module.named_buffers()))


def restore_tensor_storage(
    target: torch.Tensor,
    value: torch.Tensor,
    *,
    device: torch.device | str,
) -> None:
    """Detach ``target`` from an offloader-owned backing with a real copy."""
    restored = torch.empty_strided(
        value.shape,
        value.stride(),
        dtype=value.dtype,
        device=device,
    )
    restored.copy_(value)
    set_tensor_storage(target, restored)
