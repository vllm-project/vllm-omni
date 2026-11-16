# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Versioned PyTorch CUDA IPC codec; importing this module does not load torch.

The backend owns exported references and their exactly-once disposition. In particular,
an exception from reduce/rebuild does *not* establish whether a counter was
created or consumed. The caller must retain resources and poison that session.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from threading import Lock
from typing import TYPE_CHECKING, cast
from uuid import UUID

from .contracts import (
    CompatibilityError,
    PreparedWeights,
    SessionContext,
    StorageExport,
    StorageSpec,
    TensorViewSpec,
    UnsupportedError,
    WeightManifest,
    WeightSpec,
    dtype_size,
)

if TYPE_CHECKING:
    import torch
    from torch.multiprocessing.reductions import StorageWeakRef


# Depends on PyTorch 2.13.0's reducer, storage-sharing and allocator interfaces.
_CODEC_ID = "cuda-ipc-v1"
# Expandable allocations encode multiple physical handles in one opaque payload.
_MAX_MEMORY_HANDLE_BYTES = 1024 * 1024

# Cache successful physical-GPU checks, never NVML handles. The PID prevents
# inherited metadata from being reused by a different process.
_validated_gpu_uuids: set[tuple[int, str]] = set()
_gpu_uuid_lock = Lock()


def _byte_carrier(tensor: torch.Tensor) -> torch.Tensor:
    """Reference the complete untyped storage, including bytes outside a view."""
    import torch

    storage = tensor.untyped_storage()
    return torch.empty(0, dtype=torch.uint8, device=tensor.device).set_(storage, 0, (storage.nbytes(),), (1,))


def _describe_bindings(bindings: Mapping[str, object], spec: WeightSpec) -> PreparedWeights:
    """Describe ordinary strided tensors using zero-copy storage views.

    The backend validates ``spec`` before this call and the resulting manifest
    before export. ``inspect`` adds the CUDA device and allocator restrictions.
    """
    import torch

    if set(bindings) != {tensor.name for tensor in spec.tensors}:
        raise CompatibilityError("bindings must exactly match the weight specification")
    storage_ids: dict[int, str] = {}
    carriers: dict[str, object] = {}
    storages: list[StorageSpec] = []
    views: list[TensorViewSpec] = []
    for entry in spec.tensors:
        value = bindings[entry.name]
        if type(value) not in (torch.Tensor, torch.nn.Parameter):
            raise UnsupportedError(f"unsupported tensor subclass: {entry.name}")
        tensor = cast("torch.Tensor", value)
        if tensor.layout != torch.strided or tensor.is_nested or tensor.device.type == "meta":
            raise UnsupportedError(f"only materialized dense strided tensors are supported: {entry.name}")
        if tensor.is_conj() or tensor.is_neg() or tensor.is_quantized:
            raise UnsupportedError(f"lazy conjugate/negative and quantized tensors are unsupported: {entry.name}")
        storage = tensor.untyped_storage()
        storage_key = storage._cdata
        storage_id = storage_ids.get(storage_key)
        if storage_id is None:
            storage_id = f"storage-{len(storage_ids)}"
            storage_ids[storage_key] = storage_id
            storages.append(StorageSpec(storage_id, storage.nbytes()))
            carriers[storage_id] = _byte_carrier(tensor)
        views.append(
            TensorViewSpec(
                entry.name,
                storage_id,
                str(tensor.dtype).removeprefix("torch."),
                tuple(tensor.shape),
                tuple(tensor.stride()),
                int(tensor.storage_offset() * tensor.element_size()),
            )
        )
    manifest = WeightManifest(spec, tuple(storages), tuple(views))
    return PreparedWeights(manifest, carriers, tuple(bindings.values()))


def _allocator_segments(snapshot: object) -> list[dict[str, object]]:
    """Read allocation segments from a native allocator snapshot."""
    if not isinstance(snapshot, dict):
        raise UnsupportedError("unsupported allocator snapshot")
    segments = snapshot.get("segments")
    if not isinstance(segments, list) or any(not isinstance(segment, dict) for segment in segments):
        raise UnsupportedError("unsupported allocator segment metadata")
    return cast(list[dict[str, object]], segments)


def _native_blocks(segments: list[dict[str, object]], device_index: int) -> dict[int, int]:
    """Index supported live native allocations once for the entire model."""
    result: dict[int, int] = {}
    for segment in segments:
        if segment.get("device") != device_index:
            continue
        if segment.get("segment_pool_id") not in ((0, 0), [0, 0]):
            continue
        blocks = segment.get("blocks")
        if not isinstance(blocks, list):
            raise UnsupportedError("unsupported allocator block metadata")
        for block in blocks:
            if not isinstance(block, dict) or block.get("state") != "active_allocated":
                continue
            address, size = block.get("address"), block.get("size")
            if type(address) is not int or type(size) is not int or address <= 0 or size <= 0:
                raise UnsupportedError("invalid allocator block metadata")
            result[address] = size
    return result


def _canonical_gpu_uuid(value: str) -> str:
    # PyTorch's CUuuid string is bare 8-4-4-4-12 hex; NVML uses the GPU- prefix.
    try:
        return f"GPU-{UUID(value.removeprefix('GPU-'))}"
    except ValueError as exc:
        raise UnsupportedError("a full physical NVIDIA GPU UUID is required") from exc


def _device_uuid(device_index: int) -> str:
    # This helper expects a torch-visible ordinal. Platform.get_device_uuid
    # expects a vLLM logical index, which can differ in assigned-GPU workers.
    from vllm.utils.import_utils import import_pynvml
    from vllm.utils.platform_utils import cuda_get_device_properties

    (uuid,) = cuda_get_device_properties(device_index, ("uuid",), init_cuda=True)
    device_uuid = _canonical_gpu_uuid(str(uuid))
    # CUDA can expose a MIG instance while NVML's physical index identifies its
    # parent. Validate each actual UUID once; repeated NVML init/shutdown can
    # hang on some driver stacks when interleaved with CUDA IPC operations.
    key = (os.getpid(), device_uuid)
    with _gpu_uuid_lock:
        if key not in _validated_gpu_uuids:
            nvml = import_pynvml()
            nvml.nvmlInit()
            try:
                handle = nvml.nvmlDeviceGetHandleByUUID(device_uuid)
                if nvml.nvmlDeviceIsMigDeviceHandle(handle):
                    raise UnsupportedError("MIG instances are unsupported")
            except nvml.NVMLError as exc:
                raise UnsupportedError("cannot resolve the CUDA UUID to a full physical GPU") from exc
            finally:
                nvml.nvmlShutdown()
            _validated_gpu_uuids.add(key)
    return device_uuid


def _runtime_snapshot(context: SessionContext) -> list[dict[str, object]]:
    import torch

    if sys.platform != "linux" or torch.version.cuda is None or torch.version.hip is not None:
        raise UnsupportedError("CUDA weight sharing requires Linux and NVIDIA CUDA")
    if torch.accelerator.current_accelerator(check_available=True) != torch.device("cuda"):
        raise UnsupportedError("an available NVIDIA CUDA accelerator is required")

    device_uuid = _device_uuid(context.device_index)
    if device_uuid != context.identity.device_uuid:
        raise CompatibilityError("physical GPU UUID differs")
    torch.accelerator.set_device_index(context.device_index)
    if torch._C._cuda_getAllocatorBackend() != "native":
        raise UnsupportedError("only the native PyTorch caching allocator is supported")
    return _allocator_segments(torch._C._cuda_memorySnapshot(None))


def _validate_descriptor(descriptor: StorageExport) -> None:
    if descriptor.codec != _CODEC_ID:
        raise UnsupportedError(f"unsupported CUDA IPC codec: {descriptor.codec}")
    if not descriptor.storage_id:
        raise CompatibilityError("storage identity must be nonempty")
    integers = (
        descriptor.device_index,
        descriptor.storage_nbytes,
        descriptor.allocation_offset_bytes,
        descriptor.ref_counter_offset,
    )
    if any(type(value) is not int or value < 0 for value in integers) or descriptor.storage_nbytes == 0:
        raise CompatibilityError("invalid CUDA IPC storage size, device or offset")
    for handle, limit in (
        (descriptor.handle, _MAX_MEMORY_HANDLE_BYTES),
        (descriptor.ref_counter_handle, 4096),
        (descriptor.event_handle, 4096),
    ):
        if not isinstance(handle, bytes) or not 0 < len(handle) <= limit:
            raise CompatibilityError("invalid CUDA IPC handle")
    if type(descriptor.event_sync_required) is not bool:
        raise CompatibilityError("invalid CUDA IPC event state")


def _encode_reduction(args: tuple[object, ...], storage_id: str, nbytes: int) -> StorageExport:
    import torch

    if len(args) != 15 or (
        args[0] is not torch.Tensor
        or args[1] != (nbytes,)
        or args[2] != (1,)
        or args[3] != 0
        or args[4] is not torch.storage.TypedStorage
        or args[5] != torch.uint8
        or args[8] != nbytes
        or args[10] is not False
    ):
        raise UnsupportedError("unexpected PyTorch CUDA reducer ABI for a byte carrier")
    descriptor = StorageExport(
        storage_id=storage_id,
        codec=_CODEC_ID,
        device_index=cast(int, args[6]),
        handle=cast(bytes, args[7]),
        storage_nbytes=cast(int, args[8]),
        allocation_offset_bytes=cast(int, args[9]),
        ref_counter_handle=cast(bytes, args[11]),
        ref_counter_offset=cast(int, args[12]),
        event_handle=cast(bytes, args[13]),
        event_sync_required=cast(bool, args[14]),
    )
    _validate_descriptor(descriptor)
    return descriptor


class _TorchStorageLease:
    """Track C++ StorageImpl lifetime, including views held outside the backend."""

    def __init__(self, carrier: torch.Tensor):
        from torch.multiprocessing.reductions import StorageWeakRef

        self._reference: StorageWeakRef | None = StorageWeakRef(carrier.untyped_storage())

    def expired(self) -> bool:
        return self._reference is None or self._reference.expired()

    def close(self) -> None:
        self._reference = None


class TorchCudaIpcCodec:
    """A narrow adapter for native PyTorch storages, including expandable segments.

    Methods are called by the backend's owner thread. No method starts a thread,
    repacks tensors, or falls back to copying unsupported allocations.
    PyTorch owns expandable handle export/import and mapping. Its POSIX FD path
    requires permission to use pidfd_open/pidfd_getfd between the worker processes;
    allocations must have IPC handles enabled when created.
    """

    def inspect(self, bindings: Mapping[str, object], spec: WeightSpec, context: SessionContext) -> PreparedWeights:
        import torch

        segments = _runtime_snapshot(context)
        for name, tensor in bindings.items():
            if not isinstance(tensor, torch.Tensor) or tensor.device != torch.device("cuda", context.device_index):
                raise CompatibilityError(f"binding is not on the session's CUDA device: {name}")
        prepared = _describe_bindings(bindings, spec)
        blocks = _native_blocks(segments, context.device_index)
        for value in prepared.carriers.values():
            carrier = cast("torch.Tensor", value)
            storage = carrier.untyped_storage()
            if storage.nbytes() and (not storage.resizable() or blocks.get(storage.data_ptr(), 0) < storage.nbytes()):
                raise UnsupportedError("storage must belong to a live native allocation outside private pools")
        return prepared

    def validate_runtime(self, context: SessionContext) -> None:
        """Check the consumer environment once, before importing any storage."""
        _runtime_snapshot(context)

    def synchronize(self, context: SessionContext) -> None:
        import torch

        torch.accelerator.synchronize(context.device_index)

    def export_storage(self, carrier: object, storage_id: str) -> StorageExport:
        import torch
        from torch.multiprocessing.reductions import rebuild_cuda_tensor, reduce_tensor

        if type(carrier) is not torch.Tensor:
            raise UnsupportedError("a plain tensor byte carrier is required")
        if (
            carrier.device.type != "cuda"
            or carrier.dtype != torch.uint8
            or carrier.storage_offset() != 0
            or carrier.shape != (carrier.untyped_storage().nbytes(),)
            or carrier.stride() != (1,)
            or carrier.numel() == 0
        ):
            raise UnsupportedError("a nonempty full-storage CUDA byte carrier is required")
        rebuild, args = reduce_tensor(carrier)
        if rebuild is not rebuild_cuda_tensor:
            raise UnsupportedError("unexpected CUDA reducer rebuild function")
        return _encode_reduction(args, storage_id, carrier.numel())

    def import_storage(self, descriptor: StorageExport, device_index: int) -> object:
        import torch
        from torch.multiprocessing.reductions import rebuild_cuda_tensor

        _validate_descriptor(descriptor)
        if type(device_index) is not int or device_index < 0:
            raise CompatibilityError("consumer device index must be a nonnegative integer")
        # PyTorch 2.13.0 stores an integer ordinal, not torch.device.
        # Never accept a remote callable or class: those values are fixed here.
        return rebuild_cuda_tensor(
            torch.Tensor,
            (descriptor.storage_nbytes,),
            (1,),
            0,
            torch.storage.TypedStorage,
            torch.uint8,
            device_index,
            descriptor.handle,
            descriptor.storage_nbytes,
            descriptor.allocation_offset_bytes,
            False,
            descriptor.ref_counter_handle,
            descriptor.ref_counter_offset,
            descriptor.event_handle,
            descriptor.event_sync_required,
        )

    def empty_storage(self, device_index: int) -> object:
        import torch

        return torch.empty(0, dtype=torch.uint8, device=torch.device("cuda", device_index))

    def make_views(self, manifest: WeightManifest, carriers: Mapping[str, object]) -> dict[str, object]:
        """Reconstruct views from a manifest already validated by the backend."""
        import torch

        if set(carriers) != {storage.storage_id for storage in manifest.storages}:
            raise CompatibilityError("mapped storage identities differ from the manifest")
        for storage in manifest.storages:
            value = carriers[storage.storage_id]
            if type(value) is not torch.Tensor:
                raise CompatibilityError("a mapped byte carrier must be a plain tensor")
            carrier = cast("torch.Tensor", value)
            if carrier.untyped_storage().nbytes() != storage.nbytes:
                raise CompatibilityError("mapped storage size differs from the manifest")
        result: dict[str, object] = {}
        views: dict[tuple[object, ...], torch.Tensor] = {}
        for view in manifest.views:
            key = view.storage_view_key()
            tensor = views.get(key)
            if tensor is None:
                carrier = cast("torch.Tensor", carriers[view.storage_id])
                dtype = getattr(torch, view.dtype)  # validate_manifest restricts dtype to a fixed list.
                tensor = torch.empty(0, dtype=dtype, device=carrier.device).set_(
                    carrier.untyped_storage(), view.offset_bytes // dtype_size(view.dtype), view.shape, view.stride
                )
                views[key] = tensor
            result[view.name] = tensor
        return result

    def weak_storage(self, carrier: object) -> _TorchStorageLease:
        import torch

        if type(carrier) is not torch.Tensor:
            raise UnsupportedError("a plain tensor byte carrier is required")
        return _TorchStorageLease(carrier)

    def cancel_unconsumed(self, descriptor: StorageExport, device_index: int) -> None:
        import torch

        _validate_descriptor(descriptor)
        # Only the backend's session state can establish that no rebuild consumed
        # this reference. This primitive is never used for unknown/failed imports.
        torch.storage.TypedStorage._release_ipc_counter(
            descriptor.ref_counter_handle, descriptor.ref_counter_offset, device=device_index
        )
