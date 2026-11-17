# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ascend storage IPC adapter. Importing this module initializes no device.

Use ``TorchIpcBackend(codec=TorchNpuIpcCodec())`` on both peers. The backend
owns synchronization, single-consumption receipts and uncertain-outcome retention.
Only ordinary ND/NCHW storages in native, non-expandable allocations are supported;
internal formats (including FRACTAL_NZ), private pools and imported allocations
are rejected without converting or copying weights.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from typing import cast
from uuid import UUID

from .contracts import CompatibilityError, PreparedWeights, SessionContext, StorageExport, UnsupportedError, WeightSpec
from .torch_cuda_ipc_codec import TorchCudaIpcCodec, _allocator_segments, _describe_bindings, _native_blocks

_CODEC_ID = "npu-ipc-v1"
_ACL_IPC_EVENT_HANDLE_SIZE = 64


def _device_uuid(device_index: int) -> str:
    import torch_npu

    # Ask the runtime about the process-local ordinal. Environment variables can
    # reorder visible devices and must not be mistaken for physical identities.
    properties = torch_npu.npu.get_device_properties(device_index)
    try:
        uuid = UUID(str(properties.uuid))
    except (AttributeError, ValueError) as exc:
        raise UnsupportedError("NPU IPC requires a runtime-provided physical device UUID") from exc
    if not uuid.int:
        raise UnsupportedError("NPU IPC requires a nonzero physical device UUID")
    return f"NPU-{uuid}"


def _validate_runtime(context: SessionContext) -> None:
    if sys.platform != "linux":
        raise UnsupportedError("NPU weight sharing requires Linux and torch_npu")
    import torch_npu

    if not torch_npu.npu.is_available():
        raise UnsupportedError("an available Ascend NPU is required")
    if _device_uuid(context.device_index) != context.identity.device_uuid:
        raise CompatibilityError("physical NPU UUID differs")
    torch_npu.npu.set_device(context.device_index)


def _validate_descriptor(descriptor: StorageExport) -> None:
    if descriptor.codec != _CODEC_ID:
        raise UnsupportedError(f"unsupported NPU IPC codec: {descriptor.codec}")
    if not descriptor.storage_id:
        raise CompatibilityError("storage identity must be nonempty")
    integers = (
        descriptor.device_index,
        descriptor.storage_nbytes,
        descriptor.allocation_offset_bytes,
        descriptor.ref_counter_offset,
    )
    if any(type(value) is not int or value < 0 for value in integers) or descriptor.storage_nbytes == 0:
        raise CompatibilityError("invalid NPU IPC storage size, device or offset")
    if any(
        not isinstance(handle, bytes) or not 0 < len(handle) <= 4096
        for handle in (descriptor.handle, descriptor.ref_counter_handle, descriptor.event_handle)
    ):
        raise CompatibilityError("invalid NPU IPC handle")
    if type(descriptor.event_sync_required) is not bool:
        raise CompatibilityError("invalid NPU IPC event state")
    if descriptor.event_sync_required and len(descriptor.event_handle) != _ACL_IPC_EVENT_HANDLE_SIZE:
        raise CompatibilityError("NPU IPC event handle must match CANN's 64-byte ABI")


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
        raise UnsupportedError("unexpected torch_npu reducer ABI for a byte carrier")
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


class TorchNpuIpcCodec(TorchCudaIpcCodec):
    """Reuse device-neutral view reconstruction and C++ storage lifetime tracking.

    All device operations are overridden; no CUDA runtime is used. The wire
    descriptor has the same fields, with a distinct codec ID to reject mixing.
    The caller keeps finalized weights read-only until the provider closes.
    """

    def inspect(self, bindings: Mapping[str, object], spec: WeightSpec, context: SessionContext) -> PreparedWeights:
        import torch
        import torch_npu

        _validate_runtime(context)
        if torch_npu.npu.memory.get_allocator_backend() != "native":
            raise UnsupportedError("only the native torch_npu caching allocator is supported")
        segments = _allocator_segments(torch_npu.npu.memory._snapshot())
        blocks = _native_blocks([s for s in segments if s.get("is_expandable") is False], context.device_index)
        for name, tensor in bindings.items():
            if not isinstance(tensor, torch.Tensor) or tensor.device != torch.device("npu", context.device_index):
                raise CompatibilityError(f"binding is not on the session's NPU device: {name}")
            # NPU StorageImpl also carries a physical format. A logical shape
            # and stride cannot reconstruct padded/tiled internal formats.
            if str(torch_npu.get_npu_format(tensor)) not in ("ND", "NCHW"):
                raise UnsupportedError(f"only ND/NCHW NPU formats are supported: {name}")
            storage = tensor.untyped_storage()
            if storage.nbytes() and (not storage.resizable() or blocks.get(storage.data_ptr(), 0) < storage.nbytes()):
                raise UnsupportedError(
                    "storage must belong to a live native allocation outside private pools; "
                    "create weights with PYTORCH_NPU_ALLOC_CONF=expandable_segments:False"
                )
        return _describe_bindings(bindings, spec)

    def validate_runtime(self, context: SessionContext) -> None:
        # The consumer maps provider-owned memory; its allocator need not match.
        _validate_runtime(context)

    def synchronize(self, context: SessionContext) -> None:
        import torch_npu

        torch_npu.npu.synchronize(context.device_index)

    def export_storage(self, carrier: object, storage_id: str) -> StorageExport:
        import torch
        from torch.multiprocessing.reductions import reduce_tensor
        from torch_npu.multiprocessing.reductions import rebuild_npu_tensor

        if type(carrier) is not torch.Tensor:
            raise UnsupportedError("a plain tensor byte carrier is required")
        if (
            carrier.device.type != "npu"
            or carrier.dtype != torch.uint8
            or carrier.storage_offset() != 0
            or carrier.shape != (carrier.untyped_storage().nbytes(),)
            or carrier.stride() != (1,)
            or carrier.numel() == 0
        ):
            raise UnsupportedError("a nonempty full-storage NPU byte carrier is required")
        rebuild, args = reduce_tensor(carrier)
        if rebuild is not rebuild_npu_tensor:
            raise UnsupportedError("unexpected NPU reducer rebuild function")
        return _encode_reduction(args, storage_id, carrier.numel())

    def import_storage(self, descriptor: StorageExport, device_index: int) -> object:
        # Validate before even loading torch_npu or invoking a native operation.
        _validate_descriptor(descriptor)
        if type(device_index) is not int or device_index < 0:
            raise CompatibilityError("consumer device index must be a nonnegative integer")
        import torch
        from torch_npu.multiprocessing.reductions import rebuild_npu_tensor

        return rebuild_npu_tensor(
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
        import torch_npu  # noqa: F401 - register the NPU device

        return torch.empty(0, dtype=torch.uint8, device=torch.device("npu", device_index))

    def cancel_unconsumed(self, descriptor: StorageExport, device_index: int) -> None:
        _validate_descriptor(descriptor)
        import torch
        import torch_npu  # noqa: F401 - register NPU storage methods

        # Only an unclaimed export is cancellable. Uncertain imports stay held
        # by the backend, just as they do for the CUDA codec.
        torch.storage.TypedStorage._release_ipc_counter_npu(
            descriptor.ref_counter_handle, descriptor.ref_counter_offset, device=device_index
        )
