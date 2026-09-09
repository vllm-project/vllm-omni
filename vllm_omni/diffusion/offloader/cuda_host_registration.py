# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA implementation of read-only host-mapping registration."""

from __future__ import annotations

import ctypes
import mmap
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import torch

from vllm_omni.host_weight_runtime import MappedHostRegion

from .host_registration import (
    HostRegistrationBudgetError,
    HostRegistrationCleanupError,
    HostRegistrationError,
)

_CUDA_HOST_REGISTER_READ_ONLY = 0x08
_CUDA_DEVICE_ATTRIBUTE_HOST_REGISTER_READ_ONLY_SUPPORTED = 113


class _CudaRuntime(Protocol):
    def cudaHostRegister(self, address: int, size: int, flags: int) -> int: ...

    def cudaHostUnregister(self, address: int) -> int: ...

    def cudaGetErrorString(self, error: int) -> str | bytes: ...

    def cudaGetLastError(self) -> int: ...


@dataclass(frozen=True)
class _AddressRange:
    start: int
    end: int

    @property
    def size(self) -> int:
        return self.end - self.start


def _coalesce_ranges(
    ranges: Sequence[tuple[int, int]],
    page_size: int = mmap.PAGESIZE,
) -> tuple[_AddressRange, ...]:
    """Page-align and merge overlapping ranges from one backing mapping."""
    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")

    aligned: list[_AddressRange] = []
    for start, size in ranges:
        if start <= 0 or size < 0:
            raise ValueError(f"invalid host range start={start}, size={size}")
        if size == 0:
            continue
        aligned_start = start - start % page_size
        end = start + size
        aligned_end = ((end + page_size - 1) // page_size) * page_size
        aligned.append(_AddressRange(aligned_start, aligned_end))

    merged: list[_AddressRange] = []
    for region in sorted(aligned, key=lambda item: (item.start, item.end)):
        if merged and region.start <= merged[-1].end:
            previous = merged[-1]
            merged[-1] = _AddressRange(previous.start, max(previous.end, region.end))
        else:
            merged.append(region)
    return tuple(merged)


def _mapped_regions(regions: Sequence[MappedHostRegion]) -> tuple[_AddressRange, ...]:
    """Resolve page spans without merging unrelated artifact files."""
    by_file: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for region in regions:
        if not region.read_only:
            raise HostRegistrationError(
                f"CUDA direct H2D requires an immutable mapping, but {region.file_name!r} is writable"
            )
        by_file[region.file_name].append((region.address, region.length))
    return tuple(span for ranges in by_file.values() for span in _coalesce_ranges(ranges))


def _error_message(runtime: _CudaRuntime, error: int) -> str:
    try:
        message = runtime.cudaGetErrorString(error)
    except Exception:
        return str(error)
    if isinstance(message, bytes):
        return message.decode(errors="replace")
    return str(message)


def _consume_last_cuda_error(runtime: _CudaRuntime, expected_error: int) -> None:
    """Clear one handled CUDA Runtime error before returning to PyTorch."""
    try:
        get_last_error = getattr(runtime, "cudaGetLastError", None)
        if get_last_error is None:
            get_last_error = ctypes.CDLL(None).cudaGetLastError
            get_last_error.argtypes = []
            get_last_error.restype = ctypes.c_int
        pending_error = int(get_last_error())
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise HostRegistrationError("cannot clear CUDA's pending error after host-registration failure") from exc

    if pending_error not in (0, expected_error):
        raise HostRegistrationError(
            f"host registration returned CUDA error {expected_error}, but cudaGetLastError reported {pending_error}"
        )


def _handled_error_message(runtime: _CudaRuntime, error: int) -> str:
    error_code = int(error)
    message = _error_message(runtime, error_code)
    try:
        _consume_last_cuda_error(runtime, error_code)
    except HostRegistrationError as exc:
        raise HostRegistrationError(f"{message}; {exc}") from exc
    return message


def _supports_read_only_host_registration(runtime: _CudaRuntime) -> bool:
    """Query support required by immutable artifact mappings."""
    try:
        get_attribute = getattr(runtime, "cudaDeviceGetAttribute", None)
        if get_attribute is None:
            get_attribute = ctypes.CDLL(None).cudaDeviceGetAttribute
            get_attribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
            get_attribute.restype = ctypes.c_int
    except (AttributeError, OSError) as exc:
        raise HostRegistrationError("cannot query CUDA read-only host-registration support") from exc

    supported = ctypes.c_int()
    try:
        error = get_attribute(
            ctypes.byref(supported),
            _CUDA_DEVICE_ATTRIBUTE_HOST_REGISTER_READ_ONLY_SUPPORTED,
            torch.accelerator.current_device_index(),
        )
    except Exception as exc:
        raise HostRegistrationError(f"cannot query CUDA read-only host-registration support: {exc}") from exc
    if int(error) != 0:
        raise HostRegistrationError(
            f"cudaDeviceGetAttribute(read-only host registration) failed: {_handled_error_message(runtime, error)}"
        )
    return bool(supported.value)


class CudaHostRegistration:
    """Own CUDA registrations for immutable file-backed CPU regions."""

    def __init__(self, runtime: _CudaRuntime, regions: tuple[_AddressRange, ...]) -> None:
        self._runtime = runtime
        self._regions = regions

    @classmethod
    def create(
        cls,
        regions: Sequence[MappedHostRegion],
        *,
        max_bytes: int | None,
    ) -> CudaHostRegistration:
        mapped = _mapped_regions(regions)
        total_bytes = sum(region.size for region in mapped)
        if max_bytes is not None and total_bytes > max_bytes:
            raise HostRegistrationBudgetError(
                f"mapped host ranges need {total_bytes} bytes, exceeding the {max_bytes}-byte registration budget"
            )
        if not mapped:
            raise HostRegistrationError("no non-empty host ranges were available for registration")
        if not torch.cuda.is_available():
            raise HostRegistrationError("CUDA is not available")

        try:
            runtime = torch.cuda.cudart()
        except Exception as exc:
            raise HostRegistrationError(f"cannot access the CUDA runtime: {exc}") from exc
        if not _supports_read_only_host_registration(runtime):
            raise HostRegistrationError(
                "CUDA device does not support read-only host registration required by immutable HWR mappings"
            )

        registered: list[_AddressRange] = []
        try:
            for region in mapped:
                error = runtime.cudaHostRegister(region.start, region.size, _CUDA_HOST_REGISTER_READ_ONLY)
                if int(error) != 0:
                    raise HostRegistrationError(
                        "cudaHostRegister(read-only) failed for "
                        f"[{region.start:#x}, {region.end:#x}): {_handled_error_message(runtime, error)}"
                    )
                registered.append(region)
        except Exception as exc:
            rollback_errors: list[str] = []
            rollback_failed: list[_AddressRange] = []
            for region in reversed(registered):
                try:
                    error = runtime.cudaHostUnregister(region.start)
                    if int(error) != 0:
                        rollback_errors.append(
                            f"cudaHostUnregister({region.start:#x}) failed: {_handled_error_message(runtime, error)}"
                        )
                        rollback_failed.append(region)
                except Exception as rollback_exc:
                    rollback_errors.append(f"cudaHostUnregister({region.start:#x}) raised: {rollback_exc}")
                    rollback_failed.append(region)
            if rollback_errors:
                active_registration = cls(runtime, tuple(reversed(rollback_failed)))
                raise HostRegistrationCleanupError(
                    f"CUDA host registration failed ({exc}); rollback errors: {rollback_errors[:3]}",
                    active_registration=active_registration,
                ) from exc
            if isinstance(exc, HostRegistrationError):
                raise
            raise HostRegistrationError(f"CUDA host registration raised: {exc}") from exc

        return cls(runtime, mapped)

    @property
    def total_bytes(self) -> int:
        return sum(region.size for region in self._regions)

    @property
    def region_count(self) -> int:
        return len(self._regions)

    def close(self) -> tuple[str, ...]:
        """Unregister every range and retain failures for a later retry."""
        errors: list[str] = []
        failed: list[_AddressRange] = []
        for region in reversed(self._regions):
            try:
                error = self._runtime.cudaHostUnregister(region.start)
                if int(error) != 0:
                    errors.append(
                        f"cudaHostUnregister({region.start:#x}) failed: {_handled_error_message(self._runtime, error)}"
                    )
                    failed.append(region)
            except Exception as exc:
                errors.append(f"cudaHostUnregister({region.start:#x}) raised: {exc}")
                failed.append(region)
        self._regions = tuple(reversed(failed))
        return tuple(errors)
