# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA host-mapping registration for the unified weight stack.

The unified Offloader registers existing host ranges with the driver instead
of copying weights into pinned buffers. This module owns the CUDA side of that
contract: capability queries, ``cudaHostRegister``/``cudaHostUnregister``, and
pinned-range verification. It deliberately avoids
``vllm_omni.diffusion.offloader``, whose registration path belongs to the
legacy layerwise backend and keeps its current behavior.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import functools
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol, cast

import torch
from vllm.logger import init_logger

from vllm_omni.platforms.interface import HostRegistrationFailure

logger = init_logger(__name__)

# cudaDeviceAttr values from cuda_runtime_api.h.
_CUDA_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED = 99
_CUDA_DEVICE_ATTRIBUTE_HOST_REGISTER_READ_ONLY_SUPPORTED = 113

# cudaHostRegister flags: the default path covers private storage and rank
# shards, and the read-only path covers sealed shared mappings.
_CUDA_HOST_REGISTER_DEFAULT = 0x00
_CUDA_HOST_REGISTER_READ_ONLY = 0x08

# cudaMemoryType: only a host mapping is pinned for direct H2D copies.
_CUDA_MEMORY_TYPE_HOST = 1

_CUDA_SUCCESS = 0

# The API writes sizeof(cudaPointerAttributes) bytes; the buffer is a superset
# so its layout does not need to track CUDA releases. Only `type` is read.
_CUDA_POINTER_ATTRIBUTES_BYTES = 64


class _CudaRuntime(Protocol):
    def cudaDeviceGetAttribute(self, value: object, attribute: int, device: int) -> int: ...

    def cudaHostRegister(self, address: int, size: int, flags: int) -> int: ...

    def cudaHostUnregister(self, address: int) -> int: ...

    def cudaPointerGetAttributes(self, attributes: ctypes.Array, address: int) -> int: ...

    def cudaGetErrorString(self, error: int) -> bytes: ...

    def cudaGetLastError(self) -> int: ...


_runtime: _CudaRuntime | None = None


def _runtime_candidates() -> Iterator[str]:
    """Yield library names to try, preferring the runtime PyTorch already uses.

    A wheel install ships the runtime under ``torch/lib``; a source build
    installs it system-wide. Registering memory through the runtime instance
    that PyTorch linked against avoids loading a second CUDA runtime version
    into the process, and a re-open of an already loaded soname is cheap.
    """
    torch_lib = Path(torch.__file__).resolve().parent / "lib"
    for path in sorted(torch_lib.glob("libcudart.so*")):
        yield str(path)
    yield "libcudart.so"
    discovered = ctypes.util.find_library("cudart")
    if discovered:
        yield discovered


def _open_cuda_runtime() -> ctypes.CDLL:
    errors: list[str] = []
    for candidate in _runtime_candidates():
        try:
            return ctypes.CDLL(candidate)
        except OSError as exc:
            errors.append(f"{candidate}: {exc}")
    try:
        # The runtime may already be loaded into the process with global
        # visibility, where RTLD_DEFAULT can still resolve its symbols.
        return ctypes.CDLL(None)
    except OSError as exc:
        errors.append(f"process image: {exc}")
    raise HostRegistrationFailure("cannot load the CUDA runtime: " + "; ".join(errors))


def _load_runtime() -> _CudaRuntime:
    library = _open_cuda_runtime()
    try:
        library.cudaDeviceGetAttribute.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
        ]
        library.cudaDeviceGetAttribute.restype = ctypes.c_int
        library.cudaHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
        library.cudaHostRegister.restype = ctypes.c_int
        library.cudaHostUnregister.argtypes = [ctypes.c_void_p]
        library.cudaHostUnregister.restype = ctypes.c_int
        library.cudaPointerGetAttributes.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        library.cudaPointerGetAttributes.restype = ctypes.c_int
        library.cudaGetErrorString.argtypes = [ctypes.c_int]
        library.cudaGetErrorString.restype = ctypes.c_char_p
        library.cudaGetLastError.argtypes = []
        library.cudaGetLastError.restype = ctypes.c_int
    except AttributeError as exc:
        raise HostRegistrationFailure(f"CUDA runtime does not expose host registration: {exc}") from exc
    return cast(_CudaRuntime, library)


def _get_runtime() -> _CudaRuntime:
    global _runtime
    if _runtime is None:
        _runtime = _load_runtime()
    return _runtime


def _error_string(runtime: _CudaRuntime, error: int) -> str:
    try:
        message = runtime.cudaGetErrorString(error)
    except (OSError, RuntimeError, TypeError, ValueError):
        return f"CUDA error {error}"
    return message.decode(errors="replace") if isinstance(message, bytes) else str(message)


def _clear_last_error(runtime: _CudaRuntime, expected_error: int) -> None:
    """Clear one handled CUDA error and reject a different pending one."""
    try:
        pending_error = int(runtime.cudaGetLastError())
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise HostRegistrationFailure("cannot clear CUDA's pending error after host-registration failure") from exc
    if pending_error not in (_CUDA_SUCCESS, expected_error):
        raise HostRegistrationFailure(
            f"host registration returned CUDA error {expected_error}, but cudaGetLastError reported {pending_error}"
        )


def _fail(runtime: _CudaRuntime, action: str, error: int) -> HostRegistrationFailure:
    message = _error_string(runtime, error)
    try:
        _clear_last_error(runtime, error)
    except HostRegistrationFailure as exc:
        message = f"{message}; {exc}"
    return HostRegistrationFailure(f"{action} failed: {message}")


def _current_device_index() -> int:
    return int(torch.accelerator.current_device_index())


def _device_attribute(runtime: _CudaRuntime, attribute: int, device_index: int) -> int:
    value = ctypes.c_int()
    error = int(runtime.cudaDeviceGetAttribute(ctypes.pointer(value), attribute, device_index))
    if error != _CUDA_SUCCESS:
        failure = _fail(runtime, f"cudaDeviceGetAttribute({attribute})", error)
        raise failure
    return int(value.value)


@functools.cache
def _query_support(device_index: int) -> tuple[bool, bool]:
    runtime = _get_runtime()
    writable = _device_attribute(runtime, _CUDA_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED, device_index)
    read_only = _device_attribute(runtime, _CUDA_DEVICE_ATTRIBUTE_HOST_REGISTER_READ_ONLY_SUPPORTED, device_index)
    return bool(writable), bool(read_only)


def host_registration_support() -> tuple[bool, bool]:
    """Return ``(writable, read_only)`` registration support on this device.

    A device that cannot be queried reports no support so that callers fall
    back to staging instead of failing startup.
    """
    try:
        return _query_support(_current_device_index())
    except (HostRegistrationFailure, RuntimeError) as exc:
        logger.debug("CUDA host registration is unavailable: %s", exc)
        return False, False


def register_host_range(address: int, length: int, *, read_only: bool) -> None:
    """Register one page-aligned host range with the CUDA driver."""
    if address <= 0:
        raise HostRegistrationFailure(f"host range address must be positive, got {address}")
    if length <= 0:
        raise HostRegistrationFailure(f"host range length must be positive, got {length}")

    runtime = _get_runtime()
    flags = _CUDA_HOST_REGISTER_READ_ONLY if read_only else _CUDA_HOST_REGISTER_DEFAULT
    error = int(runtime.cudaHostRegister(address, length, flags))
    if error != _CUDA_SUCCESS:
        raise _fail(
            runtime,
            f"cudaHostRegister(address={address:#x}, length={length}, read_only={read_only})",
            error,
        )


def unregister_host_range(address: int) -> None:
    """Release a range previously registered by :func:`register_host_range`."""
    if address <= 0:
        raise HostRegistrationFailure(f"host range address must be positive, got {address}")

    runtime = _get_runtime()
    error = int(runtime.cudaHostUnregister(address))
    if error != _CUDA_SUCCESS:
        raise _fail(runtime, f"cudaHostUnregister(address={address:#x})", error)


def host_range_is_pinned(address: int) -> bool:
    """Whether the driver reports the host range at ``address`` as pinned.

    An unverifiable range reports ``False`` so the caller treats registration
    as failed and routes the unit through staging.
    """
    if address <= 0:
        return False
    try:
        runtime = _get_runtime()
    except HostRegistrationFailure as exc:
        logger.debug("Cannot verify pinned host memory: %s", exc)
        return False

    attributes = ctypes.create_string_buffer(_CUDA_POINTER_ATTRIBUTES_BYTES)
    error = int(runtime.cudaPointerGetAttributes(attributes, address))
    if error != _CUDA_SUCCESS:
        try:
            _clear_last_error(runtime, error)
        except HostRegistrationFailure as exc:
            logger.debug("Cannot clear CUDA's pending error after a pointer query: %s", exc)
        return False
    memory_type = ctypes.cast(attributes, ctypes.POINTER(ctypes.c_int)).contents.value
    return memory_type == _CUDA_MEMORY_TYPE_HOST


__all__ = [
    "host_range_is_pinned",
    "host_registration_support",
    "register_host_range",
    "unregister_host_range",
]
