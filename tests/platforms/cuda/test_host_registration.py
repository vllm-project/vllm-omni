# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for the platform host-mapping registration contract."""

from __future__ import annotations

import ctypes
from collections.abc import Callable

import pytest

import vllm_omni.platforms.cuda.host_registration as host_registration
from vllm_omni.platforms.interface import HostRegistrationFailure

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

_WRITABLE_ATTRIBUTE = 99
_READ_ONLY_ATTRIBUTE = 113
_HOST_MEMORY = 1
_DEVICE_MEMORY = 2


class _FakeCudaRuntime:
    def __init__(
        self,
        *,
        writable: int = 1,
        read_only: int = 1,
        attribute_errors: dict[int, int] | None = None,
        register_result: int = 0,
        unregister_result: int = 0,
        pointer_result: int = 0,
        pointer_memory_type: int = _HOST_MEMORY,
        last_error: int = 0,
        on_register: Callable[[int, int, int], None] | None = None,
    ) -> None:
        self.attribute_values = {_WRITABLE_ATTRIBUTE: writable, _READ_ONLY_ATTRIBUTE: read_only}
        self.attribute_errors = attribute_errors or {}
        self.attribute_queries: list[tuple[int, int]] = []
        self.register_result = register_result
        self.unregister_result = unregister_result
        self.pointer_result = pointer_result
        self.pointer_memory_type = pointer_memory_type
        self.last_error = last_error
        self.registered: list[tuple[int, int, int]] = []
        self.unregistered: list[int] = []
        self.pointer_queries: list[int] = []
        self.last_error_calls = 0
        self._on_register = on_register

    def cudaDeviceGetAttribute(self, value, attribute: int, device: int) -> int:
        self.attribute_queries.append((attribute, device))
        error = self.attribute_errors.get(attribute, 0)
        if error:
            return error
        ctypes.cast(value, ctypes.POINTER(ctypes.c_int)).contents.value = self.attribute_values[attribute]
        return 0

    def cudaHostRegister(self, address: int, size: int, flags: int) -> int:
        if self._on_register is not None:
            self._on_register(address, size, flags)
        self.registered.append((address, size, flags))
        return self.register_result

    def cudaHostUnregister(self, address: int) -> int:
        self.unregistered.append(address)
        return self.unregister_result

    def cudaPointerGetAttributes(self, attributes, address: int) -> int:
        self.pointer_queries.append(address)
        if self.pointer_result:
            return self.pointer_result
        ctypes.cast(attributes, ctypes.POINTER(ctypes.c_int)).contents.value = self.pointer_memory_type
        return 0

    def cudaGetErrorString(self, error: int) -> bytes:
        return f"cuda-error-{error}".encode()

    def cudaGetLastError(self) -> int:
        self.last_error_calls += 1
        return self.last_error


@pytest.fixture(autouse=True)
def _fake_environment(monkeypatch: pytest.MonkeyPatch):
    host_registration._query_support.cache_clear()
    monkeypatch.setattr(host_registration, "_runtime", None)
    monkeypatch.setattr(host_registration, "_current_device_index", lambda: 0)
    yield
    host_registration._query_support.cache_clear()


def _use_runtime(monkeypatch: pytest.MonkeyPatch, runtime: _FakeCudaRuntime) -> None:
    monkeypatch.setattr(host_registration, "_get_runtime", lambda: runtime)


def test_support_queries_both_attributes_once(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _FakeCudaRuntime(writable=1, read_only=1)
    _use_runtime(monkeypatch, runtime)

    assert host_registration.host_registration_support() == (True, True)
    assert host_registration.host_registration_support() == (True, True)

    assert runtime.attribute_queries == [(_WRITABLE_ATTRIBUTE, 0), (_READ_ONLY_ATTRIBUTE, 0)]


def test_support_reports_partial_and_missing_support(monkeypatch: pytest.MonkeyPatch) -> None:
    writable_only = _FakeCudaRuntime(writable=1, read_only=0)
    _use_runtime(monkeypatch, writable_only)
    assert host_registration.host_registration_support() == (True, False)

    host_registration._query_support.cache_clear()
    neither = _FakeCudaRuntime(writable=0, read_only=0)
    _use_runtime(monkeypatch, neither)
    assert host_registration.host_registration_support() == (False, False)


def test_support_falls_back_when_the_runtime_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable() -> None:
        raise HostRegistrationFailure("no CUDA runtime")

    monkeypatch.setattr(host_registration, "_get_runtime", unavailable)

    assert host_registration.host_registration_support() == (False, False)


def test_support_falls_back_when_the_device_query_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _FakeCudaRuntime(attribute_errors={_WRITABLE_ATTRIBUTE: 801}, last_error=801)
    _use_runtime(monkeypatch, runtime)

    assert host_registration.host_registration_support() == (False, False)
    assert runtime.last_error_calls == 1


@pytest.mark.parametrize(("read_only", "expected_flag"), [(False, 0x00), (True, 0x08)])
def test_register_host_range_selects_the_registration_mode(
    monkeypatch: pytest.MonkeyPatch,
    read_only: bool,
    expected_flag: int,
) -> None:
    runtime = _FakeCudaRuntime()
    _use_runtime(monkeypatch, runtime)

    host_registration.register_host_range(0x1000, 8192, read_only=read_only)

    assert runtime.registered == [(0x1000, 8192, expected_flag)]


@pytest.mark.parametrize(("address", "length"), [(0, 4096), (-1, 4096), (0x1000, 0), (0x1000, -4096)])
def test_register_host_range_rejects_invalid_ranges(
    monkeypatch: pytest.MonkeyPatch,
    address: int,
    length: int,
) -> None:
    runtime = _FakeCudaRuntime()
    _use_runtime(monkeypatch, runtime)

    with pytest.raises(HostRegistrationFailure):
        host_registration.register_host_range(address, length, read_only=False)

    assert runtime.registered == []


def test_register_host_range_reports_the_driver_error(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _FakeCudaRuntime(register_result=1, last_error=1)
    _use_runtime(monkeypatch, runtime)

    with pytest.raises(HostRegistrationFailure, match="cudaHostRegister"):
        host_registration.register_host_range(0x1000, 8192, read_only=True)
    assert runtime.last_error_calls == 1


def test_register_host_range_rejects_an_unrelated_pending_error(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _FakeCudaRuntime(register_result=1, last_error=7)
    _use_runtime(monkeypatch, runtime)

    with pytest.raises(HostRegistrationFailure, match="reported 7"):
        host_registration.register_host_range(0x1000, 8192, read_only=False)


def test_unregister_host_range_forwards_the_address(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _FakeCudaRuntime()
    _use_runtime(monkeypatch, runtime)

    host_registration.unregister_host_range(0x1000)

    assert runtime.unregistered == [0x1000]


def test_unregister_host_range_reports_the_driver_error(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _FakeCudaRuntime(unregister_result=1, last_error=1)
    _use_runtime(monkeypatch, runtime)

    with pytest.raises(HostRegistrationFailure, match="cudaHostUnregister"):
        host_registration.unregister_host_range(0x1000)


@pytest.mark.parametrize(("memory_type", "expected"), [(_HOST_MEMORY, True), (_DEVICE_MEMORY, False)])
def test_host_range_is_pinned_reads_the_memory_type(
    monkeypatch: pytest.MonkeyPatch,
    memory_type: int,
    expected: bool,
) -> None:
    runtime = _FakeCudaRuntime(pointer_memory_type=memory_type)
    _use_runtime(monkeypatch, runtime)

    assert host_registration.host_range_is_pinned(0x1000) is expected
    assert runtime.pointer_queries == [0x1000]


def test_host_range_is_pinned_treats_an_unknown_range_as_unpinned(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _FakeCudaRuntime(pointer_result=1, last_error=1)
    _use_runtime(monkeypatch, runtime)

    assert host_registration.host_range_is_pinned(0x1000) is False
    assert runtime.last_error_calls == 1


def test_host_range_is_pinned_skips_invalid_addresses(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _FakeCudaRuntime()
    _use_runtime(monkeypatch, runtime)

    assert host_registration.host_range_is_pinned(0) is False
    assert runtime.pointer_queries == []


def test_default_platform_reports_no_registration_support() -> None:
    from vllm_omni.platforms.interface import UnspecifiedOmniPlatform

    assert UnspecifiedOmniPlatform.host_registration_support() == (False, False)
    assert UnspecifiedOmniPlatform.host_range_is_pinned(0x1000) is False
    with pytest.raises(NotImplementedError):
        UnspecifiedOmniPlatform.register_host_range(0x1000, 8192, read_only=True)
    with pytest.raises(NotImplementedError):
        UnspecifiedOmniPlatform.unregister_host_range(0x1000)


def test_cuda_platform_delegates_to_the_cuda_implementation(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_omni.platforms.cuda.platform import CudaOmniPlatform

    calls: list[tuple[str, tuple, dict]] = []

    def support() -> tuple[bool, bool]:
        calls.append(("support", (), {}))
        return True, False

    def register(address: int, length: int, *, read_only: bool) -> None:
        calls.append(("register", (address, length), {"read_only": read_only}))

    def unregister(address: int) -> None:
        calls.append(("unregister", (address,), {}))

    def is_pinned(address: int) -> bool:
        calls.append(("is_pinned", (address,), {}))
        return True

    monkeypatch.setattr(host_registration, "host_registration_support", support)
    monkeypatch.setattr(host_registration, "register_host_range", register)
    monkeypatch.setattr(host_registration, "unregister_host_range", unregister)
    monkeypatch.setattr(host_registration, "host_range_is_pinned", is_pinned)

    assert CudaOmniPlatform.host_registration_support() == (True, False)
    CudaOmniPlatform.register_host_range(0x1000, 8192, read_only=True)
    CudaOmniPlatform.unregister_host_range(0x1000)
    assert CudaOmniPlatform.host_range_is_pinned(0x1000) is True

    assert calls == [
        ("support", (), {}),
        ("register", (0x1000, 8192), {"read_only": True}),
        ("unregister", (0x1000,), {}),
        ("is_pinned", (0x1000,), {}),
    ]
