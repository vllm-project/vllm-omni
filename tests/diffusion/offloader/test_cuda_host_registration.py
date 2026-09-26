# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import ctypes
from types import SimpleNamespace

import pytest
import torch

import vllm_omni.diffusion.offloader.cuda_host_registration as registration_module
from vllm_omni.diffusion.offloader.cuda_host_registration import (
    CudaHostRegistration,
    _coalesce_ranges,
    _mapped_regions,
)
from vllm_omni.diffusion.offloader.host_registration import (
    HostRegistrationBudgetError,
    HostRegistrationCleanupError,
    HostRegistrationError,
    register_host_mappings,
)
from vllm_omni.host_weight_runtime import MappedHostRegion

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


class _FakeRuntime:
    def __init__(
        self,
        register_results: list[int | Exception],
        unregister_results: list[int | Exception] | None = None,
        *,
        read_only_supported: bool = True,
        attribute_result: int = 0,
    ) -> None:
        self._register_results = iter(register_results)
        self._unregister_results = iter(unregister_results or [])
        self._read_only_supported = read_only_supported
        self._attribute_result = attribute_result
        self.attribute_queries: list[tuple[int, int]] = []
        self.registered: list[tuple[int, int, int]] = []
        self.unregistered: list[int] = []

    def cudaDeviceGetAttribute(self, value, attribute: int, device: int) -> int:
        self.attribute_queries.append((attribute, device))
        ctypes.cast(value, ctypes.POINTER(ctypes.c_int)).contents.value = int(self._read_only_supported)
        return self._attribute_result

    def cudaHostRegister(self, pointer: int, size: int, flags: int) -> int:
        self.registered.append((pointer, size, flags))
        result = next(self._register_results)
        if isinstance(result, Exception):
            raise result
        return result

    def cudaHostUnregister(self, pointer: int) -> int:
        self.unregistered.append(pointer)
        result = next(self._unregister_results, 0)
        if isinstance(result, Exception):
            raise result
        return result

    @staticmethod
    def cudaGetErrorString(error: int) -> str:
        return f"error-{error}"


def _region(
    file_name: str,
    address: int,
    length: int,
    *,
    read_only: bool = True,
) -> MappedHostRegion:
    return MappedHostRegion(file_name=file_name, address=address, length=length, read_only=read_only)


@pytest.fixture(autouse=True)
def _cuda_runtime_prerequisites(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registration_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(registration_module.torch.accelerator, "current_device_index", lambda: 0)


def test_platform_dispatch_and_unsupported_device(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = object()
    regions = (_region("weights", 0x1000, 4096),)

    def create(actual_regions, *, max_bytes):
        assert actual_regions is regions
        assert max_bytes == 4096
        return sentinel

    monkeypatch.setattr(CudaHostRegistration, "create", staticmethod(create))
    assert register_host_mappings(regions, device=torch.device("cuda"), max_bytes=4096) is sentinel
    with pytest.raises(HostRegistrationError, match="not supported on cpu"):
        register_host_mappings(regions, device=torch.device("cpu"), max_bytes=None)


def test_ranges_align_per_file_without_merging_adjacent_files() -> None:
    assert _coalesce_ranges([(0x1003, 4096), (0x2800, 1024)], page_size=4096) == (
        registration_module._AddressRange(0x1000, 0x3000),
    )
    assert _mapped_regions(
        (
            _region("first.safetensors", 0x1000, 4096),
            _region("second.safetensors", 0x2000, 4096),
        )
    ) == (
        registration_module._AddressRange(0x1000, 0x2000),
        registration_module._AddressRange(0x2000, 0x3000),
    )


def test_registration_rejects_writable_or_over_budget_before_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _FakeRuntime([0])
    monkeypatch.setattr(registration_module.torch.cuda, "cudart", lambda: runtime)

    with pytest.raises(HostRegistrationError, match="is writable"):
        CudaHostRegistration.create((_region("weights", 0x1000, 1, read_only=False),), max_bytes=None)
    with pytest.raises(HostRegistrationBudgetError, match="exceeding"):
        CudaHostRegistration.create((_region("weights", 0x1003, 4096),), max_bytes=4096)
    assert runtime.registered == []


def test_registration_requires_read_only_capability(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _FakeRuntime([0], read_only_supported=False)
    monkeypatch.setattr(registration_module.torch.cuda, "cudart", lambda: runtime)

    with pytest.raises(HostRegistrationError, match="does not support read-only"):
        CudaHostRegistration.create((_region("weights", 0x1000, 4096),), max_bytes=None)
    assert runtime.attribute_queries == [
        (registration_module._CUDA_DEVICE_ATTRIBUTE_HOST_REGISTER_READ_ONLY_SUPPORTED, 0)
    ]
    assert runtime.registered == []


def test_capability_error_is_consumed_before_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _FakeRuntime([0], attribute_result=7)
    monkeypatch.setattr(registration_module.torch.cuda, "cudart", lambda: runtime)
    consumed: list[tuple[object, int]] = []
    monkeypatch.setattr(
        registration_module,
        "_consume_last_cuda_error",
        lambda actual_runtime, error: consumed.append((actual_runtime, error)),
    )

    with pytest.raises(HostRegistrationError, match="error-7"):
        CudaHostRegistration.create((_region("weights", 0x1000, 4096),), max_bytes=None)
    assert consumed == [(runtime, 7)]


def test_partial_registration_rolls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _FakeRuntime([0, 7])
    monkeypatch.setattr(registration_module.torch.cuda, "cudart", lambda: runtime)
    monkeypatch.setattr(registration_module, "_consume_last_cuda_error", lambda _runtime, _error: None)

    with pytest.raises(HostRegistrationError, match="error-7"):
        CudaHostRegistration.create(
            (
                _region("first", 0x1000, 4096),
                _region("second", 0x9000, 4096),
            ),
            max_bytes=None,
        )
    assert runtime.unregistered == [0x1000]


def test_failed_rollback_exposes_active_registration_for_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _FakeRuntime([0, 7], unregister_results=[9, 0])
    monkeypatch.setattr(registration_module.torch.cuda, "cudart", lambda: runtime)
    monkeypatch.setattr(registration_module, "_consume_last_cuda_error", lambda _runtime, _error: None)

    with pytest.raises(HostRegistrationCleanupError, match="rollback errors") as error:
        CudaHostRegistration.create(
            (
                _region("first", 0x1000, 4096),
                _region("second", 0x9000, 4096),
            ),
            max_bytes=None,
        )
    registration = error.value.active_registration
    assert registration is not None
    assert registration.close() == ()
    assert runtime.unregistered == [0x1000, 0x1000]


def test_successful_registration_retries_failed_close(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _FakeRuntime([0], unregister_results=[9, 0])
    monkeypatch.setattr(registration_module.torch.cuda, "cudart", lambda: runtime)
    monkeypatch.setattr(registration_module, "_consume_last_cuda_error", lambda _runtime, _error: None)
    registration = CudaHostRegistration.create((_region("weights", 0x1003, 1),), max_bytes=None)

    assert registration.total_bytes == 4096
    assert registration.region_count == 1
    assert registration.close() == ("cudaHostUnregister(0x1000) failed: error-9",)
    assert registration.close() == ()
    assert runtime.unregistered == [0x1000, 0x1000]


@pytest.mark.parametrize("pending_error", [0, 801])
def test_consume_last_cuda_error_accepts_cleared_or_matching_state(
    monkeypatch: pytest.MonkeyPatch,
    pending_error: int,
) -> None:
    runtime = SimpleNamespace(cudaGetLastError=lambda: pending_error)
    monkeypatch.setattr(
        registration_module.ctypes,
        "CDLL",
        lambda _name: pytest.fail("the cudart handle should provide cudaGetLastError"),
    )

    registration_module._consume_last_cuda_error(runtime, 801)


def test_consume_last_cuda_error_falls_back_to_global_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
    global_runtime = SimpleNamespace(cudaGetLastError=lambda: 801)
    monkeypatch.setattr(registration_module.ctypes, "CDLL", lambda _name: global_runtime)

    registration_module._consume_last_cuda_error(SimpleNamespace(), 801)


def test_clean_rollback_remains_recoverable_when_error_state_cannot_be_cleared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _FakeRuntime([0, 7])
    monkeypatch.setattr(runtime, "cudaGetLastError", None, raising=False)
    monkeypatch.setattr(registration_module.torch.cuda, "cudart", lambda: runtime)

    def missing_global_symbol(_name: object) -> None:
        raise OSError("cudart symbols are local")

    monkeypatch.setattr(
        registration_module.ctypes,
        "CDLL",
        missing_global_symbol,
    )

    with pytest.raises(HostRegistrationError, match="cannot clear") as error:
        CudaHostRegistration.create(
            (
                _region("first", 0x1000, 4096),
                _region("second", 0x9000, 4096),
            ),
            max_bytes=None,
        )

    assert not isinstance(error.value, HostRegistrationCleanupError)
    assert runtime.unregistered == [0x1000]
