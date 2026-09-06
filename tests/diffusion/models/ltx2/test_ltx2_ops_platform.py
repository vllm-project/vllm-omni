# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import pytest
from vllm.platforms.interface import DeviceCapability

from vllm_omni.diffusion.models.ltx2.ops import platform as ltx2_platform

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def clear_platform_cache():
    ltx2_platform._is_verified_cuda_device.cache_clear()
    try:
        yield
    finally:
        ltx2_platform._is_verified_cuda_device.cache_clear()


@pytest.mark.parametrize(
    ("capability", "expected"),
    [
        (90, True),
        (100, False),
        (103, False),
        (110, False),
        (120, False),
        (121, False),
    ],
)
def test_only_verified_compute_capabilities_are_eligible(
    monkeypatch: pytest.MonkeyPatch,
    capability: int,
    expected: bool,
) -> None:
    monkeypatch.setattr(ltx2_platform, "HAS_TRITON", True)
    monkeypatch.setattr(ltx2_platform.current_omni_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(ltx2_platform.current_omni_platform, "is_available", lambda: True)
    monkeypatch.setattr(
        ltx2_platform.current_omni_platform,
        "get_device_capability",
        lambda device_id=0: DeviceCapability(major=capability // 10, minor=capability % 10),
    )

    assert ltx2_platform._is_verified_cuda_device(0) is expected


def test_device_capability_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    monkeypatch.setattr(ltx2_platform, "HAS_TRITON", True)

    def get_device_capability(device_id: int = 0) -> DeviceCapability:
        nonlocal calls
        calls += 1
        return DeviceCapability(major=9, minor=0)

    monkeypatch.setattr(ltx2_platform.current_omni_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(ltx2_platform.current_omni_platform, "is_available", lambda: True)
    monkeypatch.setattr(
        ltx2_platform.current_omni_platform,
        "get_device_capability",
        get_device_capability,
    )

    assert ltx2_platform._is_verified_cuda_device(0)
    assert ltx2_platform._is_verified_cuda_device(0)
    assert calls == 1


def test_triton_is_required_for_device_eligibility(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ltx2_platform, "HAS_TRITON", False)
    monkeypatch.setattr(ltx2_platform.current_omni_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(ltx2_platform.current_omni_platform, "is_available", lambda: True)

    assert not ltx2_platform._is_verified_cuda_device(0)
