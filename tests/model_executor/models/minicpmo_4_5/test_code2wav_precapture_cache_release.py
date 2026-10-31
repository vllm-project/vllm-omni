# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression test for ``_release_precapture_cache`` on Ascend NPU.

``cfm_precapture_empty_cache`` is enabled in the shipped deploy config, but the
release path used to gate on ``torch.cuda.is_available()`` and call
``torch.accelerator.empty_cache()``. On an NPU host the gate is False, so the
whole block was dead; routing it through the CUDA branch instead crashes with
"Allocator for npu is not a DeviceAllocator". These tests pin the NPU branch to
``torch.npu.empty_cache`` so the flag keeps meaning what it says.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_omni.model_executor.models.minicpmo_4_5 import (
    minicpmo_4_5_code2wav as code2wav_module,
)
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_code2wav import (
    MiniCPMO45Code2Wav,
)
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _stage() -> MiniCPMO45Code2Wav:
    stage = MiniCPMO45Code2Wav.__new__(MiniCPMO45Code2Wav)
    return stage


def _fake_npu(monkeypatch: pytest.MonkeyPatch, calls: list[str], *, current_device: int = 0) -> None:
    npu = SimpleNamespace(
        synchronize=lambda: calls.append("synchronize"),
        empty_cache=lambda: calls.append("empty_cache"),
        current_device=lambda: current_device,
        memory_allocated=lambda device=None: 1024,
        memory_reserved=lambda device=None: 2048,
    )
    monkeypatch.setattr(code2wav_module.torch, "npu", npu, raising=False)
    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: True)
    monkeypatch.setattr(current_omni_platform, "is_cuda", lambda: False)


def test_release_precapture_cache_uses_npu_empty_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """On NPU the release calls torch.npu.empty_cache, including when CUDA was never initialized.

    The CUDA-unavailable guard used to skip the whole body. It lives in this
    test so the NPU API choice and that guard are not two copies of the same call sequence.
    """
    calls: list[str] = []
    _fake_npu(monkeypatch, calls)

    def _forbidden(*args, **kwargs):
        raise AssertionError("torch.accelerator.empty_cache crashes on NPU and must not be reached")

    monkeypatch.setattr(code2wav_module.torch.accelerator, "empty_cache", _forbidden)
    monkeypatch.setattr(code2wav_module.torch.accelerator, "synchronize", _forbidden)
    monkeypatch.setattr(code2wav_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(code2wav_module.torch.cuda, "is_initialized", lambda: False)

    _stage()._release_precapture_cache()

    assert calls == ["synchronize", "empty_cache"]


def test_release_precapture_cache_survives_a_torch_without_npu_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    """A CPU-only torch build has no "npu" device type; the release must still run.

    CI runs this suite on a host whose torch never registered the npu backend,
    where ``torch.device("npu", 0)`` raises and used to abort the release before
    ``empty_cache``. The memory delta is log-only, so it must not be able to
    suppress the release.
    """
    calls: list[str] = []
    _fake_npu(monkeypatch, calls)

    def _no_npu_device_type(*args, **kwargs):
        raise RuntimeError("Expected one of cpu, cuda, ... device type at start of device string: npu")

    monkeypatch.setattr(code2wav_module.torch, "device", _no_npu_device_type, raising=False)

    _stage()._release_precapture_cache()

    assert calls == ["synchronize", "empty_cache"]
