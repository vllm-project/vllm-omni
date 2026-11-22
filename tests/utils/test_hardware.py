# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Local hardware discovery used by ``vllm-omni doctor``."""

import pytest

from vllm_omni.utils import hardware

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_nvidia_smi_rows_become_cuda_devices(monkeypatch):
    monkeypatch.setattr(
        hardware,
        "_run",
        lambda *command: "NVIDIA GeForce RTX 5080 Laptop GPU, 16303\nNVIDIA GeForce RTX 4090, 24564\n",
    )

    devices = hardware._from_nvidia_smi()

    assert [device.total_mib for device in devices] == [16303, 24564]
    assert {device.kind for device in devices} == {"cuda"}
    assert devices[0].label == "NVIDIA GeForce RTX 5080 Laptop GPU (cuda, 15.9 GiB dedicated)"


def test_unreadable_fractional_memory_does_not_crash(monkeypatch):
    monkeypatch.setattr(hardware, "_run", lambda *command: "Some GPU, [N/A]\n")

    devices = hardware._from_nvidia_smi()

    assert devices == [hardware.Device(kind="cuda", name="Some GPU", total_mib=None)]
    assert "memory unknown" in devices[0].label


def test_detect_devices_prefers_the_cli_over_importing_torch(monkeypatch):
    expected = [hardware.Device(kind="cuda", name="fake", total_mib=8192)]
    monkeypatch.setattr(hardware, "_from_nvidia_smi", lambda: expected)
    monkeypatch.setattr(hardware, "_from_torch", lambda: pytest.fail("torch must not be imported"))
    monkeypatch.setattr(hardware, "_cpu_memory_mib", lambda: pytest.fail("cpu fallback unused"))

    assert hardware.detect_devices() == expected


def test_detect_devices_falls_back_to_torch_then_cpu(monkeypatch):
    apple = [hardware.Device(kind="mps", name="arm64 Apple Silicon", total_mib=16384, unified_memory=True)]
    monkeypatch.setattr(hardware, "_from_nvidia_smi", lambda: [])
    monkeypatch.setattr(hardware, "_from_torch", lambda: apple)
    assert hardware.detect_devices() == apple

    monkeypatch.setattr(hardware, "_from_torch", lambda: [])
    monkeypatch.setattr(hardware, "_cpu_memory_mib", lambda: 16384)
    (device,) = hardware.detect_devices()
    assert (device.kind, device.total_mib) == ("cpu", 16384)


def test_cpu_device_reports_system_memory(monkeypatch):
    monkeypatch.setattr(hardware, "_from_nvidia_smi", lambda: [])
    monkeypatch.setattr(hardware, "_from_torch", lambda: [])
    monkeypatch.setattr(hardware.platform, "system", lambda: "Linux")
    monkeypatch.setattr(hardware, "_cpu_memory_mib", lambda: 16 * hardware.MIB_PER_GIB)

    (device,) = hardware.detect_devices()

    assert device.kind == "cpu"
    assert not device.unified_memory
    assert device.label.endswith("(cpu, 16.0 GiB system memory)")


def test_probe_failure_is_not_fatal(monkeypatch):
    def boom(*command, **kwargs):
        raise OSError("no such file")

    monkeypatch.setattr(hardware.subprocess, "run", boom)
    monkeypatch.setattr(hardware.shutil, "which", lambda _: "/usr/bin/nvidia-smi")

    assert hardware._run("nvidia-smi") is None
