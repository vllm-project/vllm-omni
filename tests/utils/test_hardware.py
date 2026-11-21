# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest

from vllm_omni.utils import hardware

pytestmark = [pytest.mark.cpu, pytest.mark.core_model]


def test_detect_hardware_prefers_cuda(monkeypatch):
    monkeypatch.setattr(hardware.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        hardware.torch.cuda,
        "get_device_properties",
        lambda _index: SimpleNamespace(
            name="NVIDIA RTX 4060 Laptop GPU",
            total_memory=8 * 1024**3,
            major=8,
            minor=9,
        ),
    )

    info = hardware.detect_hardware()

    assert info.backend == "cuda"
    assert info.device == "NVIDIA RTX 4060 Laptop GPU"
    assert info.memory_gb == 8.0
    assert info.detail == "CUDA capability 8.9"


def test_detect_hardware_falls_back_to_nvidia_smi(monkeypatch):
    monkeypatch.setattr(hardware.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(hardware, "_nvidia_smi", lambda: ("RTX 4090", 24.0))

    info = hardware.detect_hardware()

    assert info.backend == "cuda"
    assert info.device == "RTX 4090"
    assert info.memory_gb == 24.0
    assert "nvidia-smi only" in info.detail


def test_detect_hardware_apple_silicon(monkeypatch):
    monkeypatch.setattr(hardware.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(hardware, "_nvidia_smi", lambda: None)
    monkeypatch.setattr(hardware.sys, "platform", "darwin")
    monkeypatch.setattr(hardware.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(hardware.platform, "processor", lambda: "arm")
    monkeypatch.setattr(hardware, "_mps_available", lambda: True)
    monkeypatch.setattr(hardware, "_apple_chip", lambda: "Apple M2")
    monkeypatch.setattr(hardware, "_apple_memory_gb", lambda: 16.0)

    info = hardware.detect_hardware()

    assert info.backend == "mps"
    assert info.device == "Apple M2"
    assert info.memory_gb == 16.0
    assert info.detail == "unified memory"


def test_detect_hardware_apple_without_mps_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(hardware.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(hardware, "_nvidia_smi", lambda: None)
    monkeypatch.setattr(hardware.sys, "platform", "darwin")
    monkeypatch.setattr(hardware.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(hardware.platform, "processor", lambda: "arm")
    monkeypatch.setattr(hardware, "_mps_available", lambda: False)
    monkeypatch.setattr(hardware, "_cpu_memory_gb", lambda: 16.0)

    info = hardware.detect_hardware()

    assert info.backend == "cpu"
    assert info.memory_gb == 16.0
    assert info.detail == "no accelerator detected"


def test_detect_hardware_cpu_fallback(monkeypatch):
    monkeypatch.setattr(hardware.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(hardware, "_nvidia_smi", lambda: None)
    monkeypatch.setattr(hardware.sys, "platform", "linux")
    monkeypatch.setattr(hardware.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(hardware.platform, "processor", lambda: "x86_64")
    monkeypatch.setattr(hardware, "_cpu_memory_gb", lambda: 32.0)

    info = hardware.detect_hardware()

    assert info.backend == "cpu"
    assert info.memory_gb == 32.0
    assert info.detail == "no accelerator detected"


def test_nvidia_smi_parses_one_device_row_from_multi_gpu_output(monkeypatch):
    raw = "NVIDIA RTX 4090, 24564\nNVIDIA RTX 4090, 24564\n"
    monkeypatch.setattr(hardware.subprocess, "check_output", lambda *args, **kwargs: raw)

    name, memory_gb = hardware._nvidia_smi()

    assert name == "NVIDIA RTX 4090"
    assert memory_gb == pytest.approx(24564 / 1024)


def test_nvidia_smi_keeps_unreadable_memory_unknown(monkeypatch):
    monkeypatch.setattr(hardware.subprocess, "check_output", lambda *args, **kwargs: "NVIDIA GPU, [N/A]\n")

    name, memory_gb = hardware._nvidia_smi()

    assert name == "NVIDIA GPU"
    assert memory_gb is None


def test_detect_hardware_keeps_unknown_smi_memory_unknown(monkeypatch):
    monkeypatch.setattr(hardware.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(hardware, "_nvidia_smi", lambda: ("NVIDIA GPU", None))

    info = hardware.detect_hardware()

    assert info.backend == "cuda"
    assert info.memory_gb is None
