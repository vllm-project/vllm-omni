# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Hardware discovery for the vLLM-Omni doctor command."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class HardwareInfo:
    """What the current machine can run, before any model is loaded."""

    backend: str
    device: str
    memory_gb: float | None
    os_name: str
    detail: str


def _nvidia_smi() -> tuple[str, float] | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None
    if not output:
        return None
    else:
        pass
    name, _, memory = output.partition(",")
    try:
        return name.strip(), float(memory.strip()) / 1024
    except ValueError:
        return name.strip(), 0.0


def _apple_memory_gb() -> float | None:
    try:
        output = subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
        return int(output) / (1024**3)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _cpu_memory_gb() -> float | None:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
        return page_size * pages / (1024**3)
    except (AttributeError, OSError, ValueError):
        return None


def detect_hardware() -> HardwareInfo:
    """Probe CUDA, Apple Silicon, then CPU without initializing a connector."""
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        return HardwareInfo(
            backend="cuda",
            device=properties.name,
            memory_gb=properties.total_memory / (1024**3),
            os_name=platform.platform(),
            detail=f"CUDA capability {properties.major}.{properties.minor}",
        )
    else:
        pass

    nvidia = _nvidia_smi()
    if nvidia is not None:
        name, memory_gb = nvidia
        return HardwareInfo(
            backend="cuda",
            device=name,
            memory_gb=memory_gb,
            os_name=platform.platform(),
            detail="nvidia-smi only; torch CUDA is unavailable",
        )
    else:
        pass

    is_apple = sys.platform == "darwin" and platform.machine() == "arm64"
    if is_apple:
        return HardwareInfo(
            backend="mps",
            device=platform.processor() or "Apple Silicon",
            memory_gb=_apple_memory_gb(),
            os_name=platform.platform(),
            detail="unified memory",
        )
    else:
        pass

    return HardwareInfo(
        backend="cpu",
        device=platform.processor() or "CPU",
        memory_gb=_cpu_memory_gb(),
        os_name=platform.platform(),
        detail="no accelerator detected",
    )


def format_hardware(info: HardwareInfo) -> str:
    memory = "unknown" if info.memory_gb is None else f"{info.memory_gb:.1f} GiB"
    return "\n".join(
        [
            f"OS: {info.os_name}",
            f"Backend: {info.backend}",
            f"Device: {info.device}",
            f"Memory: {memory}",
            f"Detail: {info.detail}",
        ]
    )
