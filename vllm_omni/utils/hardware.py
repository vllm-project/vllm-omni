# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Local hardware discovery for consumer-device deployments.

The probe answers one question: how much device memory does this machine
actually expose? It deliberately avoids importing torch, because the CLI needs
an answer before any model stack is loaded, and it never raises: a probe that
fails should cost one line of output, not the command.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass

from vllm.logger import init_logger

logger = init_logger(__name__)

MIB_PER_GIB = 1024
_PROBE_TIMEOUT_S = 10.0


@dataclass(frozen=True)
class Device:
    """One accelerator or CPU visible to this process."""

    kind: str
    """``cuda``, ``rocm``, ``mps``, ``npu``, ``xpu``, or ``cpu``."""

    name: str
    total_mib: int | None
    """Memory the device exposes, in MiB; ``None`` when it cannot be read."""

    unified_memory: bool = False
    """Apple Silicon shares one memory pool between CPU and GPU."""

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "name": self.name,
            "total_mib": self.total_mib,
            "unified_memory": self.unified_memory,
        }

    @property
    def label(self) -> str:
        if self.total_mib is None:
            return f"{self.name} ({self.kind}, memory unknown)"
        memory = f"{self.total_mib / MIB_PER_GIB:.1f} GiB"
        if self.kind == "cpu":
            return f"{self.name} ({self.kind}, {memory} system memory)"
        pool = "unified" if self.unified_memory else "dedicated"
        return f"{self.name} ({self.kind}, {memory} {pool})"


def _run(*command: str) -> str | None:
    if shutil.which(command[0]) is None:
        return None
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=True,
            timeout=_PROBE_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("%s failed: %s", " ".join(command), exc)
        return None
    if completed.returncode != 0:
        logger.debug("%s exited %s", " ".join(command), completed.returncode)
        return None
    return completed.stdout


def _parse_mib(value: str) -> int | None:
    try:
        return int(float(value.strip()))
    except (TypeError, ValueError):
        return None


def _from_nvidia_smi() -> list[Device]:
    """Read NVIDIA GPUs without importing torch; also covers `nvidia-smi` on WSL2."""
    output = _run("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits")
    devices: list[Device] = []
    for line in (output or "").splitlines():
        name, _, memory = line.rpartition(",")
        if not name.strip():
            continue
        devices.append(Device(kind="cuda", name=name.strip(), total_mib=_parse_mib(memory)))
    return devices


def _apple_memory_mib() -> int | None:
    """Total unified memory on Apple Silicon, reported in bytes by sysctl."""
    output = _run("sysctl", "-n", "hw.memsize")
    if not output:
        return None
    try:
        return int(output.strip()) // (1024 * 1024)
    except ValueError:
        return None


def _from_torch() -> list[Device]:
    """Fall back to torch for ROCm, XPU, NPU, and MPS, which have no common CLI."""
    try:
        import torch
    except Exception as exc:  # torch is optional for the probe
        logger.debug("torch unavailable: %s", exc)
        return []

    if torch.cuda.is_available():
        kind = "rocm" if getattr(torch.version, "hip", None) else "cuda"
        devices = []
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                Device(kind=kind, name=properties.name, total_mib=properties.total_memory // (1024 * 1024))
            )
        return devices

    for kind, probe in (
        ("npu", lambda: hasattr(torch, "npu") and torch.npu.is_available() and torch.npu.device_count()),
        ("xpu", lambda: hasattr(torch, "xpu") and torch.xpu.is_available() and torch.xpu.device_count()),
    ):
        try:
            count = int(probe() or 0)
        except Exception as exc:
            logger.debug("%s probe failed: %s", kind, exc)
            continue
        if count:
            return [Device(kind=kind, name=f"{kind}{index}", total_mib=None) for index in range(count)]

    try:
        if torch.backends.mps.is_available():
            return [
                Device(
                    kind="mps",
                    name=f"{platform.machine()} Apple Silicon",
                    total_mib=_apple_memory_mib(),
                    unified_memory=True,
                )
            ]
    except Exception as exc:
        logger.debug("mps probe failed: %s", exc)
    return []


def _cpu_memory_mib() -> int | None:
    try:
        return (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) // (1024 * 1024)
    except (AttributeError, OSError, ValueError):
        return None


def detect_devices() -> list[Device]:
    """Every accelerator this machine exposes, or the CPU when there is none."""
    devices = _from_nvidia_smi() or _from_torch()
    if devices:
        return devices

    memory_mib = _apple_memory_mib() if platform.system() == "Darwin" else _cpu_memory_mib()
    cpu_name = platform.processor() or platform.machine() or "cpu"
    return [Device(kind="cpu", name=f"CPU ({cpu_name})", total_mib=memory_mib)]


def detect_devices_json() -> str:
    return json.dumps([device.as_dict() for device in detect_devices()], indent=2)
