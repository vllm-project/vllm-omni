# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Read-only provisioning/failure evidence, outside measured serving windows."""

from datetime import datetime, timezone
from pathlib import Path

import psutil

_PROC_CGROUP = Path("/proc/self/cgroup")
_CGROUP_ROOT = Path("/sys/fs/cgroup")
_MEMORY_FILES = (
    "memory.max",
    "memory.current",
    "memory.peak",
    "memory.events",
    "memory.stat",
    "memory.pressure",
    "memory.swap.max",
    "memory.swap.current",
    "io.pressure",
    "cpu.max",
    "cpu.stat",
    "cpu.pressure",
    "cpu.cfs_quota_us",
    "cpu.cfs_period_us",
    "cpu.shares",
    "cpuacct.usage",
    "cpuacct.stat",
    "cpuset.cpus",
    "cpuset.effective_cpus",
    "memory.limit_in_bytes",
    "memory.usage_in_bytes",
    "memory.max_usage_in_bytes",
    "memory.failcnt",
)


def collect_resources(model_weight: Path | None = None) -> dict[str, object]:
    """Retain raw kernel paths so cgroup namespace scope stays inspectable."""
    errors: dict[str, str] = {}

    def read(path: Path) -> str | None:
        try:
            return path.read_text(encoding="utf-8", errors="replace").strip()
        except FileNotFoundError:
            return None  # Optional kernel interfaces vary by OS and cgroup version.
        except OSError as error:
            errors[str(path)] = f"{type(error).__name__}: {error}"
            return None

    membership = read(_PROC_CGROUP)
    cpu_roots = {_CGROUP_ROOT / name for name in ("cpu", "cpuacct", "cpu,cpuacct")}
    roots = {_CGROUP_ROOT, _CGROUP_ROOT / "memory", _CGROUP_ROOT / "cpuset", *cpu_roots}
    for row in (membership or "").splitlines():
        fields = row.split(":", 2)
        if len(fields) != 3:
            continue
        _, controllers, relative = fields
        path = Path(relative)
        if not path.is_absolute() or ".." in path.parts:
            continue
        if not controllers:
            roots.add(_CGROUP_ROOT / path.relative_to("/"))
        elif "memory" in controllers.split(","):
            roots.add(_CGROUP_ROOT / "memory" / path.relative_to("/"))
        if {"cpu", "cpuacct"}.intersection(controllers.split(",")):
            roots.update(root / path.relative_to("/") for root in cpu_roots)
        if "cpuset" in controllers.split(","):
            roots.add(_CGROUP_ROOT / "cpuset" / path.relative_to("/"))
    kernel = {}
    for root in sorted(roots):
        for name in _MEMORY_FILES:
            path = root / name
            if (value := read(path)) is not None:
                kernel[str(path)] = value

    host_memory = None
    try:
        memory = psutil.virtual_memory()
        host_memory = {"total_bytes": memory.total, "available_bytes": memory.available}
    except (OSError, psutil.Error) as error:
        errors["host_memory"] = f"{type(error).__name__}: {error}"

    storage = None
    if model_weight is not None:
        try:
            resolved = model_weight.resolve()
            mounts = [p for p in psutil.disk_partitions(all=True) if resolved.is_relative_to(Path(p.mountpoint))]
            mount = max(mounts, key=lambda p: len(Path(p.mountpoint).parts), default=None)
            storage = {
                "weight_path": str(resolved),
                "filesystem": mount.fstype if mount else None,
                "mountpoint": mount.mountpoint if mount else None,
            }
        except (OSError, psutil.Error) as error:
            errors["model_storage"] = f"{type(error).__name__}: {error}"
    return {
        "observed_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "provisioning/failure diagnostics; excluded from performance measurements",
        "host_memory": host_memory,
        "cgroup_membership": membership,
        "kernel_memory": kernel,
        "model_storage": storage,
        "errors": errors,
    }
