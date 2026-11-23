# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Choose local CPUs before NPU offload host allocations, preserving CPU limits."""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from vllm.logger import init_logger

if TYPE_CHECKING:
    import torch

    from vllm_omni.diffusion.data import OmniDiffusionConfig

logger = init_logger(__name__)


class _PcieInfo(ctypes.Structure):
    # dcmi_interface_api.h: dcmi_pcie_info_all (64 bytes).
    _fields_ = [
        ("vendor", ctypes.c_uint),
        ("subvendor", ctypes.c_uint),
        ("device", ctypes.c_uint),
        ("subdevice", ctypes.c_uint),
        ("domain", ctypes.c_int),
        ("bus", ctypes.c_uint),
        ("slot", ctypes.c_uint),
        ("function", ctypes.c_uint),
        ("reserved", ctypes.c_ubyte * 32),
    ]


def _cpu_list(text: str) -> set[int]:
    result: set[int] = set()
    for part in text.strip().split(","):
        bounds = [int(value) for value in part.split("-")]
        if len(bounds) not in (1, 2) or bounds[0] < 0 or bounds[-1] < bounds[0]:
            raise ValueError("invalid CPU range")
        result.update(range(bounds[0], bounds[-1] + 1))
    return result


def _visible_physical_device(index: int) -> int | None:
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", os.environ.get("ASCEND_VISIBLE_DEVICES"))
    if visible is None:
        return None
    devices = [int(value.strip()) for value in visible.split(",")]
    if index < 0 or index >= len(devices) or any(value < 0 for value in devices) or len(set(devices)) != len(devices):
        raise ValueError("invalid NPU visibility or device index")
    return devices[index]


def _npu_affinity(index: int) -> tuple[int, int, set[int]]:
    library = ctypes.util.find_library("dcmi") or "/usr/local/Ascend/driver/lib64/driver/libdcmi.so"
    lib = ctypes.CDLL(library)
    lib.dcmi_init.restype = ctypes.c_int
    use_v2 = lib.dcmi_init() != 0
    if use_v2:
        # Ascend 950 rejects the legacy DCMI API; use its device-id API.
        initialize = getattr(lib, "dcmiv2_init", None)
        if initialize is None or initialize() != 0:
            raise RuntimeError("DCMI initialization failed")
        phy_from_logic = lib.dcmiv2_get_chip_phy_id_by_dev_id
        logic_from_phy = lib.dcmiv2_get_dev_id_by_chip_phy_id
        id_type: type[ctypes.c_int] | type[ctypes.c_uint] = ctypes.c_int
    else:
        phy_from_logic = lib.dcmi_get_device_phyid_from_logicid
        logic_from_phy = lib.dcmi_get_device_logicid_from_phyid
        id_type = ctypes.c_uint
    phy_from_logic.argtypes = [id_type, ctypes.POINTER(ctypes.c_uint)]
    logic_from_phy.argtypes = [id_type, ctypes.POINTER(ctypes.c_uint)]
    physical = _visible_physical_device(index)
    if physical is None:
        value = ctypes.c_uint()
        if phy_from_logic(index, ctypes.byref(value)) != 0:
            raise RuntimeError("cannot resolve physical NPU")
        physical = value.value
    logical = ctypes.c_uint()
    if logic_from_phy(physical, ctypes.byref(logical)) != 0:
        raise RuntimeError("cannot resolve logical NPU")
    info = _PcieInfo()
    if use_v2:
        lib.dcmiv2_get_device_pcie_info.argtypes = [ctypes.c_int, ctypes.POINTER(_PcieInfo)]
        result = lib.dcmiv2_get_device_pcie_info(logical.value, ctypes.byref(info))
    else:
        lib.dcmi_get_card_id_device_id_from_logicid.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_uint,
        ]
        lib.dcmi_get_device_pcie_info_v2.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(_PcieInfo)]
        card, device = ctypes.c_int(), ctypes.c_int()
        if lib.dcmi_get_card_id_device_id_from_logicid(ctypes.byref(card), ctypes.byref(device), logical) != 0:
            raise RuntimeError("cannot resolve NPU card and chip")
        result = lib.dcmi_get_device_pcie_info_v2(card, device, ctypes.byref(info))
    if result != 0:
        raise RuntimeError("cannot resolve NPU PCIe address")
    pci = Path("/sys/bus/pci/devices") / f"{info.domain:04x}:{info.bus:02x}:{info.slot:02x}.{info.function}"
    if int((pci / "vendor").read_text().strip(), 16) != 0x19E5:
        raise RuntimeError("PCIe address does not identify an Ascend device")
    node = int((pci / "numa_node").read_text())
    if node < 0:
        raise RuntimeError("NPU NUMA node is unavailable")
    return physical, node, _cpu_list((pci / "local_cpulist").read_text())


def _sd16_cache_path(od_config: OmniDiffusionConfig) -> bool:
    """Recognize configured MindIE-SD U8 x CP2, keeping native and SD8 defaults."""
    if not (getattr(od_config, "enable_layerwise_offload", False) and getattr(od_config, "cache_adaln", False)):
        return False
    parallel = getattr(od_config, "parallel_config", None)
    if (
        parallel is None
        or parallel.ulysses_degree != 8
        or parallel.allgather_degree != 2
        or parallel.ring_degree != 1
        or parallel.tensor_parallel_size != 1
        or parallel.world_size != 16
    ):
        return False
    attention = getattr(od_config, "diffusion_attention_config", None)
    if attention is None:
        return False
    specs = [attention.default, *attention.per_role.values()]
    return any(
        spec is not None
        and spec.backend.upper() == "RAINFUSION_ATTN"
        and spec.block_sparse is not None
        and spec.block_sparse.parallel is not None
        for spec in specs
    )


def _worker_threads() -> list[int]:
    """Enumerate only threads belonging to the current worker process."""
    return [int(path.name) for path in Path("/proc/self/task").iterdir() if path.name.isdigit()]


def _binding_request(od_config: OmniDiffusionConfig) -> bool | None:
    requested = (getattr(od_config, "additional_config", None) or {}).get("enable_cpu_binding")
    if requested is not None and not isinstance(requested, bool):
        raise ValueError("additional_config.enable_cpu_binding must be a boolean")
    return requested


@contextmanager
def npu_worker_start_affinity(od_config: OmniDiffusionConfig, device_index: int, *, is_npu: bool) -> Iterator[None]:
    """Let a fresh worker inherit local CPUs before Python imports Torch.

    Only the spawning thread is temporarily narrowed. Its incoming affinity is
    restored even if process.start() fails; sibling threads are untouched.
    """
    if not is_npu:
        yield
        return
    requested = _binding_request(od_config)
    if requested is False or (requested is None and not _sd16_cache_path(od_config)):
        yield
        return
    before: set[int] | None = None
    try:
        before = set(os.sched_getaffinity(0))
        physical, node, local_cpus = _npu_affinity(device_index)
        target = before & local_cpus
        if not target:
            raise RuntimeError("no allowed CPUs on the NPU NUMA node")
        os.sched_setaffinity(0, target)
        if set(os.sched_getaffinity(0)) != target:
            raise RuntimeError("worker start CPU binding verification failed")
    except (OSError, RuntimeError, ValueError, AttributeError) as exc:
        if before is not None:
            os.sched_setaffinity(0, before)
        if requested is True:
            raise RuntimeError(f"requested worker start CPU binding failed: {exc}") from exc
        logger.warning("Worker start CPU binding unavailable; preserving incoming affinity: %s", exc)
        yield
        return
    try:
        logger.info("Starting NPU worker: physical_device=%d numa_node=%d CPUs=%s", physical, node, sorted(target))
        yield
    finally:
        os.sched_setaffinity(0, before)


def maybe_bind_npu_offload_cpu(od_config: OmniDiffusionConfig, device: torch.device | None) -> bool:
    """Bind before weight loading; explicit additional_config overrides auto policy.

    Auto applies to cached MindIE-SD U8 x CP2 on NPU. Failure keeps the
    incoming CPU allowance; explicit True requires successful binding.
    Neither IRQ settings nor process-wide memory policy are modified.
    """
    if device is None or getattr(device, "type", None) != "npu":
        return False
    requested = _binding_request(od_config)
    if requested is False or (requested is None and not _sd16_cache_path(od_config)):
        return False
    changed: dict[int, set[int]] = {}
    try:
        if device.index is None:
            raise RuntimeError("CPU binding requires an indexed NPU device")
        before = set(os.sched_getaffinity(0))
        physical, node, local_cpus = _npu_affinity(device.index)
        target = before & local_cpus
        if not target:
            raise RuntimeError("no allowed CPUs on the NPU NUMA node")
        # The loader may run after OpenMP and backend thread pools were created.
        # Binding only its calling thread leaves those pools free to first-touch
        # weight buffers on another socket. Preserve each thread's own limits.
        for tid in _worker_threads():
            try:
                allowed = set(os.sched_getaffinity(tid))
                local = allowed & local_cpus
                if local and local != allowed:
                    os.sched_setaffinity(tid, local)
                    changed[tid] = allowed
            except ProcessLookupError:
                continue
        actual = set(os.sched_getaffinity(0))
        if not actual or not actual <= target:
            raise RuntimeError("CPU binding verification failed")
        logger.info(
            "NPU offload CPU binding: physical_device=%d numa_node=%d CPUs=%s existing_threads_bound=%d",
            physical,
            node,
            sorted(actual),
            len(changed),
        )
        return True
    except (OSError, RuntimeError, ValueError, AttributeError) as exc:
        for tid, allowed in changed.items():
            try:
                os.sched_setaffinity(tid, allowed)
            except ProcessLookupError:
                continue
        if requested is True:
            raise RuntimeError(f"requested NPU CPU binding failed: {exc}") from exc
        logger.warning("NPU offload CPU binding unavailable; preserving incoming affinity: %s", exc)
        return False
