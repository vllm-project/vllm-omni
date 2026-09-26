# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Test cleanup helpers (device memory, distributed state, and memory monitoring).

``vllm_omni.platforms`` is imported only inside functions that need it so importing this module
at pytest plugin load does not run before session autouse fixtures.
"""

from __future__ import annotations

import gc
import logging
import os
import subprocess
import time
from collections.abc import Iterable

import psutil

from vllm_omni.platforms import current_omni_platform

logger = logging.getLogger(__name__)


def get_physical_device_indices(devices):
    """Map logical device indices to physical IDs via the platform visibility env.

    Uses ``current_omni_platform.device_control_env_var`` (e.g. ``CUDA_VISIBLE_DEVICES``,
    ``ASCEND_RT_VISIBLE_DEVICES``) rather than hard-coding CUDA.
    """
    visible_devices = os.environ.get(current_omni_platform.device_control_env_var)
    if visible_devices is None:
        return devices
    visible_indices = [int(x) for x in visible_devices.split(",") if x.strip() != ""]
    index_mapping = {i: physical for i, physical in enumerate(visible_indices)}
    return [index_mapping[i] for i in devices if i in index_mapping]


def pick_least_used_device_indices(num_devices: int) -> list[int]:
    """Return physical device indices for the least-used accelerators.

    Args:
        num_devices: Number of device indices to return.

    Returns:
        Physical device indices sorted from least to most used memory.
    """
    if num_devices <= 0:
        raise ValueError(f"num_devices must be positive, got {num_devices}.")

    logical_count = current_omni_platform.get_device_count()
    if logical_count < num_devices:
        raise RuntimeError(f"Need {num_devices} devices, but only {logical_count} are available.")

    device_usage: list[tuple[int, int]] = []
    for device_index in range(logical_count):
        with current_omni_platform.device(device_index):
            free_bytes, total_bytes = current_omni_platform.get_device_memory()
        device_usage.append((total_bytes - free_bytes, device_index))
    device_usage.sort()
    logical_indices = [device_index for _, device_index in device_usage[:num_devices]]
    return get_physical_device_indices(logical_indices)


def _logical_to_physical(logical: int) -> int:
    """Physical id for logs / nvidia-smi; ``logical`` if the env does not map it."""
    mapped = get_physical_device_indices([logical])
    return mapped[0] if mapped else logical


def wait_for_gpu_memory_to_clear(
    *,
    devices: list[int],
    threshold_bytes: int | None = None,
    threshold_ratio: float | None = None,
    timeout_s: float = 120,
) -> None:
    """Wait until *logical* device ordinals are under the memory threshold.

    ``devices`` are CUDA/NPU ordinals in the current process (``0..N-1``),
    the same space as ``current_omni_platform.device()`` / ``mem_get_info()``.
    Physical ids (``CUDA_VISIBLE_DEVICES``) are only used for log labels and
    Whisper-worker matching — remapping them into ``device()`` raises
    ``invalid device ordinal`` when the visible set is a subset (e.g. ``4,5``).
    """
    assert threshold_bytes is not None or threshold_ratio is not None
    logical_devices = list(devices)
    physical_by_logical = {logical: _logical_to_physical(logical) for logical in logical_devices}
    start_time = time.time()

    device_list = ", ".join(str(physical_by_logical[d]) for d in logical_devices)
    if threshold_bytes is not None:
        condition_str = f"Memory usage ≤ {threshold_bytes / 2**30:.2f} GiB (excl. Whisper worker)"

        def is_free(used, total):
            return used <= threshold_bytes / 2**30
    else:
        ratio = threshold_ratio
        assert ratio is not None
        condition_str = f"Memory usage ratio ≤ {ratio * 100:.1f}% (excl. Whisper worker)"

        def is_free(used, total):
            return used / total <= ratio

    print(f"[Device Memory Monitor] Waiting for device(s) {device_list} to free memory, Condition: {condition_str}")
    whisper_allowance_gib, whisper_physical = _whisper_vram_allowance()
    if whisper_allowance_gib > 0 and whisper_physical is not None:
        print(
            f"[Device Memory Monitor] Whisper worker allowance {whisper_allowance_gib:.1f}GiB "
            f"on device {whisper_physical}"
        )

    while True:
        output_raw: dict[int, tuple[float, float, float]] = {}
        for logical in logical_devices:
            with current_omni_platform.device(logical):
                free_bytes, total_bytes = current_omni_platform.mem_get_info()
            used_gib = (total_bytes - free_bytes) / 2**30
            total_gib = total_bytes / 2**30
            physical = physical_by_logical[logical]
            whisper_gib = (
                min(used_gib, whisper_allowance_gib)
                if whisper_physical is not None and physical == whisper_physical
                else 0.0
            )
            output_raw[physical] = (max(0.0, used_gib - whisper_gib), total_gib, whisper_gib)
        print("[Device Memory Status] Current usage (engine view excludes Whisper worker):")
        for device_id, (used, total, whisper) in output_raw.items():
            ratio_pct = (used / total) * 100 if total > 0 else 0.0
            line = f"  Device {device_id}: {used:.1f}GiB/{total:.1f}GiB ({ratio_pct:.1f}%)"
            if whisper > 0:
                line += f" [Whisper worker {whisper:.1f}GiB excluded]"
            print(line)

        dur_s = time.time() - start_time
        if all(is_free(used, total) for used, total, _whisper in output_raw.values()):
            print(f"[Device Memory Freed] Device(s) {device_list} meet memory condition")
            print(f"   Condition: {condition_str}")
            print(f"   Wait time: {dur_s:.1f} seconds ({dur_s / 60:.1f} minutes)")
            break

        if dur_s >= timeout_s:
            status = "\n".join(
                f"  Device {d}: {used:.1f}GiB/{total:.1f}GiB"
                + (f" (+Whisper {whisper:.1f}GiB excluded)" if whisper > 0 else "")
                for d, (used, total, whisper) in output_raw.items()
            )
            raise ValueError(
                f"[Device Memory Timeout] Device(s) {device_list} still don't meet memory condition after {dur_s:.1f} seconds\n"
                f"Condition: {condition_str}\n"
                f"Current status:\n{status}"
            )

        gc.collect()
        current_omni_platform.empty_cache()
        time.sleep(5)


def _whisper_vram_allowance() -> tuple[float, int | None]:
    """Living Whisper worker's GPU footprint and physical device.

    Orthogonal to RFC #6851 leftover-engine PID reap / a raised wait ratio:
    this only excludes the transcriber's own allocator (capped by current used
    on that device). CPU / unknown device → ``(0.0, None)``.
    """
    try:
        from tests.helpers.media import whisper_resident_device_index, whisper_resident_vram_gib
    except Exception:
        return 0.0, None
    gib = whisper_resident_vram_gib()
    logical = whisper_resident_device_index()
    if gib <= 0.0 or logical is None:
        return 0.0, None
    mapped = get_physical_device_indices([logical])
    if not mapped:
        return 0.0, None
    return gib, mapped[0]


def _run_smi(label: str, cmd: list[str], head_lines: int, timeout: float = 5) -> None:
    print("\n" + "=" * 80)
    print(label)
    print("=" * 80)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split("\n")
            for line in lines[:head_lines]:
                print(line)
            if len(lines) > head_lines:
                print(f"... (showing first {head_lines} of {len(lines)} lines)")
        else:
            print(f"{cmd[0]} command failed or produced no output")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print(f"{cmd[0]} not available or timed out")
    except Exception as e:
        print(f"Error running {' '.join(cmd)}: {e}")


def _print_device_processes() -> None:
    """Print device information via platform SMI tools."""
    if current_omni_platform.is_cuda():
        _run_smi("NVIDIA GPU Information (nvidia-smi)", ["nvidia-smi"], 20)
        _run_smi("Detailed GPU Processes (nvidia-smi pmon)", ["nvidia-smi", "pmon", "-c", "1"], 100, timeout=3)
    elif current_omni_platform.is_npu():
        _run_smi("Ascend NPU Information (npu-smi info)", ["npu-smi", "info"], 40)
    elif current_omni_platform.is_rocm():
        # amd-smi can enter uninterruptible sleep in the amdgpu CPER ioctl
        # (amdgpu_cper_ring_write). In that state subprocess.run's timeout
        # cannot reap it, causing this diagnostic hook to hang the entire test.
        # rocm-smi provides the information needed here without that ioctl.
        _run_smi(
            "AMD GPU Information (rocm-smi)",
            ["rocm-smi", "--showuse", "--showmeminfo", "vram"],
            60,
        )
        _run_smi("Detailed AMD GPU Processes (rocm-smi)", ["rocm-smi", "--showpids"], 100, timeout=3)
    elif current_omni_platform.is_xpu():
        _run_smi("Intel XPU Information (xpu-smi discovery)", ["xpu-smi", "discovery"], 40)
    elif current_omni_platform.is_musa():
        _run_smi("Moore Threads GPU Information (mthreads-gmi)", ["mthreads-gmi"], 30)
    else:
        print("\n" + "=" * 80)
        print("WARNING: No supported device platform detected")
        print("=" * 80)


def cleanup_test_environment(*, shutdown_ray: bool = False) -> None:
    """Tear down distributed state and reset device memory for tests."""
    from vllm import envs
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
    )

    envs.disable_envs_cache()

    if current_omni_platform.is_rocm():
        from vllm._aiter_ops import rocm_aiter_ops

        rocm_aiter_ops.refresh_env_variables()

    gc.unfreeze()
    destroy_model_parallel()
    destroy_distributed_environment()
    if shutdown_ray:
        import ray

        ray.shutdown()

    print("Pre-test device status:")
    gc.collect()
    if not current_omni_platform.is_cpu():
        current_omni_platform.empty_cache()
        try:
            import torch

            torch._C._host_emptyCache()
        except AttributeError:
            logger.warning("torch._C._host_emptyCache() only available in Pytorch >=2.5")

    num_devices = current_omni_platform.device_count()
    if num_devices > 0:
        try:
            wait_for_gpu_memory_to_clear(
                devices=list(range(num_devices)),
                threshold_ratio=0.05,
                timeout_s=60,
            )
        except Exception as e:
            print(f"Device cleanup note: {e}")

    if current_omni_platform.is_available():
        print("Post-test device status:")
        _print_device_processes()


# Lower-cased substrings that identify engine worker processes spawned under the
# test process: vLLM ``VLLM::EngineCore`` / ``VLLM::Worker_*`` titles, vLLM-Omni
# diffusion workers (``vLLM-Omni::DiffusionWorker_*``) and the
# ``StageDiffusionProc`` subprocess. Matched against the psutil name and cmdline
# (``setproctitle`` rewrites both).
ENGINE_WORKER_PROCESS_MARKERS: tuple[str, ...] = (
    "enginecore",
    "stagediffusionproc",
    "vllm::worker",
    "vllm-omni::",
)


def is_engine_worker_process(proc: psutil.Process) -> bool:
    """True when *proc* looks like an engine worker (see ``ENGINE_WORKER_PROCESS_MARKERS``)."""
    try:
        blob = " ".join([proc.name(), *proc.cmdline()]).lower()
    except psutil.Error:
        return False
    return any(marker in blob for marker in ENGINE_WORKER_PROCESS_MARKERS)


def snapshot_engine_worker_pids() -> list[int]:
    """PIDs of the engine workers currently running under this process.

    Take the snapshot while the engine is alive, **before** ``shutdown()``:
    diffusion workers are daemon children of ``StageDiffusionProc``, so once
    that subprocess is joined they are reparented and a later
    ``children(recursive=True)`` scan no longer sees them although they can
    still hold device memory.
    """
    try:
        children = psutil.Process().children(recursive=True)
    except psutil.Error:
        return []
    return [proc.pid for proc in children if is_engine_worker_process(proc)]


def reap_engine_worker_pids(pids: Iterable[int], *, timeout_s: float = 3.0) -> list[int]:
    """Kill the processes in *pids* that are still alive; return the PIDs that were killed.

    ``shutdown()`` already attempted a graceful exit, so this goes straight to
    ``SIGKILL``: a ``SIGTERM`` grace period only delays the kill for CUDA
    workers that ignore it.
    """
    procs: list[psutil.Process] = []
    for pid in pids:
        try:
            procs.append(psutil.Process(pid))
        except psutil.NoSuchProcess:
            continue
    if not procs:
        return []
    for proc in procs:
        try:
            proc.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(procs, timeout=timeout_s)
    return [proc.pid for proc in procs]


__all__ = [
    "ENGINE_WORKER_PROCESS_MARKERS",
    "cleanup_test_environment",
    "get_physical_device_indices",
    "is_engine_worker_process",
    "pick_least_used_device_indices",
    "reap_engine_worker_pids",
    "snapshot_engine_worker_pids",
    "wait_for_gpu_memory_to_clear",
]
