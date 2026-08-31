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


def wait_for_gpu_memory_to_clear(
    *,
    devices: list[int],
    threshold_bytes: int | None = None,
    threshold_ratio: float | None = None,
    timeout_s: float = 120,
) -> None:
    assert threshold_bytes is not None or threshold_ratio is not None
    devices = get_physical_device_indices(devices)
    start_time = time.time()

    device_list = ", ".join(str(d) for d in devices)
    if threshold_bytes is not None:
        condition_str = f"Memory usage ≤ {threshold_bytes / 2**30:.2f} GiB"

        def is_free(used, total):
            return used <= threshold_bytes / 2**30
    else:
        ratio = threshold_ratio
        assert ratio is not None
        condition_str = f"Memory usage ratio ≤ {ratio * 100:.1f}%"

        def is_free(used, total):
            return used / total <= ratio

    print(f"[Device Memory Monitor] Waiting for device(s) {device_list} to free memory, Condition: {condition_str}")

    def get_mem_gib(device: int) -> tuple[float, float]:
        with current_omni_platform.device(device):
            free_bytes, total_bytes = current_omni_platform.mem_get_info()
        return (total_bytes - free_bytes) / 2**30, total_bytes / 2**30

    while True:
        output_raw = {d: get_mem_gib(d) for d in devices}
        output = {
            d: f"{used:.1f}GiB/{total:.1f}GiB ({(used / total) * 100 if total > 0 else 0:.1f}%)"
            for d, (used, total) in output_raw.items()
        }

        print("[Device Memory Status] Current usage:")
        for device_id, mem_info in output.items():
            print(f"  Device {device_id}: {mem_info}")

        dur_s = time.time() - start_time
        if all(is_free(used, total) for used, total in output_raw.values()):
            print(f"[Device Memory Freed] Device(s) {device_list} meet memory condition")
            print(f"   Condition: {condition_str}")
            print(f"   Wait time: {dur_s:.1f} seconds ({dur_s / 60:.1f} minutes)")
            break

        if dur_s >= timeout_s:
            raise ValueError(
                f"[Device Memory Timeout] Device(s) {device_list} still don't meet memory condition after {dur_s:.1f} seconds\n"
                f"Condition: {condition_str}\n"
                f"Current status:\n" + "\n".join(f"  Device {d}: {output[d]}" for d in devices)
            )

        gc.collect()
        current_omni_platform.empty_cache()
        time.sleep(5)


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


def _cleanup_stale_device_locks() -> None:
    """Remove stale device-initialization lock files whose recorded PID is dead.

    Lock files at ``/tmp/vllm_omni_device_*_init.lock`` may persist after a
    crashed / killed test run and block subsequent orchestrator startups.
    """
    import glob as _glob

    for lock_file in _glob.glob("/tmp/vllm_omni_device_*_init.lock"):
        try:
            with open(lock_file) as fh:
                content = fh.read().strip()
            if not content:
                continue
            pid = int(content)
        except (OSError, ValueError):
            continue

        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            print(f"  Removing stale device lock {lock_file} (PID {pid} is dead)")
            try:
                os.unlink(lock_file)
            except OSError:
                pass
        except PermissionError:
            pass


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
    _cleanup_stale_device_locks()

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

    gc.collect()
    if not current_omni_platform.is_cpu():
        current_omni_platform.empty_cache()
        try:
            import torch

            torch._C._host_emptyCache()
        except AttributeError:
            logger.warning("torch._C._host_emptyCache() only available in Pytorch >=2.5")

    if current_omni_platform.is_available():
        print("Post-test device status:")
        _print_device_processes()


__all__ = [
    "cleanup_test_environment",
    "get_physical_device_indices",
    "pick_least_used_device_indices",
    "wait_for_gpu_memory_to_clear",
]
