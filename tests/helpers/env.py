"""Test environment / lifecycle helpers (device cleanup hooks and memory monitoring for tests).

``vllm_omni.platforms`` is imported only inside functions that need it so importing this module
at pytest plugin load does not run before session autouse fixtures.
"""

from __future__ import annotations

import gc
import os
import subprocess
import threading
import time
from contextlib import contextmanager

import torch

from vllm_omni.platforms import current_omni_platform


def get_physical_device_indices(devices):
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is None:
        return devices
    visible_indices = [int(x) for x in visible_devices.split(",")]
    index_mapping = {i: physical for i, physical in enumerate(visible_indices)}
    return [index_mapping[i] for i in devices if i in index_mapping]


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
        condition_str = f"Memory usage ratio ≤ {threshold_ratio * 100:.1f}%"

        def is_free(used, total):
            return used / total <= threshold_ratio

    print(f"[GPU Memory Monitor] Waiting for GPU {device_list} to free memory, Condition: {condition_str}")

    @contextmanager
    def smi_scope():
        if current_omni_platform.is_rocm():
            from amdsmi import amdsmi_init, amdsmi_shut_down

            amdsmi_init()
            try:
                yield
            finally:
                amdsmi_shut_down()
        elif current_omni_platform.is_cuda():
            from vllm.third_party.pynvml import nvmlInit, nvmlShutdown

            nvmlInit()
            try:
                yield
            finally:
                nvmlShutdown()
        else:
            yield

    def get_mem_gib(device: int) -> tuple[float, float]:
        if current_omni_platform.is_rocm():
            from amdsmi import amdsmi_get_gpu_vram_usage, amdsmi_get_processor_handles

            info = amdsmi_get_gpu_vram_usage(amdsmi_get_processor_handles()[device])
            return info["vram_used"] / 2**10, info["vram_total"] / 2**10
        if current_omni_platform.is_npu():
            free_bytes, total_bytes = torch.npu.mem_get_info(device)
            return (total_bytes - free_bytes) / 2**30, total_bytes / 2**30
        from vllm.third_party.pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

        info = nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(device))
        return info.used / 2**30, info.total / 2**30

    with smi_scope():
        while True:
            output_raw = {d: get_mem_gib(d) for d in devices}
            output = {
                d: f"{used:.1f}GiB/{total:.1f}GiB ({(used / total) * 100 if total > 0 else 0:.1f}%)"
                for d, (used, total) in output_raw.items()
            }

            print("[GPU Memory Status] Current usage:")
            for device_id, mem_info in output.items():
                print(f"  GPU {device_id}: {mem_info}")

            dur_s = time.time() - start_time
            if all(is_free(used, total) for used, total in output_raw.values()):
                print(f"[GPU Memory Freed] Devices {device_list} meet memory condition")
                print(f"   Condition: {condition_str}")
                print(f"   Wait time: {dur_s:.1f} seconds ({dur_s / 60:.1f} minutes)")
                break

            if dur_s >= timeout_s:
                raise ValueError(
                    f"[GPU Memory Timeout] Devices {device_list} still don't meet memory condition after {dur_s:.1f} seconds\n"
                    f"Condition: {condition_str}\n"
                    f"Current status:\n" + "\n".join(f"  GPU {d}: {output[d]}" for d in devices)
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


def _print_gpu_processes() -> None:
    """Print device information via nvidia-smi/npu-smi/amd-smi."""
    from vllm_omni.platforms import current_omni_platform

    if current_omni_platform.is_cuda():
        _run_smi("NVIDIA GPU Information (nvidia-smi)", ["nvidia-smi"], 20)
        _run_smi("Detailed GPU Processes (nvidia-smi pmon)", ["nvidia-smi", "pmon", "-c", "1"], 100, timeout=3)
    elif current_omni_platform.is_npu():
        _run_smi("Ascend NPU Information (npu-smi info)", ["npu-smi", "info"], 40)
    elif current_omni_platform.is_rocm():
        _run_smi("AMD GPU Information (amd-smi)", ["amd-smi"], 30)
        _run_smi("Detailed AMD GPU Processes (amd-smi process)", ["amd-smi", "process"], 100, timeout=3)
    else:
        print("\n" + "=" * 80)
        print("WARNING: No supported device platform detected")
        print("=" * 80)

    print("\n" + "=" * 80)
    print("System Processes with GPU keywords")
    print("=" * 80)


def run_pre_test_cleanup() -> None:
    print("Pre-test GPU status:")

    num_gpus = current_omni_platform.device_count()
    if num_gpus > 0:
        try:
            wait_for_gpu_memory_to_clear(
                devices=list(range(num_gpus)),
                threshold_ratio=0.05,
            )
        except Exception as e:
            print(f"Pre-test cleanup note: {e}")


def run_post_test_cleanup() -> None:
    if current_omni_platform.is_available():
        gc.collect()
        current_omni_platform.empty_cache()

        print("Post-test GPU status:")
        _print_gpu_processes()


class DeviceMemoryMonitor:
    """Poll global device memory usage."""

    def __init__(self, device_index: int, interval: float = 0.05):
        self.device_index = device_index
        self.interval = interval
        self._peak_used_mb = 0.0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        from vllm_omni.platforms import current_omni_platform

        def monitor_loop() -> None:
            while not self._stop_event.is_set():
                try:
                    with current_omni_platform.device(self.device_index):
                        free_bytes, total_bytes = current_omni_platform.mem_get_info()
                    used_mb = (total_bytes - free_bytes) / (1024**2)
                    self._peak_used_mb = max(self._peak_used_mb, used_mb)
                except Exception:
                    pass
                time.sleep(self.interval)

        self._thread = threading.Thread(target=monitor_loop, daemon=False)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=2.0)

    @property
    def peak_used_mb(self) -> float:
        from vllm_omni.platforms import current_omni_platform

        fallback_alloc = current_omni_platform.max_memory_allocated(device=self.device_index) / (1024**2)
        fallback_reserved = current_omni_platform.max_memory_reserved(device=self.device_index) / (1024**2)
        return max(self._peak_used_mb, fallback_alloc, fallback_reserved)

    def __del__(self):
        self.stop()


__all__ = [
    "DeviceMemoryMonitor",
    "get_physical_device_indices",
    "run_post_test_cleanup",
    "run_pre_test_cleanup",
    "wait_for_gpu_memory_to_clear",
]
