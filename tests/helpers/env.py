"""Test environment / lifecycle helpers (GPU cleanup hooks and memory monitoring for tests).

``vllm.platforms`` / ``vllm_omni.platforms`` are imported only inside functions that need them
so importing this module at pytest plugin load does not run before session autouse fixtures
"""

from __future__ import annotations

import gc
import os
import subprocess
import threading
import time
from contextlib import contextmanager

import torch


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
    from vllm.platforms import current_platform

    assert threshold_bytes is not None or threshold_ratio is not None
    devices = get_physical_device_indices(devices)
    start_time = time.time()

    device_list = ", ".join(str(d) for d in devices)
    if threshold_bytes is not None:
        threshold_str = f"{threshold_bytes / 2**30:.2f} GiB"
        condition_str = f"Memory usage ≤ {threshold_str}"
    else:
        threshold_percent = threshold_ratio * 100
        threshold_str = f"{threshold_percent:.1f}%"
        condition_str = f"Memory usage ratio ≤ {threshold_str}"

    print(f"[GPU Memory Monitor] Waiting for GPU {device_list} to free memory, Condition: {condition_str}")

    if threshold_bytes is not None:

        def is_free(used, total):
            return used <= threshold_bytes / 2**30
    else:

        def is_free(used, total):
            return used / total <= threshold_ratio

    @contextmanager
    def nvml_scope():
        if current_platform.is_rocm():
            from amdsmi import amdsmi_init, amdsmi_shut_down

            amdsmi_init()
            try:
                yield
            finally:
                amdsmi_shut_down()
        elif current_platform.is_cuda():
            from vllm.third_party.pynvml import nvmlInit, nvmlShutdown

            nvmlInit()
            try:
                yield
            finally:
                nvmlShutdown()
        else:
            yield

    is_rocm = current_platform.is_rocm()

    with nvml_scope():
        if is_rocm:
            from amdsmi import amdsmi_get_gpu_vram_usage, amdsmi_get_processor_handles
        elif current_platform.is_cuda():
            from vllm.third_party.pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

        while True:
            output: dict[int, str] = {}
            output_raw: dict[int, tuple[float, float]] = {}
            for device in devices:
                if is_rocm:
                    dev_handle = amdsmi_get_processor_handles()[device]
                    mem_info = amdsmi_get_gpu_vram_usage(dev_handle)
                    gb_used = mem_info["vram_used"] / 2**10
                    gb_total = mem_info["vram_total"] / 2**10
                else:
                    dev_handle = nvmlDeviceGetHandleByIndex(device)
                    mem_info = nvmlDeviceGetMemoryInfo(dev_handle)
                    gb_used = mem_info.used / 2**30
                    gb_total = mem_info.total / 2**30
                output_raw[device] = (gb_used, gb_total)
                usage_percent = (gb_used / gb_total) * 100 if gb_total > 0 else 0
                output[device] = f"{gb_used:.1f}GiB/{gb_total:.1f}GiB ({usage_percent:.1f}%)"

            print("[GPU Memory Status] Current usage:")
            for device_id, mem_info in output.items():
                print(f"  GPU {device_id}: {mem_info}")

            dur_s = time.time() - start_time
            elapsed_minutes = dur_s / 60
            if all(is_free(used, total) for used, total in output_raw.values()):
                print(f"[GPU Memory Freed] Devices {device_list} meet memory condition")
                print(f"   Condition: {condition_str}")
                print(f"   Wait time: {dur_s:.1f} seconds ({elapsed_minutes:.1f} minutes)")
                break

            if dur_s >= timeout_s:
                raise ValueError(
                    f"[GPU Memory Timeout] Devices {device_list} still don't meet memory condition after {dur_s:.1f} seconds\n"
                    f"Condition: {condition_str}\n"
                    f"Current status:\n" + "\n".join(f"  GPU {device}: {output[device]}" for device in devices)
                )

            gc.collect()
            torch.accelerator.empty_cache()
            time.sleep(5)


def _print_gpu_processes() -> None:
    """Print GPU information including nvidia-smi/amd-smi and system processes."""
    from vllm.platforms import current_platform

    # Check for NVIDIA GPUs
    if current_platform.is_cuda():
        print("\n" + "=" * 80)
        print("NVIDIA GPU Information (nvidia-smi)")
        print("=" * 80)
        try:
            nvidia_result = subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if nvidia_result.returncode == 0:
                lines = nvidia_result.stdout.strip().split("\n")
                for line in lines[:20]:
                    print(line)
                if len(lines) > 20:
                    print(f"... (showing first 20 of {len(lines)} lines)")
            else:
                print("nvidia-smi command failed")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("nvidia-smi not available or timed out")
        except Exception as e:
            print(f"Error running nvidia-smi: {e}")

        print("\n" + "=" * 80)
        print("Detailed GPU Processes (nvidia-smi pmon)")
        print("=" * 80)
        try:
            pmon_result = subprocess.run(
                ["nvidia-smi", "pmon", "-c", "1"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            if pmon_result.returncode == 0 and pmon_result.stdout.strip():
                print(pmon_result.stdout)
            else:
                print("No active GPU processes found via nvidia-smi pmon")
        except Exception:
            print("nvidia-smi pmon not available")

    # Check for AMD ROCm GPUs
    elif current_platform.is_rocm():
        print("\n" + "=" * 80)
        print("AMD GPU Information (amd-smi)")
        print("=" * 80)
        try:
            amd_result = subprocess.run(
                ["amd-smi"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if amd_result.returncode == 0:
                lines = amd_result.stdout.strip().split("\n")
                for line in lines[:30]:  # Show more lines for AMD output
                    print(line)
                if len(lines) > 30:
                    print(f"... (showing first 30 of {len(lines)} lines)")
            else:
                print("amd-smi command failed")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("amd-smi not available or timed out")
        except Exception as e:
            print(f"Error running amd-smi: {e}")

        print("\n" + "=" * 80)
        print("Detailed AMD GPU Processes (amd-smi process)")
        print("=" * 80)
        try:
            process_result = subprocess.run(
                ["amd-smi", "process"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            if process_result.returncode == 0 and process_result.stdout.strip():
                print(process_result.stdout)
            else:
                print("No active GPU processes found via amd-smi process")
        except Exception:
            print("amd-smi process not available")

    else:
        print("\n" + "=" * 80)
        print("WARNING: No supported GPU platform detected (neither CUDA nor ROCm)")
        print("=" * 80)

    print("\n" + "=" * 80)
    print("System Processes with GPU keywords")
    print("=" * 80)


def run_pre_test_cleanup() -> None:
    print("Pre-test GPU status:")

    num_gpus = torch.accelerator.device_count()
    if num_gpus > 0:
        try:
            wait_for_gpu_memory_to_clear(
                devices=list(range(num_gpus)),
                threshold_ratio=0.05,
            )
        except Exception as e:
            print(f"Pre-test cleanup note: {e}")


def run_post_test_cleanup() -> None:
    if torch.cuda.is_available():
        gc.collect()
        torch.accelerator.empty_cache()

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
