"""NVML-based per-process GPU memory utilities.

Shared across worker types (OmniGPUWorkerBase, DiffusionWorker, etc.)
for process-scoped GPU memory accounting.
"""

from __future__ import annotations

import os

from vllm.logger import init_logger
from vllm.third_party.pynvml import (
    nvmlDeviceGetComputeRunningProcesses,
    nvmlDeviceGetHandleByIndex,
    nvmlInit,
    nvmlShutdown,
)

logger = init_logger(__name__)


def is_process_scoped_memory_available() -> bool:
    """Check if NVML process-scoped memory tracking is available.

    When True, concurrent stage initialization is safe because each
    process can accurately measure its own GPU memory via NVML.
    When False, sequential initialization (file locks) is needed.
    """
    try:
        nvmlInit()
        nvmlShutdown()
        return True
    except Exception:
        return False


def parse_cuda_visible_devices() -> list[str | int]:
    """Parse CUDA_VISIBLE_DEVICES into a list of device identifiers.

    Returns list of integers (physical indices) or strings (UUIDs/MIG IDs).
    """
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible_devices:
        return []

    result: list[str | int] = []
    for item in visible_devices.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            result.append(int(item))
        except ValueError:
            # UUID (GPU-xxx) or MIG ID (MIG-xxx)
            result.append(item)
    return result


def get_device_handle(device_id: str | int):
    """Get NVML device handle by index or UUID."""
    if isinstance(device_id, int):
        return nvmlDeviceGetHandleByIndex(device_id)
    else:
        from vllm.third_party.pynvml import nvmlDeviceGetHandleByUUID

        return nvmlDeviceGetHandleByUUID(device_id)


def get_process_gpu_memory(local_rank: int) -> int | None:
    """Get GPU memory used by current process via pynvml.

    Supports CUDA_VISIBLE_DEVICES with integer indices, UUIDs, or MIG IDs.

    Returns:
        Memory in bytes used by this process, or None when NVML cannot answer.

        None covers two cases: NVML is unavailable, and this process's PID is
        absent from the device's compute-process list. The miss is reported as
        None rather than 0 because NVML cannot distinguish "this process holds
        no device memory" from "this process is not visible to NVML" -- a PID
        namespace mismatch is the usual cause, and only the caller knows whether
        a device context is expected to exist by this point. Callers read a
        returned int as an authoritative measurement, so 0 would size budgets as
        though the process were empty; None routes them to their own fallback.

    Raises:
        RuntimeError: If device validation fails (invalid index or UUID).
    """
    from vllm.third_party.pynvml import nvmlDeviceGetCount

    my_pid = os.getpid()
    visible_devices = parse_cuda_visible_devices()

    try:
        nvmlInit()
    except Exception as e:
        logger.warning("NVML init failed, will use profiling fallback: %s", e)
        return None

    try:
        if visible_devices and local_rank < len(visible_devices):
            device_id = visible_devices[local_rank]
            try:
                handle = get_device_handle(device_id)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to get NVML handle for device '{device_id}' (local_rank={local_rank}). "
                    f"Check CUDA_VISIBLE_DEVICES or stage config 'devices' setting."
                ) from e
        else:
            # No CUDA_VISIBLE_DEVICES or local_rank out of range: use index directly
            device_count = nvmlDeviceGetCount()
            if local_rank >= device_count:
                raise RuntimeError(
                    f"Invalid GPU device {local_rank}. Only {device_count} GPU(s) available. "
                    f"Check CUDA_VISIBLE_DEVICES or stage config 'devices' setting."
                )
            handle = nvmlDeviceGetHandleByIndex(local_rank)

        for proc in nvmlDeviceGetComputeRunningProcesses(handle):
            if proc.pid == my_pid:
                return proc.usedGpuMemory
        logger.warning(
            "PID %d is not in the NVML compute-process list for GPU %d, so per-process "
            "memory cannot be measured (a PID namespace mismatch is the usual cause).",
            my_pid,
            local_rank,
        )
        return None
    except RuntimeError:
        raise
    except Exception as e:
        logger.warning("NVML query failed, will use profiling fallback: %s", e)
        return None
    finally:
        try:
            nvmlShutdown()
        except Exception:
            pass
