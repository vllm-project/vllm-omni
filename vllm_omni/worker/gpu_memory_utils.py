# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""NVML-based per-process GPU memory utilities.

Used by DiffusionWorker for process-scoped memory reporting. LLM workers size
their KV cache through device-level profiling, without these NVML helpers.
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
        Memory in bytes used by this process, or None if NVML unavailable.

    Raises:
        ValueError: If ``CUDA_VISIBLE_DEVICES`` is set (even to the empty
            string, which hides every device) but ``local_rank`` is not a
            valid index into it, i.e. a misconfigured deploy ``devices``
            setting.
        RuntimeError: If device validation fails (invalid index or UUID).
    """
    from vllm.third_party.pynvml import nvmlDeviceGetCount

    my_pid = os.getpid()
    # An empty CUDA_VISIBLE_DEVICES is a mask too: CUDA hides every device.
    mask_set = "CUDA_VISIBLE_DEVICES" in os.environ
    visible_devices = parse_cuda_visible_devices()

    try:
        nvmlInit()
    except Exception as e:
        logger.warning("NVML init failed, will use profiling fallback: %s", e)
        return None

    try:
        if mask_set:
            if not 0 <= local_rank < len(visible_devices):
                raise ValueError(
                    f"local_rank {local_rank} is not a valid index into CUDA_VISIBLE_DEVICES="
                    f"{os.environ['CUDA_VISIBLE_DEVICES']!r} ({len(visible_devices)} visible device(s)). "
                    "Check the deploy config `devices` setting."
                )
            device_id = visible_devices[local_rank]
            try:
                handle = get_device_handle(device_id)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to get NVML handle for device '{device_id}' (local_rank={local_rank}). "
                    f"Check CUDA_VISIBLE_DEVICES or the deploy config `devices` setting."
                ) from e
        else:
            # No CUDA_VISIBLE_DEVICES mask: local_rank is a physical index
            device_count = nvmlDeviceGetCount()
            if not 0 <= local_rank < device_count:
                raise RuntimeError(
                    f"Invalid GPU device {local_rank}. Only {device_count} GPU(s) available. "
                    f"Check CUDA_VISIBLE_DEVICES or the deploy config `devices` setting."
                )
            handle = nvmlDeviceGetHandleByIndex(local_rank)

        for proc in nvmlDeviceGetComputeRunningProcesses(handle):
            if proc.pid == my_pid:
                return proc.usedGpuMemory
        return 0
    except (RuntimeError, ValueError):
        raise
    except Exception as e:
        logger.warning("NVML query failed, will use profiling fallback: %s", e)
        return None
    finally:
        try:
            nvmlShutdown()
        except Exception:
            pass
