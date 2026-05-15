from __future__ import annotations

import logging
import os
from multiprocessing import shared_memory as _shm
from typing import Any

from vllm_omni.config.yaml_util import to_dict as _omega_to_dict
from vllm_omni.platforms import current_omni_platform

logger = logging.getLogger(__name__)


def set_stage_devices(
    stage_id: int,
    devices: str | int | None,
) -> str | None:
    """Configure per-stage device visibility and current device (CUDA or NPU).

    This function sets environment variables that control which devices are visible
    to the process. It must be called BEFORE worker initialization so that workers
    see the correct devices.


    NOTE: This will set the control variable for the appropriate platform.
        - CUDA: CUDA_VISIBLE_DEVICES
        - NPU: ASCEND_RT_VISIBLE_DEVICES

    Args:
        stage_id: Stage identifier for logging
        devices: Devices specified as either:
            - None / "cpu"; uses the default visibility.
            - An int or a str composed of one or more ints separated by commas,
              which correspond to logical indices. If the control env var is
              set, e.g., CUDA_VISIBLE_DEVICES, we will map the logical indices
              to physical, e.g.,
                    devices: [0,1,2,3]
                    CUDA_VISIBLE_DEVICES -> [1, 3, 4, 5, 6]
            will leverage [1, 3, 4, 5]

    Returns:
        The list of physical devices that were set for the given stage
        or None if we have no passed devices / are using cpu.
    """
    env_var = current_omni_platform.device_control_env_var
    vis = os.environ.get(env_var)

    if devices in (None, "cpu"):
        logger.debug("[Stage-%s] Using default device visibility (devices=%s)", stage_id, devices)
        return None

    elif isinstance(devices, (int, str)):
        device_list = _parse_device_list(devices)
        if vis is not None:
            visible_device_list = _parse_device_list(vis)
            device_list = _map_device_list(stage_id, device_list, visible_device_list)
        device_str = ",".join(device_list)
        current_omni_platform.set_device_control_env_var(device_str)
        return device_str

    raise TypeError(f"Expected str or int device IDs for stage initialization, got type {type(devices)}")


def _parse_device_list(devices: str | int) -> list[str]:
    """Given an int or a str representing one or more comma separated
    non-negative IDs, coerce it to a list of strs.

    Args:
        devices: devices to be converted to a list of strs.
    """
    if isinstance(devices, int):
        if devices < 0:
            raise ValueError("Device IDs must be non-negative integers!")
        return [str(devices)]
    # Devices will usually be ints, but not always
    # so we don't explicitly validate that here.
    return [t.strip() for t in devices.split(",") if t.strip() != ""]


def _map_device_list(stage_id: int, device_list: list[str], visible_device_list: list[str]) -> list[str]:
    """Map logical stage devices onto the currently available device pool.

    Args:
        stage_id: The stage ID currently configuring devices.
        device_list: List of (logical) devices to be used, which are strings
            holding non-negative nums counting from 0, 1, ..., n devices needed.
        visible_device_list: List of physical devices available.
    """
    num_visible = len(visible_device_list)

    # Ensure that the logical IDs are actually in range to avoid index errors;
    # if some requested ids exceed the available pool, we will fall back to the
    # subset that can be mapped and leave the final capacity check to the later
    # parallel-config validation path.
    if not all(device.isdigit() for device in device_list):
        raise ValueError("Logical devices must be non-negative integers")

    logical_ids = [int(device) for device in device_list]
    mapped_devices = [visible_device_list[idx] for idx in logical_ids if idx < num_visible]
    mapping_pairs = [
        f"{logical_id}->{visible_device_list[logical_id]}" for logical_id in logical_ids if logical_id < num_visible
    ]
    if not mapped_devices:
        raise ValueError(
            f"Stage {stage_id} has logical IDs {device_list}, none of which map to the visible devices "
            f"{visible_device_list}"
        )
    if len(mapped_devices) < len(logical_ids):
        logger.warning(
            "Stage %s requested logical devices %s, but only %d device(s) are currently available: %s. "
            "Resolved logical-to-physical mapping: %s. Falling back to mapped subset %s",
            stage_id,
            device_list,
            num_visible,
            visible_device_list,
            ", ".join(mapping_pairs) if mapping_pairs else "(none)",
            mapped_devices,
        )
    else:
        logger.info(
            "Stage %s logical-to-physical device mapping: %s",
            stage_id,
            ", ".join(mapping_pairs),
        )
    return mapped_devices


def serialize_obj(obj: Any) -> bytes:
    """Serialize a Python object to bytes using centralized serializer (defaults to cloudpickle)."""
    from vllm_omni.distributed.omni_connectors.utils.serialization import OmniSerializer

    return OmniSerializer.serialize(obj)


def shm_write_bytes(payload: bytes, name: str | None = None) -> dict[str, Any]:
    """Write bytes into SharedMemory and return meta dict {name,size}.

    Caller should close the segment; the receiver should unlink.
    """
    try:
        shm = _shm.SharedMemory(create=True, size=len(payload), name=name)
    except FileExistsError:
        if name:
            # If name is specified and exists, unlink it and try again
            try:
                existing = _shm.SharedMemory(name=name)
                existing.unlink()
            except Exception:
                pass
            shm = _shm.SharedMemory(create=True, size=len(payload), name=name)
        else:
            raise

    mv = memoryview(shm.buf)
    mv[: len(payload)] = payload
    del mv
    meta = {"name": shm.name, "size": len(payload)}
    try:
        shm.close()
    except Exception as e:
        logger.debug("Failed to close shared memory: %s", e)
    return meta


def shm_read_bytes(meta: dict[str, Any]) -> bytes:
    """Read bytes from SharedMemory by meta {name,size} and cleanup."""
    shm = _shm.SharedMemory(name=meta["name"])  # type: ignore[index]
    mv = memoryview(shm.buf)
    data = bytes(mv[: meta["size"]])
    del mv
    try:
        shm.close()
    except Exception:
        pass
    try:
        shm.unlink()
    except Exception:
        pass
    return data


def maybe_load_from_ipc_with_metrics(
    container: dict[str, Any], obj_key: str, shm_key: str
) -> tuple[Any, dict[str, float]]:
    """Load object and return (object, metrics) with RX bytes and decode time.

    Metrics keys:
      - rx_transfer_bytes: int
      - rx_decode_time_ms: float
    """
    import time as _time  # local import to avoid overhead at module import

    from vllm_omni.distributed.omni_connectors.utils.serialization import OmniSerializer

    t0 = _time.time()
    if shm_key in container:
        meta = container[shm_key]  # type: ignore[index]
        payload = shm_read_bytes(meta)
        obj = OmniSerializer.deserialize(payload)
        try:
            rx_bytes = int(meta.get("size", len(payload)))  # type: ignore[call-arg]
        except Exception:
            rx_bytes = len(payload)
    else:
        obj = container[obj_key]
        try:
            rx_bytes = len(serialize_obj(obj))
        except Exception:
            rx_bytes = 0
    t1 = _time.time()
    rx_decode_ms = (t1 - t0) * 1000.0
    return obj, {
        "rx_transfer_bytes": int(rx_bytes),
        "rx_decode_time_ms": float(rx_decode_ms),
    }


# Convert OmegaConf/objects to plain dicts
def _to_dict(x: Any) -> dict[str, Any]:
    try:
        if isinstance(x, dict):
            return dict(x)
        return _omega_to_dict(x)
    except Exception:
        try:
            return dict(x)
        except Exception:
            return {}
