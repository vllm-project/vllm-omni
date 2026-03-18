from __future__ import annotations

import logging
import os
from multiprocessing import shared_memory as _shm
from typing import Any

from vllm_omni.config.yaml_util import to_dict as _omega_to_dict

logger = logging.getLogger(__name__)


def set_stage_devices(
    stage_id: int,
    devices: str | int | None,
    device_type: str | None = None,
) -> None:
    """Configure per-stage device visibility and current device (CUDA or NPU).

    This function sets environment variables that control which devices are visible
    to the process, and sets the current device. It must be called BEFORE worker
    initialization so that workers see the correct devices.

    Args:
        stage_id: Stage identifier for logging
        devices: Device specification:
            - Comma-separated string (e.g. "2,5,7"): interpreted as logical
              indices against the current device visibility env var (e.g.
              CUDA_VISIBLE_DEVICES/ASCEND_RT_VISIBLE_DEVICES) when present;
              falls back to physical IDs if no mapping exists. Logical index 0
              is used as current device.
            - Integer or digit-string: treat as logical index (0-based) into the
              current device visibility mapping; map to physical device, then set
              env var to this single device.
            - None/"cpu": keep default visibility.
            - Otherwise: set env var to the provided single device string.
        device_type: Device type ("cuda" or "npu"). If None, auto-detects.

    Behavior:
        - CUDA: Sets CUDA_VISIBLE_DEVICES and calls torch.cuda.set_device()
        - NPU: Sets ASCEND_RT_VISIBLE_DEVICES and calls torch.npu.set_device()
    """
    from vllm_omni.platforms import current_omni_platform

    if device_type is None:
        device_type = current_omni_platform.device_type

    env_var = current_omni_platform.device_control_env_var

    try:
        selected_physical: int | None = None
        logical_idx: int | None = None

        if isinstance(devices, str) and "," in devices:
            toks = [t.strip() for t in devices.split(",") if t.strip() != ""]
            vis = os.environ.get(env_var)
            mapped_devices: list[str] = []
            mapping: list[int] = []
            if vis:
                try:
                    mapping = [int(x) for x in vis.split(",") if x.strip() != ""]
                except Exception as e:
                    logger.debug("[Stage-%s] Failed to parse existing %s: %s", stage_id, env_var, e)
            for tok in toks:
                try:
                    idx = int(tok)
                except Exception:
                    mapped_devices.append(tok)
                    continue
                if mapping and 0 <= idx < len(mapping):
                    mapped_devices.append(str(mapping[idx]))
                else:
                    mapped_devices.append(str(idx))
            mapped_devices_str = ",".join(mapped_devices)
            os.environ[env_var] = mapped_devices_str
            if toks:
                try:
                    selected_physical = int(mapped_devices[0])
                    logger.debug(
                        "[Stage-%s] Set %s to %s; logical 0 -> physical %s",
                        stage_id,
                        env_var,
                        mapped_devices_str,
                        selected_physical,
                    )
                except Exception as e:
                    logger.debug("[Stage-%s] Failed to parse first %s device: %s", stage_id, device_type, e)
                    selected_physical = None
        elif isinstance(devices, (int, str)) and (isinstance(devices, int) or str(devices).isdigit()):
            logical_idx = max(0, int(devices))
            vis = os.environ.get(env_var)
            if vis:
                try:
                    mapping = [int(x) for x in vis.split(",") if x.strip() != ""]
                    if 0 <= logical_idx < len(mapping):
                        selected_physical = mapping[logical_idx]
                except Exception as e:
                    logger.debug("[Stage-%s] Failed to map logical index via %s: %s", stage_id, env_var, e)
                    selected_physical = None
            if selected_physical is None:
                selected_physical = int(logical_idx)
            os.environ[env_var] = str(selected_physical)
            logger.debug(
                "[Stage-%s] Logical index %d -> physical %s; set %s to single device",
                stage_id,
                logical_idx + 1,
                selected_physical,
                env_var,
            )
        elif devices in (None, "cpu"):
            logger.debug("[Stage-%s] Using default device visibility (devices=%s)", stage_id, devices)
        else:
            selected_physical = int(str(devices))
            os.environ[env_var] = str(selected_physical)
            logger.debug("[Stage-%s] Set %s to single device %s (fallback)", stage_id, env_var, selected_physical)
    except Exception as e:
        logger.warning("Failed to interpret devices for stage %s: %s", stage_id, e)


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
