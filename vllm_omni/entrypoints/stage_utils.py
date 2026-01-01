from __future__ import annotations

import enum
import json
import logging
import os
from multiprocessing import shared_memory as _shm
from typing import Any

from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


class OmniStageTaskType(enum.Enum):
    GENERATE = "generate"
    ABORT = "abort"
    SHUTDOWN = "shutdown"


SHUTDOWN_TASK = {"type": OmniStageTaskType.SHUTDOWN}


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
    from vllm_omni.utils import detect_device_type, get_device_control_env_var

    if device_type is None:
        device_type = detect_device_type()

    env_var = get_device_control_env_var()

    # Select device-specific torch functions
    if device_type == "npu":
        try:
            import torch.npu  # type: ignore[import-untyped]
        except ImportError:
            logger.debug("[Stage-%s] torch.npu not available, skipping NPU device setup", stage_id)
            return

        is_available_fn = torch.npu.is_available
        set_device_fn = torch.npu.set_device
        device_count_fn = torch.npu.device_count
        get_device_properties_fn = torch.npu.get_device_properties
        mem_get_info_fn = torch.npu.mem_get_info
        get_device_name_fn = torch.npu.get_device_name
        device_type_label = "NPU"
    elif device_type == "cuda":
        import torch

        is_available_fn = torch.cuda.is_available
        set_device_fn = torch.cuda.set_device
        device_count_fn = torch.cuda.device_count
        get_device_properties_fn = torch.cuda.get_device_properties
        mem_get_info_fn = torch.cuda.mem_get_info
        get_device_name_fn = torch.cuda.get_device_name
        device_type_label = "CUDA"
    else:
        logger.debug("[Stage-%s] Unsupported device type: %s", stage_id, device_type)
        return

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
                    logger.debug("[Stage-%s] Failed to parse first %s device: %s", stage_id, device_type_label, e)
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

        try:
            import torch

            if is_available_fn():
                try:
                    set_device_fn(0)
                except Exception as e:
                    logger.debug(
                        "[Stage-%s] %s set_device(0) failed: %s", stage_id, device_type_label, e, exc_info=True
                    )
                num = device_count_fn()
                info = []
                for i in range(num):
                    total = get_device_properties_fn(i).total_memory
                    free, _ = mem_get_info_fn(i)
                    info.append(
                        {
                            "idx": i,
                            "name": get_device_name_fn(i),
                            "total": int(total),
                            "free": int(free),
                        }
                    )
                logger.debug("[Stage-%s] %s devices visible=%s info=%s", stage_id, device_type_label, num, info)
        except Exception as e:
            logger.debug("[Stage-%s] Failed to query %s devices: %s", stage_id, device_type_label, e, exc_info=True)
    except Exception as e:
        logger.warning("Failed to interpret devices for stage %s: %s", stage_id, e)


def serialize_obj(obj: Any) -> bytes:
    """Serialize a Python object to bytes using centralized serializer (defaults to cloudpickle)."""
    from vllm_omni.distributed.omni_connectors.utils.serialization import OmniSerializer

    return OmniSerializer.serialize(obj)


def shm_write_bytes(payload: bytes) -> dict[str, Any]:
    """Write bytes into SharedMemory and return meta dict {name,size}.

    Caller should close the segment; the receiver should unlink.
    """
    shm = _shm.SharedMemory(create=True, size=len(payload))
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


def _ensure_parent_dir(path: str) -> None:
    """Ensure the parent directory for a file path exists (best-effort)."""
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
    except Exception:
        pass


def append_jsonl(path: str, record: dict[str, Any]) -> None:
    """Append a JSON record as one line to a JSONL file (best-effort).

    This is safe to call from multiple processes when each process writes
    to a distinct file. For concurrent writes to the same file, OS append
    semantics typically suffice, but no additional locking is provided.
    """
    try:
        _ensure_parent_dir(path)
        line = json.dumps(record, ensure_ascii=False)
        fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
        with os.fdopen(fd, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        logger.exception("Failed to append JSONL to %s", path)


def maybe_dump_to_shm(obj: Any, threshold: int) -> tuple[bool, Any]:
    """Dump object to SHM if serialized size exceeds threshold.

    Returns (True, meta) when dumped; otherwise (False, original_obj).
    """
    payload = serialize_obj(obj)
    if len(payload) > threshold:
        return True, shm_write_bytes(payload)
    return False, obj


def maybe_load_from_ipc(container: dict[str, Any], obj_key: str, shm_key: str) -> Any:
    """Load object from container that may carry SHM or inline object.

    Deprecated: prefer `maybe_load_from_ipc_with_metrics` to also obtain
    decode-time and size metrics.
    """
    if shm_key in container:
        from vllm_omni.distributed.omni_connectors.utils.serialization import OmniSerializer

        return OmniSerializer.deserialize(shm_read_bytes(container[shm_key]))
    return container[obj_key]


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


def encode_for_ipc(obj: Any, threshold: int, obj_key: str, shm_key: str) -> dict[str, Any]:
    """Return a dict payload for IPC: inline (obj_key) or SHM (shm_key).

    When serialized size exceeds threshold, returns {shm_key: {name,size}};
    otherwise returns {obj_key: obj}.
    """
    payload: dict[str, Any] = {}
    use_shm, data = maybe_dump_to_shm(obj, threshold)
    if use_shm:
        payload[shm_key] = data
    else:
        payload[obj_key] = data
    return payload


# Convert OmegaConf/objects to plain dicts
def _to_dict(x: Any) -> dict[str, Any]:
    try:
        if isinstance(x, dict):
            return dict(x)
        return OmegaConf.to_container(x, resolve=True)  # type: ignore[arg-type]
    except Exception:
        try:
            return dict(x)
        except Exception:
            return {}
