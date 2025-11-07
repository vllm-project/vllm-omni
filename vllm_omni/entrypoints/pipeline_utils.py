from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import logging
import json
import os
import pickle
from multiprocessing import shared_memory as _shm

import cloudpickle

logger = logging.getLogger(__name__)


def set_stage_gpu_devices(stage_id: int, devices: Optional[Union[str, int]]) -> None:
    """Configure per-stage CUDA visibility and current device.

    Behavior
    - Comma-separated string (e.g. "2,5,7"): set CUDA_VISIBLE_DEVICES exactly
      to this list; logical index 0 is used as current device.
    - Integer or digit-string: treat as logical index (0-based) into the current
      CUDA_VISIBLE_DEVICES mapping; map to the physical device, and then set
      CUDA_VISIBLE_DEVICES to this single device.
    - None/"cpu": keep default visibility.
    - Otherwise: set CUDA_VISIBLE_DEVICES to the provided single device string.
    """
    try:
        selected_physical: Optional[int] = None
        logical_idx: Optional[int] = None

        if isinstance(devices, str) and "," in devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = devices
            toks = [t.strip() for t in devices.split(",") if t.strip() != ""]
            if toks:
                try:
                    selected_physical = int(toks[0])
                    logger.debug(
                        "[Stage-%s] Set CUDA_VISIBLE_DEVICES to %s; logical 0 -> physical %s",
                        stage_id, devices, selected_physical,
                    )
                except Exception as e:
                    logger.debug("[Stage-%s] Failed to parse first CUDA device: %s", stage_id, e)
                    selected_physical = None
        elif isinstance(devices, (int, str)) and (isinstance(devices, int) or str(devices).isdigit()):
            logical_idx = max(0, int(devices))
            vis = os.environ.get("CUDA_VISIBLE_DEVICES")
            if vis:
                try:
                    mapping = [int(x) for x in vis.split(",") if x.strip() != ""]
                    if 0 <= logical_idx < len(mapping):
                        selected_physical = mapping[logical_idx]
                except Exception as e:
                    logger.debug("[Stage-%s] Failed to map logical index via CUDA_VISIBLE_DEVICES: %s", stage_id, e)
                    selected_physical = None
            if selected_physical is None:
                selected_physical = int(logical_idx)
            os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_physical)
            logger.debug(
                "[Stage-%s] Logical index %d -> physical %s; set CUDA_VISIBLE_DEVICES to single device",
                stage_id, logical_idx + 1, selected_physical,
            )
        elif devices in (None, "cpu"):
            logger.debug("[Stage-%s] Using default device visibility (devices=%s)", stage_id, devices)
        else:
            selected_physical = int(str(devices))
            os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_physical)
            logger.debug("[Stage-%s] Set CUDA_VISIBLE_DEVICES to single device %s (fallback)", stage_id, selected_physical)

        try:
            import torch  # noqa: WPS433
            if torch.cuda.is_available():
                try:
                    torch.cuda.set_device(0)
                except Exception as e:
                    logger.debug("[Stage-%s] torch.cuda.set_device(0) failed: %s", stage_id, e, exc_info=True)
                num = torch.cuda.device_count()
                info = []
                for i in range(num):
                    total = torch.cuda.get_device_properties(i).total_memory
                    free, _ = torch.cuda.mem_get_info(i)
                    info.append({
                        "idx": i,
                        "name": torch.cuda.get_device_name(i),
                        "total": int(total),
                        "free": int(free),
                    })
                logger.debug("[Stage-%s] CUDA devices visible=%s info=%s", stage_id, num, info)
        except Exception as e:
            logger.debug("[Stage-%s] Failed to query CUDA devices: %s", stage_id, e, exc_info=True)
    except Exception as e:
        logger.warning("Failed to interpret devices for stage %s: %s", stage_id, e)


def serialize_obj(obj: Any) -> bytes:
    """Serialize a Python object to bytes using cloudpickle."""
    return cloudpickle.dumps(obj)


def shm_write_bytes(payload: bytes) -> Dict[str, Any]:
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
        logger.debug("Ensure parent dir failed for %s: %s", path, e)
    return meta


def shm_read_bytes(meta: Dict[str, Any]) -> bytes:
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


def append_jsonl(path: str, record: Dict[str, Any]) -> None:
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


def maybe_dump_to_shm(obj: Any, threshold: int) -> Tuple[bool, Any]:
    """Dump object to SHM if serialized size exceeds threshold.

    Returns (True, meta) when dumped; otherwise (False, original_obj).
    """
    payload = serialize_obj(obj)
    if len(payload) > threshold:
        return True, shm_write_bytes(payload)
    return False, obj


def maybe_load_from_ipc(container: Dict[str, Any], obj_key: str, shm_key: str) -> Any:
    """Load object from container that may carry SHM or inline object."""
    if shm_key in container:
        return pickle.loads(shm_read_bytes(container[shm_key]))
    return container[obj_key]


def encode_for_ipc(obj: Any, threshold: int, obj_key: str, shm_key: str) -> Dict[str, Any]:
    """Return a dict payload for IPC: inline (obj_key) or SHM (shm_key).

    When serialized size exceeds threshold, returns {shm_key: {name,size}};
    otherwise returns {obj_key: obj}.
    """
    payload: Dict[str, Any] = {}
    use_shm, data = maybe_dump_to_shm(obj, threshold)
    if use_shm:
        payload[shm_key] = data
    else:
        payload[obj_key] = data
    return payload


# Convert OmegaConf/objects to plain dicts
def _to_dict(x: Any) -> Dict[str, Any]:
    try:
        if isinstance(x, dict):
            return dict(x)
        return OmegaConf.to_container(x, resolve=True)  # type: ignore[arg-type]
    except Exception:
        try:
            return dict(x)
        except Exception:
            return {}
