# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

from vllm_omni.entrypoints.stage_utils import shm_read_bytes, shm_write_bytes

from ..utils.logging import get_connector_logger
from .base import OmniConnectorBase

logger = get_connector_logger(__name__)


class SharedMemoryConnector(OmniConnectorBase):
    """
    Connector that uses SharedMemory for large objects and inline data for small objects.
    Acts as a unified replacement for the legacy IPC fallback logic.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        # Default threshold matches legacy behavior (64KB)
        self.threshold = int(config.get("shm_threshold_bytes", 65536))
        self._metrics = {
            "puts": 0,
            "gets": 0,
            "bytes_transferred": 0,
            "shm_writes": 0,
            "inline_writes": 0,
        }

    def put(
        self, from_stage: str, to_stage: str, request_id: str, data: Any
    ) -> tuple[bool, int, dict[str, Any] | None]:
        try:
            # Always serialize first to check size (and for SHM writing)
            # Note: For extremely large objects in "inline" mode (e.g. Ray),
            # we might double-serialize if we're not careful, but here we assume
            # if it's huge we use SHM, or if Ray, threshold is maxsize.
            payload = self.serialize_obj(data)
            size = len(payload)

            if size > self.threshold:
                # Use Shared Memory
                shm_name = f"omni_{from_stage}_to_{to_stage}_{request_id}"
                meta = shm_write_bytes(payload, name=shm_name)
                # meta contains {'name': ..., 'size': ...}
                metadata = {"shm": meta, "size": size}
                self._metrics["shm_writes"] += 1
            else:
                # Inline - pass bytes directly to avoid double serialization of the object
                # We already serialized it to check size, so we pass the bytes.
                # The Queue will pickle these bytes (fast), avoiding re-serializing the complex object.
                metadata = {"inline_bytes": payload, "size": size}
                self._metrics["inline_writes"] += 1

            self._metrics["puts"] += 1
            self._metrics["bytes_transferred"] += size

            return True, size, metadata

        except Exception as e:
            logger.error(f"SharedMemoryConnector put failed for req {request_id}: {e}")
            return False, 0, None

    def get(
        self, from_stage: str, to_stage: str, request_id: str, metadata: dict[str, Any] | None = None
    ) -> tuple[Any, int] | None:
        # Helper to try reading SHM with a specific name
        def try_read_shm(name: str):
            # _shm is imported at module level as: import multiprocessing.shared_memory as _shm
            # but to be safe inside this closure if not picked up:
            from multiprocessing import shared_memory as shm_pkg

            shm = shm_pkg.SharedMemory(name=name)
            try:
                # We assume the buffer size matches the data size
                return shm_read_bytes({"name": name, "size": shm.size})
            finally:
                shm.close()

        if not metadata:
            # Try to infer metadata from request_id for KV cache transfer scenarios
            # where metadata is not passed out-of-band.
            shm_name = f"omni_{from_stage}_to_{to_stage}_{request_id}"

            # We need the size to read from SHM using stage_utils.shm_read_bytes
            try:
                # Use local import to avoid potential scope issues if _shm isn't available
                from multiprocessing import shared_memory as shm_pkg

                temp_shm = shm_pkg.SharedMemory(name=shm_name)
                size = temp_shm.size
                temp_shm.close()

                metadata = {"shm": {"name": shm_name, "size": size}, "is_fallback": True}
                logger.debug(f"SharedMemoryConnector inferred metadata for {request_id}: {metadata}")
            except Exception as e:
                # If file not found or other error, standard path will likely fail too, but we let it try or return None
                logger.debug(f"SharedMemoryConnector failed to infer metadata for {request_id} (fallback): {e}")
                # We can't proceed without size if we rely on shm_read_bytes
                return None

        try:
            obj = None
            size = 0

            if "shm" in metadata:
                meta = metadata["shm"]
                # shm_read_bytes handles reading and unlinking
                data_bytes = shm_read_bytes(meta)
                obj = self.deserialize_obj(data_bytes)
                size = metadata.get("size", len(data_bytes))
            elif "inline_bytes" in metadata:
                # Deserialize bytes back to object
                payload = metadata["inline_bytes"]
                obj = self.deserialize_obj(payload)
                size = metadata.get("size", len(payload))
            elif "inline" in metadata:
                obj = metadata["inline"]
                size = metadata.get("size", 0)
                if size == 0:
                    # Fallback if size wasn't recorded
                    try:
                        size = len(self.serialize_obj(obj))
                    except Exception:
                        pass
            else:
                logger.error(
                    f"Unknown metadata format in SharedMemoryConnector for req {request_id}: {list(metadata.keys())}"
                )
                return None

            self._metrics["gets"] += 1
            return obj, size

        except Exception as e:
            logger.error(f"SharedMemoryConnector get failed for req {request_id}: {e}")
            return None

    def cleanup(self, request_id: str) -> None:
        # SHM segments are automatically unlinked during 'get' (shm_read_bytes).
        # If 'get' is never called (e.g. error flow), the SHM segment might leak.
        # A robust implementation might track created segments and unlink them here
        # if they haven't been consumed.
        # For now, we rely on the consumer to read and unlink.
        pass

    def health(self) -> dict[str, Any]:
        return {"status": "healthy", "threshold": self.threshold, **self._metrics}
