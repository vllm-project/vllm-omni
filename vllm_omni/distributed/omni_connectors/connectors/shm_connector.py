# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import fcntl
import os
from multiprocessing import shared_memory as shm_pkg
from typing import Any

from vllm_omni.entrypoints.stage_utils import shm_read_bytes, shm_write_bytes

from ..utils.logging import get_connector_logger
from .base import OmniConnectorBase

logger = get_connector_logger(__name__)


class SharedMemoryConnector(OmniConnectorBase):
    """Key-addressed local shared-memory connector.

    SHM is a local-only transport: it reads/writes POSIX shared memory
    segments identified purely by *key*.  It does **not** understand
    remote-transport metadata such as ``source_host`` / ``source_port``
    (that is the RDMA connector's job).  When such metadata is passed in,
    the connector silently falls back to key-based lookup.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.stage_id = config.get("stage_id", -1)
        self.device = config.get("device", "cuda:0")
        self.threshold = int(config.get("shm_threshold_bytes", 65536))
        self._pending_keys: set[str] = set()
        self._metrics = {
            "puts": 0,
            "gets": 0,
            "bytes_transferred": 0,
            "shm_writes": 0,
            "inline_writes": 0,
        }

    def put(
        self,
        from_stage: str,
        to_stage: str,
        put_key: str,
        data: Any,
    ) -> tuple[bool, int, dict[str, Any] | None]:
        try:
            # Always serialize first to check size (and for SHM writing)
            # Note: For extremely large objects in "inline" mode (e.g. Ray),
            # we might double-serialize if we're not careful, but here we assume
            # if it's huge we use SHM, or if Ray, threshold is maxsize.
            payload = self.serialize_obj(data)
            size = len(payload)

            # Currently, we always use SHM.
            if True:
                # Use Shared Memory
                lock_file = f"/dev/shm/shm_{put_key}_lockfile.lock"
                with open(lock_file, "wb+") as lockf:
                    fcntl.flock(lockf, fcntl.LOCK_EX)
                    meta = shm_write_bytes(payload, name=put_key)
                    fcntl.flock(lockf, fcntl.LOCK_UN)

                # meta contains {'name': ..., 'size': ...}
                metadata = {"shm": meta, "size": size}
                self._pending_keys.add(put_key)
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
            logger.error(f"SharedMemoryConnector put failed for req {put_key}: {e}")
            return False, 0, None

    def _get_data_with_lock(self, lock_file: str, shm_handle: dict):
        obj = None
        try:
            with open(lock_file, "rb+") as lockf:
                fcntl.flock(lockf, fcntl.LOCK_EX)
                data_bytes = shm_read_bytes(shm_handle)
                fcntl.flock(lockf, fcntl.LOCK_UN)
            obj = self.deserialize_obj(data_bytes)
            return obj, int(shm_handle.get("size", 0))
        except Exception as e:
            logger.error(f"SharedMemoryConnector shm get failed for req : {e}")
            return None
        finally:
            # If data has been received, delete lock_file.
            if obj and os.path.exists(lock_file):
                os.remove(lock_file)

    def _get_by_key(self, get_key: str) -> tuple[Any, int] | None:
        """Read a SHM segment addressed purely by *get_key*."""
        shm = None
        try:
            shm = shm_pkg.SharedMemory(name=get_key)
            if shm is None or shm.size == 0:
                return None
            lock_file = f"/dev/shm/shm_{get_key}_lockfile.lock"
            shm_handle = {"name": get_key, "size": shm.size}
            result = self._get_data_with_lock(lock_file, shm_handle)
            if result is not None:
                self._pending_keys.discard(get_key)
            return result
        except FileNotFoundError:
            return None
        except Exception:
            logger.debug("_get_by_key: unexpected error reading SHM segment %s", get_key, exc_info=True)
            return None
        finally:
            if shm:
                shm.close()

    def get(
        self,
        from_stage: str,
        to_stage: str,
        get_key: str,
        metadata=None,
    ) -> tuple[Any, int] | None:
        if metadata is not None:
            if isinstance(metadata, dict) and get_key in metadata:
                metadata = metadata.get(get_key)

            if not isinstance(metadata, dict):
                return self._get_by_key(get_key)

            if "inline_bytes" in metadata:
                try:
                    obj = self.deserialize_obj(metadata["inline_bytes"])
                    self._pending_keys.discard(get_key)
                    return obj, int(metadata.get("size", 0))
                except Exception as e:
                    logger.error(f"SharedMemoryConnector inline get failed for req {get_key}: {e}")
                    return None

            if "shm" in metadata:
                shm_handle = metadata["shm"]
                lock_file = f"/dev/shm/shm_{shm_handle['name']}_lockfile.lock"
                result = self._get_data_with_lock(lock_file, shm_handle)
                if result is not None:
                    self._pending_keys.discard(get_key)
                return result

            # Metadata is a dict but has no SHM-specific handle (e.g. RDMA-
            # style source_host/source_port).  Fall back to key-based read.
            return self._get_by_key(get_key)

        return self._get_by_key(get_key)

    def cleanup(self, request_id: str) -> None:
        """Best-effort cleanup of unconsumed SHM segments for *request_id*.

        Matches pending keys where *request_id* appears as the full key,
        as a ``_``-delimited prefix, or as a ``_``-delimited suffix.
        If ``get()`` was never called, we unlink it here so /dev/shm
        doesn't leak.
        """
        stale = [
            k
            for k in self._pending_keys
            if k == request_id or k.startswith(request_id + "_") or k.endswith("_" + request_id)
        ]
        for key in stale:
            self._pending_keys.discard(key)
            try:
                seg = shm_pkg.SharedMemory(name=key)
                seg.close()
                seg.unlink()
                logger.debug("cleanup: unlinked unconsumed SHM segment %s", key)
            except FileNotFoundError:
                pass
            except Exception as e:
                logger.debug("cleanup: failed to unlink SHM segment %s: %s", key, e)
            lock_file = f"/dev/shm/shm_{key}_lockfile.lock"
            if os.path.exists(lock_file):
                try:
                    os.remove(lock_file)
                except OSError:
                    pass

    def close(self) -> None:
        """Unlink all remaining tracked SHM segments."""
        for key in list(self._pending_keys):
            try:
                seg = shm_pkg.SharedMemory(name=key)
                seg.close()
                seg.unlink()
            except Exception:
                pass
            lock_file = f"/dev/shm/shm_{key}_lockfile.lock"
            if os.path.exists(lock_file):
                try:
                    os.remove(lock_file)
                except OSError:
                    pass
        self._pending_keys.clear()

    def health(self) -> dict[str, Any]:
        return {"status": "healthy", "threshold": self.threshold, **self._metrics}
