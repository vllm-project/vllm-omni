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
        self._pending_keys: set[str] = set()
        self._metrics = {
            "puts": 0,
            "gets": 0,
            "bytes_transferred": 0,
        }

    def put(
        self,
        from_stage: str,
        to_stage: str,
        put_key: str,
        data: Any,
    ) -> tuple[bool, int, dict[str, Any] | None]:
        try:
            payload = self.serialize_obj(data)
            size = len(payload)

            lock_file = f"/dev/shm/shm_{put_key}_lockfile.lock"
            with open(lock_file, "wb+") as lockf:
                fcntl.flock(lockf, fcntl.LOCK_EX)
                meta = shm_write_bytes(payload, name=put_key)
                fcntl.flock(lockf, fcntl.LOCK_UN)

            # meta contains {'name': ..., 'size': ...}
            metadata = {"shm": meta, "size": size}
            self._pending_keys.add(put_key)

            self._metrics["puts"] += 1
            self._metrics["bytes_transferred"] += size

            return True, size, metadata

        except Exception as e:
            logger.error(f"SharedMemoryConnector put failed for req {put_key}: {e}")
            return False, 0, None

    def _get_data_with_lock(self, lock_file: str, shm_handle: dict[str, Any]) -> tuple[Any, int] | None:
        deserialized = False
        try:
            with open(lock_file, "rb+") as lockf:
                fcntl.flock(lockf, fcntl.LOCK_EX)
                data_bytes = shm_read_bytes(shm_handle)
                fcntl.flock(lockf, fcntl.LOCK_UN)
            obj = self.deserialize_obj(data_bytes)
            result = (obj, int(shm_handle.get("size", 0)))
            deserialized = True
            return result
        except Exception as e:
            logger.error(f"SharedMemoryConnector shm get failed for req : {e}")
            return None
        finally:
            if deserialized:
                try:
                    os.remove(lock_file)
                except FileNotFoundError:
                    pass

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
        except ValueError as e:
            # A receiver can observe a newly-created POSIX SHM object before
            # the writer has finished sizing it. Treat that as "not ready yet"
            # so async polling can retry without a traceback.
            if "empty file" in str(e):
                return None
            logger.debug("_get_by_key: unexpected error reading SHM segment %s", get_key, exc_info=True)
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

            if isinstance(metadata, dict) and "shm" in metadata:
                shm_handle = metadata["shm"]
                lock_file = f"/dev/shm/shm_{shm_handle['name']}_lockfile.lock"
                result = self._get_data_with_lock(lock_file, shm_handle)
                if result is not None:
                    self._pending_keys.discard(get_key)
            else:
                # Missing or non-SHM metadata falls back to key-based lookup.
                result = self._get_by_key(get_key)
        else:
            result = self._get_by_key(get_key)

        if result is not None:
            self._metrics["gets"] += 1
        return result

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
        return {"status": "healthy", **self._metrics}
