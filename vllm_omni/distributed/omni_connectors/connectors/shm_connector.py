# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import fcntl
import os
import threading
import time
from collections import OrderedDict
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

    def get_with_deadline(self, from_stage, to_stage, get_key, metadata=None, *, deadline):
        if time.monotonic() >= deadline:
            return None
        return self.get(from_stage, to_stage, get_key, metadata)

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.stage_id = config.get("stage_id", -1)
        self._pending_keys: OrderedDict[str, None] = OrderedDict()
        self._pending_keys_lock = threading.Lock()
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
            self.reap_consumed()
            payload = self.serialize_obj(data)
            size = len(payload)

            lock_file = f"/dev/shm/shm_{put_key}_lockfile.lock"
            with open(lock_file, "wb+") as lockf:
                fcntl.flock(lockf, fcntl.LOCK_EX)
                meta = shm_write_bytes(payload, name=put_key)
                fcntl.flock(lockf, fcntl.LOCK_UN)

            # meta contains {'name': ..., 'size': ...}
            metadata = {"shm": meta, "size": size}
            with self._pending_keys_lock:
                self._pending_keys[put_key] = None

            self._metrics["puts"] += 1
            self._metrics["bytes_transferred"] += size

            return True, size, metadata

        except Exception as e:
            logger.error(f"SharedMemoryConnector put failed for req {put_key}: {e}")
            return False, 0, None

    def _get_data_with_lock(self, lock_file: str, shm_handle: dict[str, Any]) -> tuple[Any, int] | None:
        consumed = False
        try:
            with open(lock_file, "rb+") as lockf:
                fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
                data_bytes = shm_read_bytes(shm_handle)
                consumed = True
                fcntl.flock(lockf, fcntl.LOCK_UN)
            obj = self.deserialize_obj(data_bytes)
            result = (obj, int(shm_handle.get("size", 0)))
            return result
        except BlockingIOError:
            return None
        except Exception as e:
            logger.error(f"SharedMemoryConnector shm get failed for req : {e}")
            return None
        finally:
            if consumed:
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
                with self._pending_keys_lock:
                    self._pending_keys.pop(get_key, None)
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
                    with self._pending_keys_lock:
                        self._pending_keys.pop(get_key, None)
            else:
                # Missing or non-SHM metadata falls back to key-based lookup.
                result = self._get_by_key(get_key)
        else:
            result = self._get_by_key(get_key)

        if result is not None:
            self._metrics["gets"] += 1
        return result

    def cleanup(self, request_id: str) -> bool:
        """Unlink the exact key passed to ``put()``, never a request-id prefix.

        Returns True when an unconsumed segment was actually unlinked.
        """
        key = request_id
        unlinked = False
        with self._pending_keys_lock:
            self._pending_keys.pop(key, None)
            try:
                seg = shm_pkg.SharedMemory(name=key)
                seg.close()
                seg.unlink()
                logger.debug("cleanup: unlinked unconsumed SHM segment %s", key)
                unlinked = True
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
        return unlinked

    def cleanup_prefix(self, key_prefix: str) -> int:
        """Unlink every tracked key of the form ``{key_prefix}{chunk_id}``.

        Only keys still in ``_pending_keys`` are considered, and the suffix
        must be a bare integer chunk id, so another request whose id merely
        shares the prefix is never matched. Returns the unlinked count.
        """
        with self._pending_keys_lock:
            keys = [k for k in self._pending_keys if k.startswith(key_prefix) and k[len(key_prefix) :].isdigit()]
        return sum(self.cleanup(key) for key in keys)

    def close(self) -> None:
        """Unlink all remaining tracked SHM segments."""
        with self._pending_keys_lock:
            keys = list(self._pending_keys)
        for key in keys:
            self.cleanup(key)

    def reap_consumed(self) -> None:
        """Bounded round-robin sweep; receivers unlink SHM in another process."""
        with self._pending_keys_lock:
            for _ in range(min(64, len(self._pending_keys))):
                key, _ = self._pending_keys.popitem(last=False)
                if os.path.exists(f"/dev/shm/{key}"):
                    self._pending_keys[key] = None

    def health(self) -> dict[str, Any]:
        return {"status": "healthy", **self._metrics}
