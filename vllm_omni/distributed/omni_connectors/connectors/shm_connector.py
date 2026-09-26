# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import fcntl
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from multiprocessing import shared_memory as shm_pkg
from typing import Any

from vllm_omni.entrypoints.stage_utils import shm_read_bytes, shm_write_bytes

from ..utils.logging import get_connector_logger
from .base import OmniConnectorBase

logger = get_connector_logger(__name__)


@dataclass
class _PendingSegment:
    identity: tuple[int, int]
    lock_identity: tuple[int, int]
    cleanup_attempts: int = 0
    retry_at: float | None = None


def _identity(path: str) -> tuple[int, int] | None:
    try:
        stat = os.stat(path)
        return stat.st_dev, stat.st_ino
    except FileNotFoundError:
        return None


def _lock_identity(lockf) -> tuple[int, int]:
    stat = os.fstat(lockf.fileno())
    return stat.st_dev, stat.st_ino


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
        self._pending_keys: OrderedDict[str, _PendingSegment] = OrderedDict()
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
            for _ in range(3):
                with open(lock_file, "wb+") as lockf:
                    fcntl.flock(lockf, fcntl.LOCK_EX)
                    lock_identity = _lock_identity(lockf)
                    # A consumer may have removed this lock while we waited.
                    # Reopen its current pathname before writing a new generation.
                    if _identity(lock_file) != lock_identity:
                        continue
                    meta = shm_write_bytes(payload, name=put_key)
                    identity = _identity(f"/dev/shm/{put_key}")
                    if identity is None:
                        raise FileNotFoundError(put_key)
                    with self._pending_keys_lock:
                        self._pending_keys[put_key] = _PendingSegment(identity, lock_identity)
                    break
            else:
                raise RuntimeError("SHM lock changed repeatedly during put")

            # meta contains {'name': ..., 'size': ...}
            metadata = {"shm": meta, "size": size}
            self._metrics["puts"] += 1
            self._metrics["bytes_transferred"] += size

            return True, size, metadata

        except Exception as e:
            logger.error(f"SharedMemoryConnector put failed for req {put_key}: {e}")
            return False, 0, None

    def _get_data_with_lock(self, lock_file: str, shm_handle: dict[str, Any]) -> tuple[Any, int] | None:
        try:
            with open(lock_file, "rb+") as lockf:
                fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if _identity(lock_file) != _lock_identity(lockf):
                    return None
                key = shm_handle["name"]
                stat = os.stat(f"/dev/shm/{key}")
                identity = (stat.st_dev, stat.st_ino)
                data_bytes = shm_read_bytes({"name": key, "size": stat.st_size})
                try:
                    if _identity(f"/dev/shm/{key}") is None:
                        # Remove the lock before releasing it. Waiters must reject
                        # this old fd rather than write through a detached lock.
                        os.remove(lock_file)
                        with self._pending_keys_lock:
                            owner = self._pending_keys.get(key)
                            if owner is not None and owner.identity == identity:
                                self._pending_keys.pop(key)
                except OSError as e:
                    # Reading succeeded. An uncertain observation or failed lock
                    # removal must retain ownership without discarding the payload.
                    logger.debug("get: failed to finalize consumed SHM %s: %s", key, e)
            obj = self.deserialize_obj(data_bytes)
            result = (obj, stat.st_size)
            return result
        except BlockingIOError:
            return None
        except Exception as e:
            logger.error(f"SharedMemoryConnector shm get failed for req : {e}")
            return None

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
            else:
                # Missing or non-SHM metadata falls back to key-based lookup.
                result = self._get_by_key(get_key)
        else:
            result = self._get_by_key(get_key)

        if result is not None:
            self._metrics["gets"] += 1
        return result

    def cleanup(self, request_id: str) -> None:
        """Unlink the exact key passed to ``put()``, never a request-id prefix."""
        self._cleanup(request_id)

    def _cleanup(self, key: str, *, force: bool = False) -> None:
        with self._pending_keys_lock:
            owner = self._pending_keys.get(key)
            if owner is not None:
                self._cleanup_locked(key, owner, force=force)

    def _cleanup_locked(self, key: str, owner: _PendingSegment, *, force: bool = False) -> None:
        """Called under the ownership mutex; file locks are always nonblocking."""
        if not force and owner.retry_at is not None:
            if owner.cleanup_attempts >= 3 or time.monotonic() < owner.retry_at:
                return
        owner.cleanup_attempts += 1
        lock_file = f"/dev/shm/shm_{key}_lockfile.lock"
        try:
            identity = _identity(f"/dev/shm/{key}")
            if identity is not None and identity != owner.identity:
                self._pending_keys.pop(key)
                return
            if identity is None and _identity(lock_file) is None:
                self._pending_keys.pop(key)
                return
            with open(lock_file, "rb+") as lockf:
                fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if _lock_identity(lockf) != owner.lock_identity or _identity(lock_file) != owner.lock_identity:
                    raise BlockingIOError("SHM lock generation changed")
                # Identity check AND unlink share the writer/consumer lock.
                # A stat check alone would allow deleting a replacement segment.
                identity = _identity(f"/dev/shm/{key}")
                if identity is not None and identity != owner.identity:
                    self._pending_keys.pop(key)
                    return
                if identity is not None:
                    seg = shm_pkg.SharedMemory(name=key)
                    seg.close()
                    seg.unlink()
                try:
                    os.remove(lock_file)
                except FileNotFoundError:
                    pass
                self._pending_keys.pop(key)
        except OSError as e:
            # Reuse the adapter's periodic reap_consumed maintenance. Duplicate
            # cleanup calls share this budget; failed ownership is never dropped.
            owner.retry_at = time.monotonic() + 0.1 * 2 ** min(owner.cleanup_attempts - 1, 1)
            if owner.cleanup_attempts == 3:
                logger.warning("SHM cleanup retry budget exhausted for %s; retained for close: %s", key, e)
            else:
                logger.debug("cleanup: failed to unlink SHM segment %s: %s", key, e)

    def close(self) -> None:
        """One immediate attempt per owned segment, including exhausted retries."""
        with self._pending_keys_lock:
            keys = list(self._pending_keys)
        for key in keys:
            self._cleanup(key, force=True)

    def reap_consumed(self) -> None:
        """Bounded sweep of consumed keys and due cleanup retries.

        The adapter's idle save loop calls this every 100 ms. Untouched terminal
        payloads have no retry_at and stay readable until consumed or closed.
        """
        with self._pending_keys_lock:
            for _ in range(min(64, len(self._pending_keys))):
                key, owner = self._pending_keys.popitem(last=False)
                try:
                    identity = _identity(f"/dev/shm/{key}")
                    if identity != owner.identity:
                        if (
                            identity is not None
                            or _identity(f"/dev/shm/shm_{key}_lockfile.lock") != owner.lock_identity
                        ):
                            continue
                        if owner.retry_at is None:
                            # SHM was consumed, but its own lock remains. Only
                            # this confirmed remnant may start cleanup here.
                            owner.retry_at = time.monotonic()
                except OSError:
                    # A failed observation cannot prove ownership ended.
                    pass
                self._pending_keys[key] = owner
                if owner.retry_at is not None:
                    self._cleanup_locked(key, owner)

    def health(self) -> dict[str, Any]:
        return {"status": "healthy", **self._metrics}
