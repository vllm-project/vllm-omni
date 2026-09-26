# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import fcntl
import glob
import os
import select
import stat
import threading
import time
import uuid
from collections import OrderedDict
from multiprocessing import shared_memory as shm_pkg
from typing import Any

from vllm_omni.entrypoints.stage_utils import shm_read_bytes, shm_write_bytes

from ..utils.logging import get_connector_logger
from .base import OmniConnectorBase

logger = get_connector_logger(__name__)


def _wakeup_enabled() -> bool:
    return os.environ.get("VLLM_OMNI_SHM_WAKEUP", "1") == "1"


def _wakeup_path(to_stage: Any) -> str:
    # Stage engine processes of one deployment share their launching parent.
    return f"/dev/shm/omni_shm_wake_{os.getuid()}_{os.getppid()}_{int(to_stage)}"


class SharedMemoryConnector(OmniConnectorBase):
    """Key-addressed local shared-memory connector.

    SHM is a local-only transport: it reads/writes POSIX shared memory
    segments identified purely by *key*.  It does **not** understand
    remote-transport metadata such as ``source_host`` / ``source_port``
    (that is the RDMA connector's job).  When such metadata is passed in,
    the connector silently falls back to key-based lookup.

    Arrival wakeups: each receiving connector owns a unique named FIFO. A
    ``put`` broadcasts one byte to every FIFO for the destination stage, so the receive loop blocks in
    ``wait_for_change`` until data arrives instead of re-polling on a fixed
    interval. Wakeups are hints: a stage that cannot reach the FIFO (another
    deployment layout, or ``VLLM_OMNI_SHM_WAKEUP=0``) keeps the timed poll.
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
        # Receiver side: FIFO read end (plus a write end of our own, so the
        # FIFO never reports EOF when no sender has it open) and a counter of
        # drained wakeups. Each receiver owns its path, including across restarts.
        self._wake_lock = threading.Lock()
        self._wake_read_fd: int | None = None
        self._wake_hold_fd: int | None = None
        self._wake_path: str | None = None
        self._wake_generation = 0
        self._wake_closed = False

    def _open_wakeup_receiver(self) -> bool:
        if self._wake_closed:
            return False
        if self._wake_read_fd is not None:
            return True
        if not _wakeup_enabled():
            return False
        try:
            if int(self.stage_id) < 0:
                return False
            path = f"{_wakeup_path(self.stage_id)}_{uuid.uuid4().hex}"
        except (TypeError, ValueError):
            return False
        read_fd = hold_fd = None
        created = False
        try:
            os.mkfifo(path, 0o600)
            created = True
            read_fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
            hold_fd = os.open(path, os.O_WRONLY | os.O_NONBLOCK)
        except OSError as e:
            for fd in (read_fd, hold_fd):
                if fd is not None:
                    os.close(fd)
            if created:
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass
            logger.debug("SHM wakeup FIFO unavailable at %s: %s", path, e)
            return False
        self._wake_read_fd, self._wake_hold_fd, self._wake_path = read_fd, hold_fd, path
        return True

    def get_wakeup_generation(self) -> int | None:
        """Snapshot before polling; a later arrival changes it (see ``wait_for_change``).

        ``None`` when this stage has no wakeup FIFO: the caller keeps its timed poll.
        """
        with self._wake_lock:
            return self._wake_generation if self._open_wakeup_receiver() else None

    def wait_for_change(self, generation: int, *, timeout: float) -> bool:
        """Block until a ``put`` to this stage since ``generation``, or ``timeout``."""
        with self._wake_lock:
            read_fd = self._wake_read_fd
            if self._wake_closed or read_fd is None:
                return False
        if self._wake_generation == generation:
            try:
                readable, _, _ = select.select([read_fd], [], [], timeout)
            except (OSError, ValueError):
                return False
            if not readable:
                return False
            with self._wake_lock:
                if self._wake_closed or self._wake_read_fd != read_fd:
                    return False
                try:
                    while os.read(read_fd, 4096):
                        pass
                except BlockingIOError:
                    pass
                except OSError:
                    return False
                self._wake_generation += 1
        return self._wake_generation != generation

    def _wake_receiver(self, to_stage: Any) -> None:
        if not _wakeup_enabled():
            return
        try:
            pattern = f"{_wakeup_path(to_stage)}_*"
        except (TypeError, ValueError):
            return  # non-numeric stage names use the ordinary polling path
        with self._wake_lock:
            if self._wake_closed:
                return
            # Discover every replica, including newly restarted receivers. Do
            # not cache writers across receiver lifetimes or unlink peers' paths.
            for path in glob.iglob(pattern):
                fd = None
                try:
                    fd = os.open(path, os.O_WRONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
                    if stat.S_ISFIFO(os.fstat(fd).st_mode):
                        os.write(fd, b"\0")
                except OSError:
                    # Missing reader/file or full pipe: timed polling remains
                    # the correctness fallback. A full FIFO already has hints.
                    pass
                finally:
                    if fd is not None:
                        os.close(fd)

    def _close_wakeups(self) -> None:
        with self._wake_lock:
            self._wake_closed = True
            for fd in (self._wake_read_fd, self._wake_hold_fd):
                if fd is not None:
                    os.close(fd)
            self._wake_read_fd = self._wake_hold_fd = None
            if self._wake_path is not None:
                try:
                    os.unlink(self._wake_path)
                except FileNotFoundError:
                    pass
                self._wake_path = None

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
            self._wake_receiver(to_stage)

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

    def cleanup(self, request_id: str) -> None:
        """Unlink the exact key passed to ``put()``, never a request-id prefix."""
        key = request_id
        with self._pending_keys_lock:
            self._pending_keys.pop(key, None)
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
        with self._pending_keys_lock:
            keys = list(self._pending_keys)
        for key in keys:
            self.cleanup(key)
        self._close_wakeups()

    def reap_consumed(self) -> None:
        """Bounded round-robin sweep; receivers unlink SHM in another process."""
        with self._pending_keys_lock:
            for _ in range(min(64, len(self._pending_keys))):
                key, _ = self._pending_keys.popitem(last=False)
                if os.path.exists(f"/dev/shm/{key}"):
                    self._pending_keys[key] = None

    def health(self) -> dict[str, Any]:
        return {"status": "healthy", **self._metrics}
