# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import fcntl
import glob
import hashlib
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


def _wakeup_directory() -> str:
    # Stage engine processes of one deployment share their launching parent.
    return f"/dev/shm/omni_shm_wake_{os.getuid()}_{os.getppid()}"


_LOCK_FILE_PREFIX = "shm_"
_LOCK_FILE_SUFFIX = "_lockfile.lock"
_STALE_LOCK_GRACE_SECONDS = 60.0
# Time-gated orphan sweep: instead of a once-per-process flag, remember when
# the last sweep ran. A process that starts (or calls put/get) within the
# grace window of a crash retries the sweep later, so young orphan lock files
# are not skipped forever.
_last_stale_sweep = float("-inf")
_sweep_gate_lock = threading.Lock()


def _reclaim_orphan_lock_file(key: str, grace: float | None = None) -> bool:
    """Remove *key*'s lock file when its segment is already gone (single key).

    A lock file has no lifecycle of its own — it exists exactly while its
    segment does.  Uses the same check–lock–recheck protocol as
    ``_sweep_stale_lock_files`` so a concurrent creator or consumer never
    loses its lock; see that function for the protocol description.
    """
    if grace is None:
        grace = _STALE_LOCK_GRACE_SECONDS
    lock = f"/dev/shm/{_LOCK_FILE_PREFIX}{key}{_LOCK_FILE_SUFFIX}"
    try:
        if time.time() - os.stat(lock).st_mtime < grace:
            return False
    except OSError:
        return False
    if os.path.exists(f"/dev/shm/{key}"):
        return False
    try:
        fd = os.open(lock, os.O_RDONLY)
    except OSError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return False
    try:
        st_fd = os.fstat(fd)
        st_path = os.stat(lock)  # ENOENT: already removed, goal reached
        if (
            st_fd.st_ino == st_path.st_ino
            and st_fd.st_dev == st_path.st_dev
            and st_fd.st_uid == os.geteuid()
            and time.time() - st_path.st_mtime >= grace
            and not os.path.exists(f"/dev/shm/{key}")
        ):
            os.remove(lock)
            logger.debug("reclaimed orphan SHM lock file %s", lock)
            return True
    except OSError:
        pass
    finally:
        os.close(fd)
    return False


def _sweep_stale_lock_files(grace: float | None = None) -> int:
    """Best-effort removal of orphan lock files whose segment is already gone.

    A lock file only guards one transfer; its segment is unlinked by the
    receiving read or by ``resource_tracker`` when the owner process dies, so
    a lock file whose segment no longer exists is garbage.  Uses a
    check–lock–recheck–unlink protocol so a concurrent creator/reader can
    never lose its lock:

    1. prefilter: lock older than *grace* (mtime; young files may sit in the
       creator's ``open() -> flock()`` window) and segment absent
    2. ``os.open(path, O_RDONLY)`` (read-only: never truncates or refreshes
       mtime) + ``flock(LOCK_EX | LOCK_NB)`` — fails while a critical section
       holds it (``flock`` needs no write access, unlike ``fcntl`` locks)
    3. recheck under the lock: segment still absent, mtime still past grace,
       path still resolves to the locked inode, owned by this uid
    4. ``os.remove`` while still holding the lock, then ``close`` releases it
    """
    if grace is None:
        grace = _STALE_LOCK_GRACE_SECONDS
    removed = 0
    try:
        names = os.listdir("/dev/shm")
    except OSError:
        return 0
    for name in names:
        if not (name.startswith(_LOCK_FILE_PREFIX) and name.endswith(_LOCK_FILE_SUFFIX)):
            continue
        key = name[len(_LOCK_FILE_PREFIX) : -len(_LOCK_FILE_SUFFIX)]
        if key and _reclaim_orphan_lock_file(key, grace=grace):
            removed += 1
    return removed


def _maybe_sweep_stale_lock_files() -> None:
    """Run :func:`_sweep_stale_lock_files` at most once per grace period.

    Called from ``__init__`` / ``put()`` / ``get()``; the hot path pays one
    monotonic-clock comparison under a tiny lock, and a full sweep runs at
    most once per ``_STALE_LOCK_GRACE_SECONDS``.
    """
    global _last_stale_sweep
    with _sweep_gate_lock:
        now = time.monotonic()
        if now - _last_stale_sweep < _STALE_LOCK_GRACE_SECONDS:
            return
        _last_stale_sweep = now
    _sweep_stale_lock_files()


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

    Lock-file lifecycle contract: the zero-byte lock file
    ``/dev/shm/shm_{key}_lockfile.lock`` has no lifecycle of its own — it
    exists exactly while its segment does.  It is removed on the successful
    receiving read, on a read whose segment is already gone, on a key-based
    poll whose segment no longer exists (``_reclaim_orphan_lock_file``), on a
    failed ``put()`` (the segment never came to life), and by ``cleanup()`` /
    ``close()`` for keys still tracked in ``_pending_keys``.  Abnormally
    terminated processes leave lock files behind once their segments are
    reaped by ``resource_tracker``; a time-gated sweep
    (``_maybe_sweep_stale_lock_files``, called from ``__init__`` / ``put()`` /
    ``get()``) reaps such orphans, retrying after the grace window so a
    restart inside the window is not left with young orphans forever.
    ``_pending_keys`` is bookkeeping for ``cleanup()`` / ``close()`` only.
    """

    def get_with_deadline(self, from_stage, to_stage, get_key, metadata=None, *, deadline):
        if time.monotonic() >= deadline:
            return None
        return self.get(from_stage, to_stage, get_key, metadata)

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.stage_id = config.get("stage_id", -1)
        scope = config.get("extra", {}).get("wakeup_scope")
        self._wake_directory = (
            f"/dev/shm/omni_shm_wake_{os.getuid()}_{hashlib.sha256(str(scope).encode()).hexdigest()[:24]}"
            if scope is not None
            else _wakeup_directory()
        )
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
        # Orphan lock files left by crashed predecessors are reaped here; the
        # gate retries after the grace window (via put/get) when this process
        # starts inside it, instead of skipping young orphans forever.
        _maybe_sweep_stale_lock_files()

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
            path = f"{self._wake_directory}/{int(self.stage_id)}_{uuid.uuid4().hex}"
        except (TypeError, ValueError):
            return False
        read_fd = hold_fd = None
        created = False
        try:
            os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
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
            pattern = f"{self._wake_directory}/{int(to_stage)}_*"
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
                try:
                    os.rmdir(self._wake_directory)
                except OSError:
                    # Other receivers may still own FIFOs in this deployment.
                    pass

    def put(
        self,
        from_stage: str,
        to_stage: str,
        put_key: str,
        data: Any,
    ) -> tuple[bool, int, dict[str, Any] | None]:
        try:
            # Keep per-send housekeeping small when unread chunks accumulate.
            # Explicit sweeps retain a larger budget to drain stale records.
            self.reap_consumed(max_keys=4)
            _maybe_sweep_stale_lock_files()
            payload = self.serialize_obj(data)
            size = len(payload)

            lock_file = f"/dev/shm/shm_{put_key}_lockfile.lock"
            with open(lock_file, "wb+") as lockf:
                fcntl.flock(lockf, fcntl.LOCK_EX)
                try:
                    meta = shm_write_bytes(payload, name=put_key)
                except BaseException:
                    # The segment never came to life, so the lock file guarding
                    # it must not either — remove it while we still hold it.
                    try:
                        os.remove(lock_file)
                    except OSError:
                        pass
                    raise
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
            # The lock file has no lifecycle of its own: once the segment is
            # gone (consumed by this read, or already reaped elsewhere) the
            # lock guarding it must go too.  A failed read with the segment
            # still alive keeps the lock — the transfer can be retried.
            seg_gone = not os.path.exists(f"/dev/shm/{shm_handle['name']}")
            if consumed or seg_gone:
                try:
                    os.remove(lock_file)
                except OSError:
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
            # No segment: either the sender has not created it yet, or the
            # owner died and resource_tracker reaped it. In the latter case an
            # orphan lock file may be left behind — reclaim it once it is old
            # enough (the grace window protects an in-flight put()).
            _reclaim_orphan_lock_file(get_key)
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
        _maybe_sweep_stale_lock_files()
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
        self._close_wakeups()

    def reap_consumed(self, *, max_keys: int = 64) -> None:
        """Bounded round-robin sweep; receivers unlink SHM in another process."""
        with self._pending_keys_lock:
            for _ in range(min(max_keys, len(self._pending_keys))):
                key, _ = self._pending_keys.popitem(last=False)
                if os.path.exists(f"/dev/shm/{key}"):
                    self._pending_keys[key] = None

    def health(self) -> dict[str, Any]:
        return {"status": "healthy", **self._metrics}
