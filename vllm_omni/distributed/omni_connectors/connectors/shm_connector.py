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
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing import shared_memory as shm_pkg
from typing import Any

from vllm_omni.entrypoints.stage_utils import shm_read_bytes, shm_write_bytes

from ..utils.logging import get_connector_logger
from ..utils.serialization import OmniSerde
from .base import OmniConnectorBase

logger = get_connector_logger(__name__)

_LOCK_PREFIX = "shm_"
_LOCK_SUFFIX = "_lockfile.lock"
_STALE_LOCK_GRACE_SECONDS = 60.0
_BUSY_CLEANUP_RETRY_SECONDS = 0.1
_next_sweep: dict[str, float] = {}
_sweep_gate = threading.Lock()


@dataclass(eq=False, slots=True)
class _Publication:
    segment_identity: object | None
    retry_after: float | None = None


def _reset_sweep_after_fork() -> None:
    global _sweep_gate
    _sweep_gate = threading.Lock()
    _next_sweep.clear()


os.register_at_fork(after_in_child=_reset_sweep_after_fork)


def _current_lock(fd: int, path: str) -> bool:
    try:
        return os.path.samestat(os.fstat(fd), os.stat(path, follow_symlinks=False))
    except FileNotFoundError:
        return False


@contextmanager
def _publication_lock(path: str, *, blocking: bool = True) -> Iterator[int]:
    operation = fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB
    for _ in range(3):
        with open(path, "a+b") as lock:
            fcntl.flock(lock, operation)
            if _current_lock(lock.fileno(), path):
                yield lock.fileno()
                return
    raise BlockingIOError(f"SHM lock kept changing at {path}")


def _unlink_current_lock(fd: int, path: str) -> None:
    try:
        if _current_lock(fd, path):
            os.unlink(path)
    except OSError:
        pass


def _lock_candidate(info: os.stat_result, cutoff: float | None) -> bool:
    return (
        stat.S_ISREG(info.st_mode)
        and info.st_size == 0
        and info.st_uid == os.geteuid()
        and (cutoff is None or info.st_mtime <= cutoff)
    )


def _unlink_if_orphan(fd: int, lock: str, segment: str, cutoff: float | None = None) -> None:
    try:
        if not _lock_candidate(os.fstat(fd), cutoff):
            return
        os.stat(segment, follow_symlinks=False)
    except FileNotFoundError:
        _unlink_current_lock(fd, lock)
    except OSError:
        pass


def _sweep_orphan_locks(directory: str) -> None:
    cutoff = time.time() - _STALE_LOCK_GRACE_SECONDS
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                if not (entry.name.startswith(_LOCK_PREFIX) and entry.name.endswith(_LOCK_SUFFIX)):
                    continue
                name = entry.name[len(_LOCK_PREFIX) : -len(_LOCK_SUFFIX)]
                segment = os.path.join(directory, name)
                try:
                    if not _lock_candidate(entry.stat(follow_symlinks=False), cutoff) or os.path.lexists(segment):
                        continue
                    fd = os.open(entry.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        _unlink_if_orphan(fd, entry.path, segment, cutoff)
                    finally:
                        os.close(fd)
                except OSError:
                    continue
    except OSError:
        pass


def _wakeup_enabled() -> bool:
    return os.environ.get("VLLM_OMNI_SHM_WAKEUP", "1") == "1"


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

    shm_directory = "/dev/shm"

    def _shm_name(self, key: str) -> str:
        """Map a logical key to an OS segment name."""
        return key

    def _shm_exists(self, key: str) -> bool:
        """Return True while a published payload remains unread.

        Override when segments are not files under ``shm_directory``.
        """
        return os.path.exists(f"{self.shm_directory}/{self._shm_name(key)}")

    def _segment_identity(self, name: str) -> object | None:
        """Return an identity that changes on replacement, or raise FileNotFoundError if absent.

        Backends with separate storage may override this hook. None disables waiting for a
        busy lock because this backend cannot distinguish a replacement from the original.
        """
        if type(self)._shm_exists is not SharedMemoryConnector._shm_exists:
            return None
        info = os.stat(f"{self.shm_directory}/{name}", follow_symlinks=False)
        return info.st_dev, info.st_ino

    def _wakeup_directory(self) -> str:
        # Stage engine processes of one deployment share their launching parent.
        return f"{self.shm_directory}/omni_shm_wake_{os.getuid()}_{os.getppid()}"

    def _lock_file(self, name: str) -> str:
        return f"{self.shm_directory}/{_LOCK_PREFIX}{name}{_LOCK_SUFFIX}"

    def _write_bytes(self, payload: bytes | bytearray | memoryview, name: str) -> dict[str, Any]:
        """Publish payload bytes while the caller holds the segment lock."""
        return shm_write_bytes(payload, name=name)

    def _read_bytes(self, metadata: dict[str, Any]) -> bytes:
        """Read logical payload bytes and unlink under the segment lock."""
        return shm_read_bytes(metadata, require_unlink=True)

    def _unlink_segment(self, name: str) -> None:
        """Unlink a segment by its OS name; raise FileNotFoundError if absent."""
        seg = shm_pkg.SharedMemory(name=name)
        seg.close()
        seg.unlink()

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.stage_id = config.get("stage_id", -1)
        # load_omni_transfer_config puts the scope in the edge's extra, which
        # the factory passes as this config.
        scope = config.get("wakeup_scope")
        self._wake_directory = (
            f"{self.shm_directory}/omni_shm_wake_{os.getuid()}_{hashlib.sha256(str(scope).encode()).hexdigest()[:24]}"
            if scope is not None
            else self._wakeup_directory()
        )
        self._pending_keys: OrderedDict[str, _Publication] = OrderedDict()
        self._pending_keys_lock = threading.Lock()
        self._serde = OmniSerde()
        self._encode_buffer = threading.local()
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
        self._visibility_path: str | None = None
        self._publish_visibility()
        self._orphan_lock_directory = (
            self.shm_directory
            if type(self)._shm_exists is SharedMemoryConnector._shm_exists
            and type(self)._lock_file is SharedMemoryConnector._lock_file
            else None
        )
        self._maybe_sweep_stale_lock_files()

    def _maybe_sweep_stale_lock_files(self) -> None:
        directory = self._orphan_lock_directory
        if directory is None:
            return
        now = time.monotonic()
        if now < _next_sweep.get(directory, 0.0):
            return
        with _sweep_gate:
            if now < _next_sweep.get(directory, 0.0):
                return
            _next_sweep[directory] = now + _STALE_LOCK_GRACE_SECONDS
        _sweep_orphan_locks(directory)

    def _visibility_directory(self) -> str:
        # Key by deployment, as the wakeup directory does: an unrelated
        # deployment under the same UID may reuse these stage ids in this
        # namespace, and its markers must not count as our peer.
        scope = self.config.get("wakeup_scope")
        key = str(scope) if scope is not None else f"ppid_{os.getppid()}"
        digest = hashlib.sha256(key.encode()).hexdigest()[:24]
        return f"{self.shm_directory}/omni_shm_visible_{os.getuid()}_{digest}"

    def _publish_visibility(self) -> None:
        """Leave a file in this process's IPC namespace so peers can see us.

        A connector in another container has its own ``/dev/shm``. Its ``put``
        still succeeds locally, and the receiver polls an empty namespace with
        no error. Publishing here lets the sender refuse that transfer.
        """
        try:
            stage = int(self.stage_id)
        except (TypeError, ValueError):
            return
        if stage < 0:
            return
        try:
            directory = self._visibility_directory()
            os.makedirs(directory, mode=0o700, exist_ok=True)
            # One file per connector. Two connectors for this stage in this
            # process must not share a name: closing one would hide the other.
            path = f"{directory}/{stage}_{os.getpid()}_{uuid.uuid4().hex}"
            fd = os.open(path, os.O_CREAT | os.O_WRONLY, 0o600)
            os.close(fd)
        except OSError as exc:
            logger.error("SharedMemoryConnector could not publish visibility for stage %s: %s", stage, exc)
            return
        self._visibility_path = path

    def _peer_visible(self, to_stage: str) -> bool:
        """True when this connector is not a staged sender, or the peer's file is here."""
        try:
            stage = int(to_stage)
            own = int(self.stage_id)
        except (TypeError, ValueError):
            return True
        if own < 0:
            return True
        try:
            names = os.listdir(self._visibility_directory())
        except FileNotFoundError:
            return False
        prefix = f"{stage}_"
        return any(name.startswith(prefix) for name in names)

    def _drop_visibility(self) -> None:
        path = self._visibility_path
        self._visibility_path = None
        if path is None:
            return
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass

    def _thread_encode_buffer(self) -> bytearray:
        """Return this thread's reusable encode buffer."""
        buffer = getattr(self._encode_buffer, "value", None)
        if buffer is None:
            buffer = bytearray()
            self._encode_buffer.value = buffer
        return buffer

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
            self._maybe_sweep_stale_lock_files()
            if not self._peer_visible(to_stage):
                logger.error(
                    "SharedMemoryConnector cannot see stage %s shared memory; refusing put for %s",
                    to_stage,
                    put_key,
                )
                return False, 0, None
            # Keep per-send housekeeping small when unread chunks accumulate.
            # Explicit sweeps retain a larger budget to drain stale records.
            self.reap_consumed(max_keys=4)
            buffer = self._thread_encode_buffer()
            size = self._serde.serialize_into(data, buffer)
            payload = memoryview(buffer)[:size]

            name = self._shm_name(put_key)
            lock_file = self._lock_file(name)
            with _publication_lock(lock_file) as fd:
                try:
                    meta = self._write_bytes(payload, name)
                except BaseException:
                    if self._orphan_lock_directory is not None:
                        _unlink_if_orphan(fd, lock_file, f"{self.shm_directory}/{name}")
                    raise
                try:
                    identity = self._segment_identity(name)
                except OSError:
                    identity = None
                with self._pending_keys_lock:
                    self._pending_keys[put_key] = _Publication(identity)

            # meta contains {'name': ..., 'size': ...}
            metadata = {"shm": meta, "size": size}
            self._wake_receiver(to_stage)

            self._metrics["puts"] += 1
            self._metrics["bytes_transferred"] += size

            return True, size, metadata

        except Exception as e:
            logger.error(f"SharedMemoryConnector put failed for req {put_key}: {e}")
            return False, 0, None

    def _get_data_with_lock(
        self, get_key: str, shm_handle: dict[str, Any], *, by_key: bool, deadline: float | None
    ) -> tuple[Any, int] | None:
        lock_file = self._lock_file(shm_handle["name"])
        try:
            with open(lock_file, "rb+") as lockf:
                fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fd = lockf.fileno()
                if not _current_lock(fd, lock_file):
                    return None
                if deadline is not None and time.monotonic() >= deadline:
                    return None
                try:
                    data_bytes = self._read_bytes(shm_handle)
                except BaseException:
                    if self._orphan_lock_directory is not None:
                        cutoff = time.time() - _STALE_LOCK_GRACE_SECONDS if by_key else None
                        _unlink_if_orphan(fd, lock_file, f"{self.shm_directory}/{shm_handle['name']}", cutoff)
                    raise
                if by_key or shm_handle["name"] == self._shm_name(get_key):
                    with self._pending_keys_lock:
                        self._pending_keys.pop(get_key, None)
                _unlink_current_lock(fd, lock_file)
            obj = self._serde.deserialize(data_bytes)
            result = (obj, len(data_bytes))
            return result
        except (BlockingIOError, FileNotFoundError):
            return None
        except Exception as e:
            logger.error(f"SharedMemoryConnector shm get failed for req : {e}")
            return None

    def _get_payload(self, get_key: str, metadata, *, deadline: float | None) -> tuple[Any, int] | None:
        if isinstance(metadata, dict) and get_key in metadata:
            metadata = metadata[get_key]
        by_key = not (isinstance(metadata, dict) and "shm" in metadata)
        shm_handle = {"name": self._shm_name(get_key)} if by_key else metadata["shm"]
        result = self._get_data_with_lock(get_key, shm_handle, by_key=by_key, deadline=deadline)
        if result is not None:
            self._metrics["gets"] += 1
        return result

    def get(
        self,
        from_stage: str,
        to_stage: str,
        get_key: str,
        metadata=None,
    ) -> tuple[Any, int] | None:
        self._maybe_sweep_stale_lock_files()
        return self._get_payload(get_key, metadata, deadline=None)

    def get_with_deadline(self, from_stage, to_stage, get_key, metadata=None, *, deadline):
        """Limit this receive to its own key; a directory sweep has no time bound."""
        if time.monotonic() >= deadline:
            return None
        return self._get_payload(get_key, metadata, deadline=deadline)

    def cleanup(self, request_id: str) -> bool:
        """Try to unlink the exact key without waiting. Return True if its segment was removed."""
        return self._cleanup_key(request_id, None)

    def cleanup_tracked(self, key: str, *, defer_if_busy: bool = False) -> bool:
        """Release only the publication this sender has recorded for an aborted transfer."""
        with self._pending_keys_lock:
            publication = self._pending_keys.get(key)
        return publication is not None and self._cleanup_key(key, publication, defer_if_busy=defer_if_busy)

    def _cleanup_key(self, key: str, publication: _Publication | None, *, defer_if_busy: bool = False) -> bool:
        name = self._shm_name(key)
        lock_file = self._lock_file(name)
        expected = publication.segment_identity if publication is not None else None
        unlinked = False
        try:
            with _publication_lock(lock_file, blocking=expected is not None and not defer_if_busy) as fd:
                if publication is not None:
                    with self._pending_keys_lock:
                        still_tracked = self._pending_keys.get(key) is publication
                        publication.retry_after = None
                    if not still_tracked:
                        if self._orphan_lock_directory is not None:
                            _unlink_if_orphan(fd, lock_file, f"{self.shm_directory}/{name}")
                        return False
                try:
                    if expected is not None:
                        current = self._segment_identity(name)
                        if current is None:
                            return False
                        if current != expected:
                            with self._pending_keys_lock:
                                self._pending_keys.pop(key, None)
                            return False
                    self._unlink_segment(name)
                    unlinked = True
                except FileNotFoundError:
                    pass
                except Exception as e:
                    logger.debug("cleanup: could not reclaim SHM segment %s: %s", key, e)
                    return False
                with self._pending_keys_lock:
                    self._pending_keys.pop(key, None)
                _unlink_current_lock(fd, lock_file)
        except OSError as e:
            if defer_if_busy and publication is not None and expected is not None:
                with self._pending_keys_lock:
                    publication.retry_after = (
                        time.monotonic() + _BUSY_CLEANUP_RETRY_SECONDS if isinstance(e, BlockingIOError) else None
                    )
            logger.debug("cleanup: could not acquire SHM lock for %s: %s", key, e)
        return unlinked

    def cleanup_prefix(self, key_prefix: str, *, defer_if_busy: bool = False) -> int:
        """Release tracked publications whose suffix after the prefix is a numeric chunk ID."""
        with self._pending_keys_lock:
            matches = [
                (key, publication)
                for key, publication in self._pending_keys.items()
                if key.startswith(key_prefix) and key[len(key_prefix) :].isdigit()
            ]
        return sum(self._cleanup_key(key, publication, defer_if_busy=defer_if_busy) for key, publication in matches)

    def close(self) -> None:
        """Attempt to reclaim remaining publications and close this connector's local resources."""
        with self._pending_keys_lock:
            remaining = list(self._pending_keys.items())
        for key, publication in remaining:
            self._cleanup_key(key, publication)
        self._drop_visibility()
        self._close_wakeups()

    def reap_consumed(self, *, max_keys: int = 64) -> None:
        """Retire consumed publications and retry terminal cleanup within the same scan budget."""
        retries = []
        now = time.monotonic()
        with self._pending_keys_lock:
            for _ in range(min(max_keys, len(self._pending_keys))):
                key, publication = self._pending_keys.popitem(last=False)
                if publication.retry_after is None:
                    if not self._shm_exists(key):
                        continue
                elif publication.retry_after <= now:
                    publication.retry_after = now + _BUSY_CLEANUP_RETRY_SECONDS
                    retries.append((key, publication))
                self._pending_keys[key] = publication
        for key, publication in retries:
            self._cleanup_key(key, publication, defer_if_busy=True)

    def health(self) -> dict[str, Any]:
        return {"status": "healthy", **self._metrics}
