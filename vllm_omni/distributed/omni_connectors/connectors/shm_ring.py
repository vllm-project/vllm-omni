# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bounded host frames shared by local stage processes.

Each producer/destination edge owns one mmap, with a lock on that same file.
Readers copy before marking a frame consumed, and all readers participate in
the same lock. Pressure returns False to the existing per-key SHM path; this
fast path does not implement end-to-end stream admission or GPU ownership.
"""

from __future__ import annotations

import fcntl
import glob
import hashlib
import mmap
import os
import select
import struct
import threading
import time
import uuid
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from multiprocessing import shared_memory as shm_pkg

from ..utils.logging import get_connector_logger
from ..utils.tensor_frame import TensorFrame, read_tensor_frame

logger = get_connector_logger(__name__)

_HEADER_SIZE = 64
_HEADER = struct.Struct("<8sQQQQQ")  # magic, capacity, head, tail, creator pid, process start
_MAGIC = b"OMSHMR01"
_FRAME = struct.Struct("<IIBBHI")  # total, payload size, kind, state, key length, reserved
_ALIGN = 16
_READY, _DONE = 0, 1
_BYTES, _TENSORS, _PADDING = 0, 1, 2
_STATE_OFFSET = 9
_DEFAULT_BYTES = 4 << 20


def _align(size: int) -> int:
    return (size + _ALIGN - 1) & ~(_ALIGN - 1)


def _process_start(pid: int) -> int:
    with open(f"/proc/{pid}/stat") as file:
        return int(file.read().rsplit(")", 1)[1].split()[19])


@contextmanager
def _registry_update(directory: str) -> Iterator[int]:
    """Publish a strictly changing discovery stamp at channel birth/retirement.

    Kernel directory mtimes can coincide within a clock tick. Serialize marker
    updates and preserve the previous stamp before the filesystem changes it.
    Receivers can then cache discovery without a timer or a per-message scan.
    """
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        previous = os.fstat(fd).st_mtime_ns
        yield fd
        stamp = max(previous + 1, time.time_ns())
        os.utime(fd, ns=(stamp, stamp))
    finally:
        os.close(fd)


@dataclass
class _Channel:
    name: str
    fd: int
    mapping: mmap.mmap
    capacity: int
    owner: bool
    creator_pid: int
    creator_start: int
    segment: shm_pkg.SharedMemory | None = None
    registry: str | None = None
    pidfd: int | None = None
    cursor: int = 0
    index: dict[str, int] = field(default_factory=dict)
    positions: deque[tuple[int, str]] = field(default_factory=deque)

    @classmethod
    def open(cls, name: str, capacity: int | None = None) -> _Channel:
        owner = capacity is not None
        segment = None
        mapping = None
        pidfd = None
        fd = -1
        try:
            if capacity is not None:
                # Only the producer registers the allocation with Python's
                # resource tracker. Readers map the existing file directly.
                segment = shm_pkg.SharedMemory(name=name, create=True, size=_HEADER_SIZE + capacity)
            fd = os.open(f"/dev/shm/{name}", os.O_RDWR | os.O_NOFOLLOW)
            fcntl.flock(fd, fcntl.LOCK_EX | (0 if owner else fcntl.LOCK_NB))
            try:
                if capacity is not None:
                    os.posix_fallocate(fd, 0, _HEADER_SIZE + capacity)
                size = os.fstat(fd).st_size
                if size < _HEADER_SIZE:
                    raise ValueError("incomplete SHM ring header")
                mapping = mmap.mmap(fd, size)
                if capacity is not None:
                    # Fault once per edge, outside the per-message path.
                    mapping[:] = bytes(size)
                    _HEADER.pack_into(mapping, 0, _MAGIC, capacity, 0, 0, os.getpid(), _process_start(os.getpid()))
                magic, actual_capacity, head, tail, creator_pid, creator_start = _HEADER.unpack_from(mapping)
                if magic != _MAGIC or actual_capacity != size - _HEADER_SIZE or not 0 <= tail <= head:
                    raise ValueError("invalid SHM ring header")
                if not owner:
                    try:
                        pidfd = os.pidfd_open(creator_pid)
                    except (AttributeError, OSError):
                        # Older Linux kernels use the process-start check on receive.
                        pidfd = None
                    if _process_start(creator_pid) != creator_start:
                        raise FileNotFoundError("SHM ring producer was replaced")
                return cls(
                    name, fd, mapping, actual_capacity, owner, creator_pid, creator_start, segment=segment, pidfd=pidfd
                )
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
        except BaseException:
            if mapping is not None:
                mapping.close()
            if fd >= 0:
                os.close(fd)
            if segment is not None:
                segment.close()
                segment.unlink()
            if pidfd is not None:
                os.close(pidfd)
            raise

    @contextmanager
    def locked(self, *, blocking: bool = True) -> Iterator[None]:
        fcntl.flock(self.fd, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        try:
            yield
        finally:
            fcntl.flock(self.fd, fcntl.LOCK_UN)

    def bounds(self) -> tuple[int, int]:
        _magic, _capacity, head, tail, _pid, _start = _HEADER.unpack_from(self.mapping)
        return head, tail

    def producer_alive(self) -> bool:
        if self.pidfd is not None:
            return not select.select([self.pidfd], [], [], 0)[0]
        try:
            return _process_start(self.creator_pid) == self.creator_start
        except FileNotFoundError:
            return False

    def frame(self, position: int) -> tuple[int, int, int, int, int, int]:
        frame = _FRAME.unpack_from(self.mapping, _HEADER_SIZE + position % self.capacity)
        total, length, kind, _state, key_length, _reserved = frame
        if total < _FRAME.size or total % _ALIGN or position % self.capacity + total > self.capacity:
            raise ValueError("invalid SHM ring frame size")
        if kind != _PADDING and _FRAME.size + length + key_length > total:
            raise ValueError("invalid SHM ring payload bounds")
        return frame

    def scan(self) -> None:
        head, tail = self.bounds()
        self.cursor = max(self.cursor, tail)
        while self.cursor < head:
            position = self.cursor
            total, _length, kind, state, key_length, _reserved = self.frame(position)
            self.cursor += total
            if kind != _PADDING and state == _READY:
                start = _HEADER_SIZE + position % self.capacity + _FRAME.size
                key = self.mapping[start : start + key_length].decode()
                self.index[key] = position
                self.positions.append((position, key))
        # Remove positions retired by another reader or overwritten after wrap.
        while self.positions and self.positions[0][0] < tail:
            position, key = self.positions.popleft()
            if self.index.get(key) == position:
                del self.index[key]

    def consume(self, position: int) -> None:
        self.mapping[_HEADER_SIZE + position % self.capacity + _STATE_OFFSET] = _DONE

    def reclaim(self) -> None:
        head, tail = self.bounds()
        while tail < head:
            total, _length, kind, state, _key_length, _reserved = self.frame(tail)
            if kind != _PADDING and state != _DONE:
                break
            tail += total
        struct.pack_into("<Q", self.mapping, 24, tail)

    def cancel(self, key: str | None = None, prefix: str | None = None) -> int:
        self.scan()
        count = 0
        candidates = (
            (((key, self.index[key]),) if key in self.index else ()) if key is not None else tuple(self.index.items())
        )
        for candidate, position in candidates:
            matches = candidate == key or (
                prefix is not None and candidate.startswith(prefix) and candidate[len(prefix) :].isdigit()
            )
            if matches:
                if self.frame(position)[3] == _READY:
                    self.consume(position)
                    count += 1
                del self.index[candidate]
        self.reclaim()
        return count

    def close(self) -> None:
        self.mapping.close()
        os.close(self.fd)
        if self.pidfd is not None:
            os.close(self.pidfd)
        if self.segment is not None:
            self.segment.close()
            try:
                self.segment.unlink()
            except FileNotFoundError:
                pass
        if self.registry is not None:
            try:
                with _registry_update(os.path.dirname(self.registry)) as directory_fd:
                    os.unlink(os.path.basename(self.registry), dir_fd=directory_fd)
            except FileNotFoundError:
                pass
            except OSError as error:
                # The allocation is already retired. A failed discovery stamp
                # must not mask setup failure or prevent the per-key fallback.
                logger.debug("Failed to retire host ring marker %s: %s", self.registry, error)


class HostRingTransport:
    """Private SharedMemoryConnector fast path; no new model-specific API."""

    def __init__(self, scope: str, capacity: int = _DEFAULT_BYTES):
        if capacity < 0 or capacity and not 4096 <= capacity <= 64 << 20:
            raise ValueError("host_ring_bytes must be zero or between 4096 and 67108864")
        self.capacity = _align(capacity)
        scope_id = hashlib.sha256(scope.encode()).hexdigest()[:24]
        self.directory = f"omni_shm_ring_v1_{os.getuid()}_{scope_id}"
        self.prefix = f"{self.directory}_"
        self.producers: dict[tuple[str, str], _Channel] = {}
        self._disabled_edges: set[tuple[str, str]] = set()
        self.readers: dict[str, _Channel] = {}
        self._discovery_versions: dict[tuple[str, str], tuple[int, int, int]] = {}
        self._lock = threading.RLock()
        self._closed = False

    def put(self, source: str, destination: str, key: str, payload: bytes | TensorFrame) -> dict | None:
        """Return an immutable descriptor, or None without consuming ring credit."""
        encoded_key = key.encode()
        length = payload.size if isinstance(payload, TensorFrame) else len(payload)
        total = _align(_FRAME.size + len(encoded_key) + length)
        if not self.capacity or not source.isdigit() or not destination.isdigit():
            return None
        # Keep one exceptional payload from monopolizing a channel.
        if total > self.capacity // 4 or len(encoded_key) > 65535:
            return None
        with self._lock:
            if self._closed:
                raise RuntimeError("host ring is closed")
            if (source, destination) in self._disabled_edges:
                return None
            channel = self.producers.get((source, destination))
            if channel is None:
                name = f"{self.prefix}{source}_{destination}_{os.getpid()}_{uuid.uuid4().hex}"
                try:
                    os.makedirs(f"/dev/shm/{self.directory}", mode=0o700, exist_ok=True)
                    channel = _Channel.open(name, self.capacity)
                    channel.registry = f"/dev/shm/{self.directory}/{name}"
                    # Discovery is confined to this deployment directory. The
                    # marker appears only after the allocation is ready.
                    with _registry_update(os.path.dirname(channel.registry)) as directory_fd:
                        marker = os.open(name, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600, dir_fd=directory_fd)
                        os.close(marker)
                except OSError as error:
                    if channel is not None:
                        channel.close()
                    self._disabled_edges.add((source, destination))
                    logger.warning(
                        "Host ring unavailable on edge %s -> %s; using per-key SHM: %s", source, destination, error
                    )
                    return None
                self.producers[source, destination] = channel
            with channel.locked():
                channel.reclaim()
                head, tail = channel.bounds()
                position = head % channel.capacity
                padding = channel.capacity - position if position + total > channel.capacity else 0
                if padding + total > channel.capacity - (head - tail):
                    return None
                if padding:
                    _FRAME.pack_into(channel.mapping, _HEADER_SIZE + position, padding, 0, _PADDING, _DONE, 0, 0)
                    head += padding
                    position = 0
                start = _HEADER_SIZE + position + _FRAME.size
                channel.mapping[start : start + len(encoded_key)] = encoded_key
                start += len(encoded_key)
                if isinstance(payload, TensorFrame):
                    view = memoryview(channel.mapping)
                    try:
                        payload.write(view, start)
                    finally:
                        view.release()
                else:
                    channel.mapping[start : start + length] = payload
                _FRAME.pack_into(
                    channel.mapping,
                    _HEADER_SIZE + position,
                    total,
                    length,
                    _TENSORS if isinstance(payload, TensorFrame) else _BYTES,
                    _READY,
                    len(encoded_key),
                    0,
                )
                # Retire an earlier version while holding the same lock as all readers.
                channel.cancel(key=key)
                struct.pack_into("<Q", channel.mapping, 16, head + total)
                channel.index[key] = head
                channel.positions.append((head, key))
                channel.cursor = head + total
            return {"name": channel.name, "position": head, "size": length}

    def _reader(self, name: str) -> _Channel:
        if not name.startswith(f"omni_shm_ring_v1_{os.getuid()}_") or not name.replace("_", "").isalnum():
            raise ValueError("invalid local SHM ring name")
        channel = self.readers.get(name)
        if channel is None:
            channel = _Channel.open(name)
            self.readers[name] = channel
        return channel

    def _receive(self, channel: _Channel, key: str, descriptor: dict | None) -> tuple[object, int] | None:
        if not channel.producer_alive():
            self.readers.pop(channel.name, None)
            channel.close()
            self._discovery_versions.clear()
            return None
        # This optimistic check only rejects a miss; successful reads still
        # take the shared lock. An unchanged publication cursor cannot have
        # introduced a new key since this receiver's last locked scan.
        if key not in channel.index and struct.unpack_from("<Q", channel.mapping, 16)[0] == channel.cursor:
            return None
        try:
            fcntl.flock(channel.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return None
        try:
            channel.scan()
            position = channel.index.get(key)
            if position is None or descriptor is not None and position != descriptor["position"]:
                return None
            _total, length, kind, state, key_length, _reserved = channel.frame(position)
            del channel.index[key]
            if state != _READY:
                return None
            start = _HEADER_SIZE + position % channel.capacity + _FRAME.size + key_length
            view = memoryview(channel.mapping)[start : start + length]
            try:
                # No reference to ring storage escapes this locked boundary.
                result = read_tensor_frame(view) if kind == _TENSORS else bytes(view)
            finally:
                view.release()
                channel.consume(position)
            return result, length
        finally:
            fcntl.flock(channel.fd, fcntl.LOCK_UN)

    def get(self, source: str, destination: str, key: str, descriptor: dict | None) -> tuple[object, int] | None:
        with self._lock:
            if self._closed:
                return None
            if descriptor is not None:
                try:
                    result = self._receive(self._reader(descriptor["name"]), key, descriptor)
                    if result is None:
                        self._reap_readers()
                    return result
                except (FileNotFoundError, BlockingIOError):
                    return None
            if not self.capacity or not source.isdigit() or not destination.isdigit():
                return None
            edge_prefix = f"{self.prefix}{source}_{destination}_"
            for name, channel in tuple(self.readers.items()):
                if name.startswith(edge_prefix):
                    result = self._receive(channel, key, None)
                    if result is not None:
                        return result
            directory = f"/dev/shm/{self.directory}"
            try:
                status = os.stat(directory)
            except FileNotFoundError:
                self._discovery_versions.clear()
                self._reap_readers()
                return None
            version = (status.st_dev, status.st_ino, status.st_mtime_ns)
            edge = (source, destination)
            if version == self._discovery_versions.get(edge):
                return None
            self._reap_readers()
            # A new producer advances the stamp before publishing any frames,
            # retaining first-message discovery without scanning every miss.
            discovered_result = None
            discovery_complete = True
            for path in glob.iglob(f"/dev/shm/{self.directory}/{edge_prefix}*"):
                name = os.path.basename(path)
                if name in self.readers:
                    continue
                try:
                    channel = self._reader(name)
                except FileNotFoundError:
                    # A published marker whose allocation or producer is gone
                    # cannot become valid again: allocation names are unique.
                    try:
                        with _registry_update(directory) as directory_fd:
                            os.unlink(name, dir_fd=directory_fd)
                    except FileNotFoundError:
                        pass
                    except OSError as error:
                        logger.debug("Failed to prune host ring marker %s: %s", path, error)
                    continue
                except (BlockingIOError, ValueError):
                    # Attachment may race a writer holding the frame lock.
                    # Retry discovery after it releases, even at this stamp.
                    discovery_complete = False
                    continue
                if discovered_result is None:
                    discovered_result = self._receive(channel, key, None)
            # Attach every producer before caching this edge's stamp. Returning
            # as soon as one frame is found would hide its unvisited peers.
            if discovery_complete:
                self._discovery_versions[edge] = version
            else:
                self._discovery_versions.pop(edge, None)
            return discovered_result

    def cancel(self, key: str | None = None, prefix: str | None = None) -> int:
        with self._lock:
            count = 0
            for channel in self.producers.values():
                with channel.locked():
                    count += channel.cancel(key, prefix)
            return count

    def reap(self) -> None:
        with self._lock:
            for channel in self.producers.values():
                with channel.locked():
                    channel.reclaim()
            self._reap_readers()

    def _reap_readers(self) -> None:
        for name, channel in tuple(self.readers.items()):
            if not channel.producer_alive() or not os.path.exists(f"/dev/shm/{name}"):
                channel.close()
                del self.readers[name]

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for channel in self.producers.values():
                with channel.locked():
                    channel.scan()
                    for position in channel.index.values():
                        channel.consume(position)
                channel.close()
            for channel in self.readers.values():
                channel.close()
            self.producers.clear()
            self.readers.clear()
            self._discovery_versions.clear()
            try:
                os.rmdir(f"/dev/shm/{self.directory}")
            except (FileNotFoundError, OSError):
                # Other producers in this deployment may still own rings.
                pass
