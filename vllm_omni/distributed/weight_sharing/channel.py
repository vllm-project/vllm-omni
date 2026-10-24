# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bounded CPU-only messages over a private, same-user Unix socket.

The caller authenticates the deployment token and session identity in its
protocol handler. This module checks the operating-system peer identity and
never serializes Python callables, CUDA tensors, or arbitrary Python objects.
"""

from __future__ import annotations

import math
import os
import socket
import stat
import struct
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import msgspec

from .contracts import Message, ProtocolError


class ChannelError(ProtocolError):
    """The metadata connection or its bounded wire protocol failed."""


@dataclass(frozen=True)
class Limits:
    max_bundle_bytes: int = 64 * 1024 * 1024
    timeout: float = 10.0
    max_entries: int = 100_000

    def __post_init__(self) -> None:
        if not 1 <= self.max_bundle_bytes <= (1 << 32) - 1:
            raise ValueError("max_bundle_bytes must be between 1 and 2**32 - 1")
        if self.max_entries < 1:
            raise ValueError("entry limit must be positive")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout must be finite and positive")


_HEADER = struct.Struct("!I")


class FramedConnection:
    """A single-reader, single-writer connection carrying typed CPU messages.

    Each message is a four-byte length followed by its MessagePack payload.
    Each send/receive has one deadline covering the header and payload, including
    partial reads. Receivers may wait indefinitely for the first byte between
    messages; the deadline then starts when that byte arrives. Callers must
    serialize operations; ``close`` may interrupt one from another thread during
    shutdown. Successful send does not prove peer receipt.
    """

    def __init__(self, connection: socket.socket, limits: Limits = Limits()) -> None:
        self._socket = connection
        self.limits = limits
        self._decoder = msgspec.msgpack.Decoder(Message)
        self._closed = False
        self._close_lock = threading.Lock()

    def _set_remaining(self, deadline: float) -> None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ChannelError("Metadata operation exceeded its total timeout")
        self._socket.settimeout(remaining)

    def _read_exact(self, count: int, deadline: float, *, allow_eof: bool = False) -> bytearray:
        data = bytearray()
        while len(data) < count:
            self._set_remaining(deadline)
            chunk = self._socket.recv(count - len(data))
            if not chunk:
                if allow_eof and not data and not self._closed:
                    raise EOFError("Metadata peer closed between messages")
                raise ChannelError("Metadata connection closed during an operation")
            data.extend(chunk)
        return data

    def send(self, message: Message) -> None:
        if type(message) is not Message:
            raise ChannelError("Only declared protocol message types may be sent")
        deadline = time.monotonic() + self.limits.timeout
        try:
            data = msgspec.msgpack.encode(message)
            if len(data) > self.limits.max_bundle_bytes:
                raise ChannelError("Metadata message exceeds its size limit")
            self._set_remaining(deadline)
            self._socket.sendall(_HEADER.pack(len(data)))
            self._set_remaining(deadline)
            self._socket.sendall(data)
        except (OSError, msgspec.EncodeError, TypeError) as error:
            raise ChannelError("Could not send metadata message") from error

    def recv(self, *, wait_for_message: bool = False) -> Message:
        try:
            if wait_for_message:
                self._socket.settimeout(None)
                first = self._socket.recv(1)
                if not first:
                    if not self._closed:
                        raise EOFError("Metadata peer closed between messages")
                    raise ChannelError("Metadata connection closed during an operation")
                deadline = time.monotonic() + self.limits.timeout
                header = first + self._read_exact(_HEADER.size - 1, deadline)
            else:
                deadline = time.monotonic() + self.limits.timeout
                header = bytes(self._read_exact(_HEADER.size, deadline, allow_eof=True))
            size = _HEADER.unpack(header)[0]
            if not 0 < size <= self.limits.max_bundle_bytes:
                raise ChannelError("Metadata message exceeds its size limit")
            message = self._decoder.decode(self._read_exact(size, deadline))
            self._set_remaining(deadline)
            return message
        except (OSError, msgspec.DecodeError) as error:
            raise ChannelError("Could not receive a valid metadata message") from error

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                # A disconnected socket is already unable to perform I/O.
                pass
            finally:
                self._socket.close()

    def __enter__(self) -> FramedConnection:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def _open_private_parent(endpoint: str) -> tuple[Path, int]:
    if sys.platform != "linux" or not hasattr(socket, "SO_PEERCRED"):
        raise ChannelError("Weight sharing Unix endpoints require Linux peer credentials")
    path = Path(endpoint)
    if not path.is_absolute() or not path.name or len(os.fsencode(path)) >= 108:
        raise ChannelError("Endpoint must be an absolute Unix socket path shorter than 108 bytes")
    parent_fd: int | None = None
    try:
        parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        info = os.fstat(parent_fd)
        if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077:
            raise ChannelError("Endpoint parent must be a private directory owned by this user")
        return path, parent_fd
    except (OSError, ChannelError) as error:
        if parent_fd is not None:
            os.close(parent_fd)
        raise ChannelError("Could not open the endpoint private directory") from error


def _check_peer(connection: socket.socket) -> None:
    credentials = connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
    _pid, uid, _gid = struct.unpack("3i", credentials)
    if uid != os.geteuid():
        raise ChannelError("Metadata peer has a different user identity")


def connect(endpoint: str, limits: Limits = Limits()) -> FramedConnection:
    path, parent_fd = _open_private_parent(endpoint)
    connection: socket.socket | None = None
    try:
        info = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077:
            raise ChannelError("Endpoint must be a private socket owned by this user")
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.settimeout(limits.timeout)
        connection.connect(str(path))
        _check_peer(connection)
        return FramedConnection(connection, limits)
    except (OSError, ChannelError) as error:
        if connection is not None:
            connection.close()
        raise ChannelError("Could not connect to the weight-sharing endpoint") from error
    finally:
        os.close(parent_fd)


class SessionServer:
    """One CPU thread accepts and handles connections serially.

    There is no queue of worker threads. The handler owns session/token checks
    for a single preselected consumer, without reconnect retries, and must perform
    CPU work only. Message transfers are bounded by connection deadlines; an idle
    session can wait for a message until ``stop`` closes its connection.
    """

    def __init__(self, endpoint: str, limits: Limits, handler: Callable[[FramedConnection], None]) -> None:
        self.endpoint = endpoint
        self.limits = limits
        self._handler = handler
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._stopping = threading.Event()
        self._thread: threading.Thread | None = None
        self._listener: socket.socket | None = None
        self._connection: FramedConnection | None = None
        self._startup_error: Exception | None = None
        self._last_error: Exception | None = None

    @property
    def last_error(self) -> Exception | None:
        with self._lock:
            return self._last_error

    def start(self) -> None:
        with self._lock:
            if self._thread is not None or self._stopping.is_set():
                raise ChannelError("A session server may only be started once")
            self._thread = threading.Thread(target=self._serve, name="weight-sharing-session", daemon=True)
            self._thread.start()
        if not self._ready.wait(self.limits.timeout):
            self.stop()
            raise ChannelError("Timed out starting the weight-sharing endpoint")
        with self._lock:
            error = self._startup_error
        if error is not None:
            self.stop()
            raise ChannelError("Could not start the weight-sharing endpoint") from error

    def stop(self) -> None:
        self._stopping.set()
        with self._lock:
            listener = self._listener
            connection = self._connection
            thread = self._thread
        if connection is not None:
            connection.close()
        if listener is not None:
            listener.close()
        if thread is not None and thread is not threading.current_thread():
            thread.join(self.limits.timeout)
            if thread.is_alive():
                raise ChannelError("The CPU session handler did not stop before its deadline")

    def _serve(self) -> None:
        listener: socket.socket | None = None
        parent_fd: int | None = None
        path: Path | None = None
        bound_identity: tuple[int, int] | None = None
        try:
            path, parent_fd = _open_private_parent(self.endpoint)
            try:
                os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise ChannelError("Refusing to replace an existing endpoint")
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            listener.bind(str(path))
            info = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
            bound_identity = (info.st_dev, info.st_ino)
            os.chmod(path.name, 0o600, dir_fd=parent_fd)
            listener.listen(1)
            listener.settimeout(min(0.2, self.limits.timeout))
            with self._lock:
                self._listener = listener
            self._ready.set()
            while not self._stopping.is_set():
                try:
                    raw, _address = listener.accept()
                except TimeoutError:
                    continue
                connection = FramedConnection(raw, self.limits)
                with self._lock:
                    self._connection = connection
                try:
                    _check_peer(raw)
                    if not self._stopping.is_set():
                        self._handler(connection)
                except EOFError:
                    pass  # The handler decides whether peer closure ends the session.
                except Exception as error:
                    # Expose handler failures to the owner; never print tokens,
                    # reducer descriptions, or exception text from the peer.
                    with self._lock:
                        if not self._stopping.is_set():
                            self._last_error = error
                finally:
                    connection.close()
                    with self._lock:
                        self._connection = None
        except Exception as error:
            with self._lock:
                if not self._ready.is_set():
                    self._startup_error = error
                if not self._stopping.is_set():
                    self._last_error = error
        finally:
            if listener is not None:
                listener.close()
            if parent_fd is not None:
                try:
                    if path is not None and bound_identity is not None:
                        try:
                            info = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
                        except FileNotFoundError:
                            pass
                        else:
                            if stat.S_ISSOCK(info.st_mode) and (info.st_dev, info.st_ino) == bound_identity:
                                os.unlink(path.name, dir_fd=parent_fd)
                except OSError as error:
                    with self._lock:
                        self._last_error = error
                finally:
                    os.close(parent_fd)
            with self._lock:
                self._listener = None
            self._ready.set()
