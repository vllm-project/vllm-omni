# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Linux SHM publication notifications for the native local receiver.

SharedMemoryConnector closes its write-open lock file *after* publishing the
payload. Watching CLOSE_WRITE avoids opening every outstanding SHM key to
discover arrivals. Notifications are hints: registration/rearm and overflow
must still reconcile against the authoritative key-addressed connector.
"""

from __future__ import annotations

import ctypes
import os
import select
import socket
import struct
import sys

_EVENT = struct.Struct("iIII")
_CLOSE_WRITE = 0x00000008
_OVERFLOW = 0x00004000
_INVALIDATED = 0x00008000 | 0x00002000 | 0x00000400 | 0x00000800


class ShmReadiness:
    _fd: int
    _reader: socket.socket
    _writer: socket.socket

    def __init__(self, directory: str = "/dev/shm", *, cohort_name: str | None = None) -> None:
        namespace = os.getenv("VLLM_OMNI_SHM_COHORT_NAMESPACE", "")
        self._cohort_prefix = cohort_name or (f"omni_cohort_{namespace}_" if namespace else None)
        if sys.platform != "linux":
            raise OSError("SHM publication notifications require Linux inotify")
        libc = ctypes.CDLL(None, use_errno=True)
        libc.inotify_init1.argtypes = [ctypes.c_int]
        libc.inotify_init1.restype = ctypes.c_int
        libc.inotify_add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
        libc.inotify_add_watch.restype = ctypes.c_int
        self._fd = libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
        if self._fd < 0:
            raise OSError(ctypes.get_errno(), "inotify_init1")
        if libc.inotify_add_watch(self._fd, os.fsencode(directory), _CLOSE_WRITE | _INVALIDATED) < 0:
            error = ctypes.get_errno()
            os.close(self._fd)
            raise OSError(error, "inotify_add_watch")
        try:
            self._reader, self._writer = socket.socketpair()
            self._reader.setblocking(False)
            self._writer.setblocking(False)
        except BaseException:
            os.close(self._fd)
            raise

    def wake(self) -> None:
        """Wake on registration, consumption or shutdown; full pipe coalesces."""
        try:
            self._writer.send(b"\0")
        except (BlockingIOError, OSError):
            # A full socket is already readable; close can race with shutdown.
            pass

    @staticmethod
    def _decode(data: bytes, cohort_prefix: str | None = None) -> tuple[set[str], bool]:
        keys: set[str] = set()
        rescan = False
        offset = 0
        while offset < len(data):
            _watch, mask, _cookie, length = _EVENT.unpack_from(data, offset)
            offset += _EVENT.size
            name = os.fsdecode(data[offset : offset + length].split(b"\0", 1)[0])
            offset += length
            if mask & _INVALIDATED:
                raise OSError("SHM publication watch invalidated")
            rescan |= bool(mask & _OVERFLOW)
            if (
                cohort_prefix
                and mask & _CLOSE_WRITE
                and name.endswith(".lock")
                and (name == cohort_prefix if cohort_prefix.endswith(".lock") else name.startswith(cohort_prefix))
            ):
                rescan = True
            if mask & _CLOSE_WRITE and name.startswith("shm_") and name.endswith("_lockfile.lock"):
                keys.add(name[4 : -len("_lockfile.lock")])
        return keys, rescan

    def wait(self, timeout: float = 1.0) -> tuple[set[str], bool]:
        readable, _, _ = select.select([self._fd, self._reader], [], [], timeout)
        keys: set[str] = set()
        # A low-frequency reconciliation also covers filesystem anomalies and
        # a writer that uses another publication mechanism.
        rescan = not readable
        if self._reader in readable:
            try:
                while self._reader.recv(4096):
                    pass
            except BlockingIOError:
                pass
        if self._fd in readable:
            while True:
                try:
                    data = os.read(self._fd, 65536)
                except BlockingIOError:
                    break
                arrived, overflow = self._decode(data, self._cohort_prefix)
                keys.update(arrived)
                rescan |= overflow
        return keys, rescan

    def close(self) -> None:
        os.close(self._fd)
        self._reader.close()
        self._writer.close()
