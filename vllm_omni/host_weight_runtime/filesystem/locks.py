# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stable POSIX advisory locks used by the filesystem store."""

from __future__ import annotations

import errno
import fcntl
import os
import time
from pathlib import Path


class FileLockTimeoutError(TimeoutError):
    pass


class FileLock:
    """One held flock whose inode is stable outside artifact directories."""

    def __init__(self, path: Path, *, exclusive: bool, deadline: float | None, nonblocking: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
        self._creator_pid = os.getpid()
        self._closed = False
        operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        if nonblocking:
            operation |= fcntl.LOCK_NB
        try:
            if nonblocking:
                self._acquire_once(operation)
            elif deadline is None:
                self._acquire_blocking(operation)
            else:
                self._acquire_until(operation | fcntl.LOCK_NB, deadline)
        except BaseException:
            os.close(self._fd)
            self._closed = True
            raise

    def _acquire_once(self, operation: int) -> None:
        try:
            fcntl.flock(self._fd, operation)
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                raise FileLockTimeoutError(f"lock is active: {self.path}") from exc
            raise

    def _acquire_blocking(self, operation: int) -> None:
        while True:
            try:
                fcntl.flock(self._fd, operation)
                return
            except OSError as exc:
                if exc.errno != errno.EINTR:
                    raise

    def _acquire_until(self, operation: int, deadline: float) -> None:
        while True:
            try:
                fcntl.flock(self._fd, operation)
                return
            except OSError as exc:
                if exc.errno == errno.EINTR:
                    continue
                if exc.errno not in (errno.EACCES, errno.EAGAIN):
                    raise
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise FileLockTimeoutError(f"timed out waiting for {self.path}") from exc
                time.sleep(min(0.05, remaining))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if os.getpid() != self._creator_pid:
            # flock state belongs to the inherited open-file description. A
            # child must only drop its descriptor reference; LOCK_UN would
            # release the creator process's active lock as well.
            os.close(self._fd)
            return
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)

    def __enter__(self) -> FileLock:
        return self

    def __exit__(self, exc_type: object, exc: object, _traceback: object) -> None:
        self.close()


def lock_is_active(path: Path) -> bool:
    """Return a point-in-time contention snapshot without creating a lock file."""
    try:
        fd = os.open(path, os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW)
    except FileNotFoundError:
        return False
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                return True
            raise
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


__all__ = ["FileLock", "FileLockTimeoutError", "lock_is_active"]
