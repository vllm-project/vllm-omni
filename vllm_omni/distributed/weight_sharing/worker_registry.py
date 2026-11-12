# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Track consumers before CUDA initialization, including failed stage startup.

The existing diffusion manager and workers share this small CPU array. Sealing
it prevents a late worker from importing after cleanup has enumerated peers.
No monitor process or polling thread is needed.
"""

from __future__ import annotations

import time
from contextlib import contextmanager

import psutil


class ConsumerStartupError(RuntimeError):
    def __init__(self, message: str, *, registry: ConsumerWorkerRegistry):
        super().__init__(message)
        self.registry = registry


class ConsumerWorkerRegistry:
    def __init__(self, ctx, ranks: int):
        # pid + creation time distinguish dead workers from reused OS PIDs.
        self._data = ctx.Array("d", 1 + 2 * ranks)

    @contextmanager
    def _locked(self):
        lock = self._data.get_lock()
        if not lock.acquire(timeout=5):
            raise RuntimeError("consumer registry lock is unavailable; retain provider")
        try:
            yield
        finally:
            lock.release()

    def register(self, rank: int) -> bool:
        process = psutil.Process()
        with self._locked():
            if self._data[0]:
                return False
            index = 1 + 2 * rank
            if rank < 0 or index + 1 >= len(self._data) or self._data[index]:
                raise RuntimeError("duplicate or invalid consumer worker rank")
            self._data[index] = process.pid
            self._data[index + 1] = process.create_time()
            return True

    def terminate(self, timeout: float) -> None:
        """Seal admission, kill registered peers, and confirm their exit."""
        with self._locked():
            self._data[0] = 1
            entries = tuple(zip(self._data[1::2], self._data[2::2], strict=True))
        processes = []
        for pid, created in entries:
            if not pid:
                continue
            try:
                process = psutil.Process(int(pid))
                if process.create_time() != created or process.status() == psutil.STATUS_ZOMBIE:
                    continue
                process.kill()
                processes.append(process)
            except psutil.NoSuchProcess:
                pass
        # Observe exit without reaping: multiprocessing.Process owns waitpid
        # and must collect its child's exit status itself during join().
        deadline = time.monotonic() + timeout
        while processes:
            alive = []
            for process in processes:
                try:
                    if process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                        alive.append(process)
                except psutil.NoSuchProcess:
                    pass
            if not alive:
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("consumer worker exit unconfirmed; retain provider")
            processes = alive
            time.sleep(min(0.05, remaining))
