# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Device memory monitoring helpers for tests."""

from __future__ import annotations

import threading
import time

from vllm_omni.platforms import current_omni_platform


class DeviceMemoryMonitor:
    """Poll global device memory usage."""

    def __init__(self, device_index: int, interval: float = 0.05):
        self.device_index = device_index
        self.interval = interval
        self._peak_used_mb = 0.0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        def monitor_loop() -> None:
            while not self._stop_event.is_set():
                try:
                    with current_omni_platform.device(self.device_index):
                        free_bytes, total_bytes = current_omni_platform.mem_get_info()
                    used_mb = (total_bytes - free_bytes) / (1024**2)
                    self._peak_used_mb = max(self._peak_used_mb, used_mb)
                except Exception:
                    pass
                time.sleep(self.interval)

        self._thread = threading.Thread(target=monitor_loop, daemon=False)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=2.0)

    @property
    def peak_used_mb(self) -> float:
        fallback_alloc = current_omni_platform.max_memory_allocated(device=self.device_index) / (1024**2)
        fallback_reserved = current_omni_platform.max_memory_reserved(device=self.device_index) / (1024**2)
        return max(self._peak_used_mb, fallback_alloc, fallback_reserved)

    def __del__(self):
        self.stop()


__all__ = ["DeviceMemoryMonitor"]
