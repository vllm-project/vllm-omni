# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Owner-thread scheduling with an ordered GPU completion notification."""

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any


class GenerationCompletionObserver:
    """Observe one oldest output; never consume it or update scheduler state."""

    def __init__(self, wakeup: Any) -> None:
        self._wakeup = wakeup
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="omni-generation-completion")
        self._observed: Any = None
        self._completion: Any = None

    def _wait(self, event: Any, completed: Future) -> None:
        try:
            device = getattr(event, "device", None)
            if device is not None:
                from vllm.platforms import current_platform

                current_platform.set_device(device)
            event.synchronize()
        except BaseException as error:
            completed.set_exception(error)
        else:
            completed.set_result(None)
        finally:
            # The owner consumes the original output, including any CUDA error.
            self._wakeup()

    def ready(self, future: Any) -> bool:
        if future.done():
            return True
        event = getattr(getattr(future, "async_output", None), "copy_event", None)
        if event is None:
            return True  # Preserve upstream behavior for other executor futures.
        if self._observed is future:
            return self._completion.done()
        if event.query():
            return True
        self._observed = future
        self._completion = Future()
        self._pool.submit(self._wait, event, self._completion)
        return False

    def consumed(self, future: Any) -> None:
        if self._observed is future:
            self._completion.result()
            self._observed = None
            self._completion = None

    def close(self) -> None:
        self._pool.shutdown(wait=True)
        self._observed = None
        self._completion = None
