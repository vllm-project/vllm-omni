# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared async-output Future lifecycle for multiproc and Ray executors.

Transport differs (MessageQueue vs Ray Queue); waiter/cache/drop bookkeeping
is the same Future contract.
"""

from __future__ import annotations

import concurrent.futures
import threading
from collections import OrderedDict
from typing import TYPE_CHECKING

from vllm_omni.diffusion.utils.future_utils import try_set_exception, try_set_result

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import DiffusionOutput

_DROPPED_OUTPUT_IDS_MAX = 4096


class DropPlaceholder(concurrent.futures.Future):
    """Placeholder registered by ``drop_output``.

    Distinguishes "abort said nobody wants this output" from a genuine
    ``wait_output_ready`` waiter, so pumps can discard tensors for the former
    while still resolving the latter directly.
    """


def dropped_output_error(async_output_id: str) -> RuntimeError:
    return RuntimeError(
        f"async output {async_output_id} was dropped: the request was aborted "
        "before its output was claimed; retry with a new request."
    )


class AsyncOutputFutureMixin:
    """Mixin: ``_finish_output`` / ``wait_output_ready`` / ``drop_output``.

    Subclasses must provide ``_futures_lock``, ``_output_futures``,
    ``_completed_outputs``, ``_dropped_output_ids``, and ``_closed``.
    """

    _futures_lock: threading.RLock
    _output_futures: dict[str, concurrent.futures.Future]
    _completed_outputs: dict[str, concurrent.futures.Future]
    _dropped_output_ids: OrderedDict[str, None]
    _closed: bool

    def _finish_output(
        self,
        async_output_id: str,
        result: DiffusionOutput | None,
        exc: BaseException | None,
    ) -> None:
        """Hand one delivered async output to its waiter. Caller holds ``_futures_lock``."""
        pending = self._output_futures.pop(async_output_id, None)
        if pending is None:
            if async_output_id in self._dropped_output_ids:
                # Timeline-B: wait_output_ready already moved this id into the
                # dropped-id LRU. Late delivery must not repopulate the cache.
                self._dropped_output_ids.move_to_end(async_output_id)
                return
            fut: concurrent.futures.Future = concurrent.futures.Future()
            if exc is not None:
                fut.set_exception(exc)
            else:
                fut.set_result(result)
            self._completed_outputs[async_output_id] = fut
        elif isinstance(pending, DropPlaceholder):
            try_set_exception(pending, dropped_output_error(async_output_id))
            self._remember_dropped(async_output_id)
        elif not pending.done():
            if exc is not None:
                delivered = try_set_exception(pending, exc)
            else:
                delivered = try_set_result(pending, result)
            if not delivered:
                self._remember_dropped(async_output_id)
        else:
            self._remember_dropped(async_output_id)

    def _remember_dropped(self, async_output_id: str) -> None:
        dropped = self._dropped_output_ids
        dropped[async_output_id] = None
        dropped.move_to_end(async_output_id)
        while len(dropped) > _DROPPED_OUTPUT_IDS_MAX:
            dropped.popitem(last=False)

    def wait_output_ready(self, async_output_id: str) -> concurrent.futures.Future:
        """Return a Future that resolves when the async output is ready."""
        with self._futures_lock:
            if self._closed:
                closed: concurrent.futures.Future = concurrent.futures.Future()
                closed.set_exception(RuntimeError("Executor shut down"))
                return closed
            cached = self._completed_outputs.pop(async_output_id, None)
            if cached is not None:
                return cached
            existing = self._output_futures.get(async_output_id)
            if existing is not None:
                if isinstance(existing, DropPlaceholder) or not existing.done():
                    return existing
                self._output_futures.pop(async_output_id, None)
                self._remember_dropped(async_output_id)
            if async_output_id in self._dropped_output_ids:
                self._dropped_output_ids.move_to_end(async_output_id)
                dropped: concurrent.futures.Future = concurrent.futures.Future()
                dropped.set_exception(dropped_output_error(async_output_id))
                return dropped
            fut: concurrent.futures.Future = concurrent.futures.Future()
            self._output_futures[async_output_id] = fut
        return fut

    def drop_output(self, async_output_id: str) -> None:
        """Discard an async output that will never be waited on."""
        with self._futures_lock:
            if self._closed:
                return
            if self._completed_outputs.pop(async_output_id, None) is not None:
                self._remember_dropped(async_output_id)
                return
            if async_output_id in self._output_futures:
                return
            self._output_futures[async_output_id] = DropPlaceholder()
