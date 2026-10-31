# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Materialize native producer output independently of scheduler consumption."""

from concurrent.futures import Future, ThreadPoolExecutor
from threading import BoundedSemaphore, Lock
from typing import Any

from vllm.v1.outputs import AsyncModelRunnerOutput, KVConnectorOutput


class NativeAsyncOutput(AsyncModelRunnerOutput):
    """Consume an ordered publication on the engine's owner thread."""

    def __init__(self, future: Future, plane: Any, copy_event: Any) -> None:
        self._future: Future | None = future
        self._plane = plane
        self.copy_event = copy_event
        self._resolved: Future = Future()
        # Filled when the worker future yields an output, before plane attach.
        # enqueue_output salvages this after get_output() raises.
        self._kv_connector_output: KVConnectorOutput | None = None

    def get_output(self) -> Any:
        # Like the upstream lazy AsyncOutputFuture, consumption belongs to one
        # engine thread. Cache finalization and errors for repeated consumption.
        if not self._resolved.done():
            try:
                assert self._future is not None
                output = self._future.result()
                self._kv_connector_output = getattr(output, "kv_connector_output", None)
                output.omni_connector_output = self._plane.get_omni_connector_output()
                self._resolved.set_result(output)
            except Exception as error:
                self._resolved.set_exception(error)
            finally:
                if self._resolved.done():
                    # Do not retain raw snapshots or a callback bound to self.
                    self._future = None
                    self._plane = None
        return self._resolved.result()

    def get_kv_connector_output(self) -> KVConnectorOutput | None:
        # Do not block the engine thread. A connector output captured before
        # plane attach survives a later get_output() failure. Otherwise only a
        # future that has already finished can be salvaged.
        snap = getattr(self, "_kv_connector_output", None)
        if snap is not None:
            return snap
        future = (
            self._resolved
            if getattr(self, "_resolved", None) is not None and self._resolved.done()
            else getattr(self, "_future", None)
        )
        if future is None or not future.done():
            return None
        try:
            output = future.result()
        except Exception:
            return None
        return getattr(output, "kv_connector_output", None)


class NativeOutputWorker:
    """Bounded FIFO CPU materialization and native-queue publication.

    Input outputs must have no owner-thread finalizer installed. Reservations
    and metadata writes must be complete before submit. This worker never
    updates scheduler state or drains connector notifications.
    """

    def __init__(self, capacity: int, device: Any | None = None) -> None:
        if capacity < 1:
            raise ValueError("Native output capacity must be positive")
        self._slots = BoundedSemaphore(capacity)
        self._lock = Lock()
        self._closed = False
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="omni-native-materialize",
            initializer=self._initialize_device,
            initargs=(device,),
        )

    @staticmethod
    def _initialize_device(device: Any | None) -> None:
        if device is not None:
            from vllm.platforms import current_platform

            current_platform.set_device(device)

    @staticmethod
    def _materialize(output: AsyncModelRunnerOutput, plane: Any) -> Any:
        result = output.get_output()
        plane.enqueue_outputs(
            req_ids=list(result.req_ids),
            inter_stage_outputs=getattr(result, "inter_stage_outputs", None),
            sampled_token_ids=getattr(result, "sampled_token_ids", None),
        )
        result.inter_stage_outputs = None
        return result

    def submit(self, output: AsyncModelRunnerOutput, plane: Any) -> NativeAsyncOutput:
        with self._lock:
            if self._closed:
                raise RuntimeError("Native output worker is closed")
        self._slots.acquire()
        try:
            with self._lock:
                if self._closed:
                    raise RuntimeError("Native output worker is closed")
                future = self._executor.submit(self._materialize, output, plane)
        except BaseException:
            self._slots.release()
            raise
        future.add_done_callback(lambda _: self._slots.release())
        return NativeAsyncOutput(future, plane, output.copy_event)

    def close(self) -> None:
        with self._lock:
            self._closed = True
        # Drain submitted publications before the native data plane is closed.
        self._executor.shutdown(wait=True)
