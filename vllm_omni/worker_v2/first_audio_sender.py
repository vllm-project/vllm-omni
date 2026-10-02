# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Deliver in-stage first audio as soon as its device copy completes.

The regular output of a step is published only after the whole step's
outputs are copied, materialized for every request, popped by the engine loop
and serialized together. A stream's first audio chunk does not need any of
that, so a dedicated thread waits for the chunk's own copy event and hands a
one-request output straight to the engine core's output queue.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from vllm.logger import init_logger

from vllm_omni.data_entry_keys import FIRST_AUDIO_KEY

logger = init_logger(__name__)


class FirstAudioSink(Protocol):
    def prepare(self, request_ids: list[str]) -> _PreparedDelivery:
        """Freeze request routes on the submitting thread before accepting delivery."""
        ...


class FirstAudioSender:
    def __init__(self, sink: FirstAudioSink) -> None:
        if not callable(getattr(sink, "prepare", None)):
            raise TypeError("FirstAudioSender requires a sink with prepare(request_ids)")
        self._sink = sink
        self._queue: queue.SimpleQueue[
            tuple[torch.cuda.Event, torch.Tensor, list[str], torch.Tensor, torch.Tensor | None, _PreparedDelivery]
            | None
        ]
        self._queue = queue.SimpleQueue()
        self._thread = threading.Thread(target=self._run, daemon=True, name="omni-first-audio-sender")
        self._thread.start()

    def submit(
        self,
        request_ids: list[str],
        pcm: torch.Tensor,
        sample_rate: torch.Tensor,
        valid: torch.Tensor | None = None,
    ) -> list[str]:
        """Queue a D2H copy of ``pcm`` [n, samples] on the current stream and deliver it once done.

        ``valid`` [n] (on device) drops rows whose frame turned out not to be
        audio (e.g. a codec EOS sample); it is read only after the copy.

        Return the request IDs whose delivery was accepted. Rows without an
        engine output route must retain regular codec delivery instead of
        promising the orchestrator a first frame that can never arrive.
        """
        delivery = self._sink.prepare(request_ids)
        accepted = list(delivery.routes)
        if not accepted:
            return []
        host = torch.empty(pcm.shape, dtype=pcm.dtype, pin_memory=True)
        host.copy_(pcm, non_blocking=True)
        host_valid = None
        if valid is not None:
            host_valid = torch.empty(valid.shape, dtype=torch.bool, pin_memory=True)
            host_valid.copy_(valid, non_blocking=True)
        copied = torch.cuda.Event()
        copied.record()
        self._queue.put((copied, host, list(request_ids), sample_rate, host_valid, delivery))
        return accepted

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=5)

    def _run(self) -> None:
        while (item := self._queue.get()) is not None:
            copied, host, request_ids, sample_rate, host_valid, delivery = item
            pending = request_ids
            try:
                copied.synchronize()
                rows = [row for row in range(len(request_ids)) if host_valid is None or bool(host_valid[row])]
                pending = [request_ids[row] for row in rows]
                if rows:
                    delivery(pending, [host[row].clone() for row in rows], sample_rate)
            except Exception:
                # If synchronization failed, validity is not readable yet.
                # Otherwise only valid, not-yet-delivered rows owe first audio.
                logger.exception("First-audio delivery failed for %s", pending)
                delivery.fail(pending)


@dataclass
class _PreparedDelivery:
    sink: _EngineOutputSink
    # Frozen client indices for outstanding deliveries. Successful enqueues
    # retire their routes so a later failure cannot abort completed deliveries.
    routes: dict[str, int]

    def __call__(self, request_ids: list[str], pcm_rows: list[torch.Tensor], sample_rate: torch.Tensor) -> None:
        self.sink._send(self.routes, request_ids, pcm_rows, sample_rate)

    def fail(self, request_ids: list[str]) -> None:
        self.sink._fail(self.routes, request_ids)


class _EngineOutputSink:
    def __init__(self, output_queue: Any, scheduler: Any) -> None:
        self.output_queue = output_queue
        self.scheduler = scheduler

    def prepare(self, request_ids: list[str]) -> _PreparedDelivery:
        routes = {}
        for request_id in request_ids:
            request = self.scheduler.requests.get(request_id)
            if request is not None:
                routes[request_id] = int(getattr(request, "client_index", 0) or 0)
        return _PreparedDelivery(self, routes)

    def _send(self, routes: dict[str, int], request_ids, pcm_rows, sample_rate) -> None:
        from vllm_omni.engine import OmniEngineCoreOutput, OmniEngineCoreOutputs

        by_client: dict[int, list[OmniEngineCoreOutput]] = {}
        for request_id, pcm in zip(request_ids, pcm_rows, strict=True):
            if request_id not in routes:
                continue
            by_client.setdefault(routes[request_id], []).append(
                OmniEngineCoreOutput(
                    request_id=request_id,
                    new_token_ids=[],
                    multimodal_output={
                        "model_outputs": pcm,
                        "sr": sample_rate,
                        FIRST_AUDIO_KEY: torch.tensor(True),
                    },
                )
            )
        for client_index, outputs in by_client.items():
            self.output_queue.put_nowait((client_index, OmniEngineCoreOutputs(outputs=outputs)))
            for output in outputs:
                routes.pop(output.request_id)

    def _fail(self, routes: dict[str, int], request_ids: list[str]) -> None:
        from vllm.v1.engine import FinishReason

        from vllm_omni.engine import OmniEngineCoreOutput, OmniEngineCoreOutputs

        for request_id in request_ids:
            if request_id in routes:
                output = OmniEngineCoreOutput(
                    request_id=request_id,
                    new_token_ids=[],
                    finish_reason=FinishReason.ERROR,
                    stop_reason="First-audio delivery failed",
                )
                self.output_queue.put_nowait((routes[request_id], OmniEngineCoreOutputs(outputs=[output])))
                routes.pop(request_id)


def engine_output_queue_sink(output_queue: Any, scheduler: Any) -> _EngineOutputSink:
    return _EngineOutputSink(output_queue, scheduler)
