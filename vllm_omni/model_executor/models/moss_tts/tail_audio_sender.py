# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Deliver detached codec tails for the MOSS Local fused pipeline."""

from __future__ import annotations

import queue
import threading
from typing import TYPE_CHECKING

import torch
from vllm.logger import init_logger

from vllm_omni.data_entry_keys import FIRST_AUDIO_REQUIRED_KEY

if TYPE_CHECKING:
    from .fused_scheduler import MossFusedScheduler

logger = init_logger(__name__)


class TailAudioSender:
    """Prepare audio decoded after a request stopped generating.

    The scheduler released the request at its stop and holds its terminal
    output in ``scheduler.omni_deferred_finishes``. Pieces submitted here are
    handled in order on one thread: each waits for its decode event and is
    copied on a private stream (no legacy-stream synchronization with the
    talker), then queued on ``scheduler.omni_tail_outputs``. The scheduler emits
    them after its next step's own outputs, so a request's audio from steps in
    flight at its stop, its tail and its terminal output stay in order.
    """

    def __init__(self, scheduler: MossFusedScheduler, device: torch.device) -> None:
        self.scheduler = scheduler
        # The sender thread selects the device by index.
        index = device.index if device.index is not None else torch.accelerator.current_device_index()
        self.device = torch.device(device.type, index)
        self._queue: queue.SimpleQueue[
            tuple[str, list[tuple[torch.Tensor, torch.cuda.Event | None]], torch.Tensor, bool, bool] | None
        ] = queue.SimpleQueue()
        self._started = threading.Event()
        self._start_error: Exception | None = None
        self._thread = threading.Thread(target=self._run, daemon=True, name="omni-tail-audio-sender")
        self._thread.start()
        if not self._started.wait(timeout=10):
            self._queue.put(None)
            raise TimeoutError("Tail-audio sender initialization timed out")
        if self._start_error is not None:
            raise RuntimeError("Tail-audio sender initialization failed") from self._start_error

    def is_deferred(self, request_id: str) -> bool:
        from vllm.v1.engine import FinishReason

        held = self.scheduler.omni_deferred_finishes.get(request_id)
        return held is not None and held.get("finish_reason") != FinishReason.ERROR

    def submit(
        self,
        request_id: str,
        pieces: list[tuple[torch.Tensor, torch.cuda.Event | None]],
        sample_rate: torch.Tensor,
        final: bool,
        *,
        after_first: bool = False,
    ) -> None:
        """``after_first``: the request's first frame went ahead through the first-audio sender."""
        self._queue.put((request_id, pieces, sample_rate, final, after_first))

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=5)
        if self._thread.is_alive():
            raise TimeoutError("Tail-audio sender did not drain before shutdown")

    def _run(self) -> None:
        try:
            torch.accelerator.set_device_index(self.device.index)
            copy_stream = torch.cuda.Stream(device=self.device)
        except Exception as exc:
            self._start_error = exc
            return
        finally:
            self._started.set()
        while (item := self._queue.get()) is not None:
            request_id, pieces, sample_rate, final, after_first = item
            payload = None
            error = None
            try:
                hosts = []
                with torch.cuda.stream(copy_stream):
                    for pcm, event in pieces:
                        if not pcm.numel():
                            continue
                        if event is not None:
                            copy_stream.wait_event(event)
                        host = torch.empty(pcm.shape, dtype=pcm.dtype, pin_memory=True)
                        host.copy_(pcm, non_blocking=True)
                        hosts.append(host)
                copy_stream.synchronize()
                if hosts:
                    pcm = hosts[0] if len(hosts) == 1 else torch.cat(hosts, dim=-1)
                    payload = {"model_outputs": pcm, "sr": sample_rate}
            except Exception as exc:
                logger.exception("Tail audio copy failed for %s", request_id)
                error = f"Tail audio copy failed: {exc}"
                final = True
            finally:
                if error is None and after_first and (payload is not None or final):
                    # Even a terminal output without audio must not overtake the sent first frame.
                    payload = {**(payload or {}), FIRST_AUDIO_REQUIRED_KEY: torch.tensor(True)}
                del pieces
                self.scheduler.omni_tail_outputs.append((request_id, payload, final, error))
