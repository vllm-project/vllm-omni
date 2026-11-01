# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Cross-session Silero scoring for duplex turn detection.

Every session's detector advances on Silero's 32 ms frame grid. Scored from the
session thread pool one session and one frame at a time, each frame is an ONNX
call that releases the GIL and must take it back afterwards; with many sessions
those hand-offs, not the model, fill the pool and stall the API process while
the CPU is still mostly idle.

:class:`TurnDetectionBatcher` runs every pending append's turn detection on one
thread instead. It takes all appends queued so far, steps their detectors in
lockstep (a detector's frames depend on each other through Silero's recurrent
state, so each round holds at most one frame per session) and scores each round
in a single batched run of the shared model. A batch therefore costs a few model
calls whatever the number of sessions, and each append's caller is woken once.
"""

from __future__ import annotations

import asyncio
import threading
from collections import deque
from dataclasses import dataclass

import numpy as np
from vllm.logger import init_logger

from vllm_omni.engine.duplex.turn_detection import TurnDetectionSegment, TurnDetector
from vllm_omni.engine.duplex.vad import FrameSteps, SileroStreamingVAD, score_frames

logger = init_logger(__name__)


@dataclass(eq=False, slots=True)
class _Append:
    """One append's turn detection, from submission to its caller's future."""

    steps: FrameSteps[list[TurnDetectionSegment]]
    vad: SileroStreamingVAD
    future: asyncio.Future[list[TurnDetectionSegment]]
    #: The frame ``steps`` is waiting to have scored; None once it has finished.
    frame: np.ndarray | None = None
    segments: list[TurnDetectionSegment] | None = None
    error: BaseException | None = None

    def advance(self, score: float | BaseException | None) -> None:
        """Resume ``steps`` with ``score`` (None to start it) until it asks for the next frame or ends."""
        try:
            if score is None:
                self.frame = next(self.steps)
            elif isinstance(score, BaseException):
                self.frame = self.steps.throw(score)
            else:
                self.frame = self.steps.send(score)
        except StopIteration as stop:
            self.frame = None
            self.segments = stop.value
        except Exception as exc:
            self.frame = None
            self.error = exc


def _resolve(appends: list[_Append]) -> None:
    """Complete the callers' futures; runs on the orchestrator loop."""
    for append in appends:
        if append.future.done():
            continue  # the caller gave up (session closed); its detector state is still advanced
        if append.error is not None:
            append.future.set_exception(append.error)
        else:
            append.future.set_result(append.segments or [])


class TurnDetectionBatcher:
    """Runs duplex turn detection for every session on one thread, batching Silero across sessions.

    One per session manager, so every caller is a session runner on the one
    orchestrator loop; that loop is bound by the first append, and a batch's
    results go back to it in a single hand-off. The worker thread starts with
    the first append and is stopped by :meth:`close`.
    """

    def __init__(self, *, thread_name: str = "duplex-vad") -> None:
        self._thread_name = thread_name
        self._cond = threading.Condition()
        self._queue: deque[_Append] = deque()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._closed = False

    async def process_segments(
        self, detector: TurnDetector, base64_audio: str, *, fmt: str, sample_rate_hz: int | None
    ) -> list[TurnDetectionSegment]:
        """``detector.process_segments`` for one append, scored together with every other session's.

        Raises what ``process_segments`` raises.
        """
        loop = asyncio.get_running_loop()
        append = _Append(
            steps=detector.segment_steps(base64_audio, fmt=fmt, sample_rate_hz=sample_rate_hz),
            vad=detector.vad,
            future=loop.create_future(),
        )
        with self._cond:
            if self._closed:
                raise RuntimeError("duplex turn detection batcher is closed")
            if self._loop is None:
                self._loop = loop
            elif loop is not self._loop:
                raise RuntimeError("duplex turn detection batcher serves one event loop (the orchestrator's)")
            self._queue.append(append)
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name=self._thread_name, daemon=True)
                self._thread.start()
            self._cond.notify()
        return await append.future

    def close(self) -> None:
        """Stop the worker after it finishes the appends already submitted."""
        with self._cond:
            self._closed = True
            self._cond.notify()

    def _run(self) -> None:
        while True:
            with self._cond:
                while not self._queue and not self._closed:
                    self._cond.wait()
                if not self._queue:
                    return
                # Everything queued joins this one:
                appends = list(self._queue)
                self._queue.clear()
            try:
                self._detect(appends)
            except BaseException as exc:  # never strand a caller on its future
                logger.exception("Duplex turn detection batch failed")
                for append in appends:
                    if append.segments is None and append.error is None:
                        append.error = exc
            try:
                self._loop.call_soon_threadsafe(_resolve, appends)
            except RuntimeError:
                pass  # the loop already closed; nobody is waiting

    @staticmethod
    def _detect(appends: list[_Append]) -> None:
        for append in appends:
            append.advance(None)
        waiting = [append for append in appends if append.frame is not None]
        while waiting:
            scores = score_frames([(append.vad, append.frame) for append in waiting])
            for append, score in zip(waiting, scores):
                append.advance(score)
            waiting = [append for append in waiting if append.frame is not None]


__all__ = ["TurnDetectionBatcher"]
