# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Cross-session Silero scoring for duplex turn detection.

Every session's detector advances on Silero's 32 ms frame grid. Scored from the
session thread pool one session and one frame at a time, each frame is an ONNX
call that releases the GIL and must take it back afterwards; with many sessions
those hand-offs, not the model, fill the pool and stall the API process while
the CPU is still mostly idle.

:class:`TurnDetectionBatcher` runs every session's turn detection on one thread
instead, in iterations. An iteration admits every append submitted since the last
one, advances each running append's detector by one frame (a detector's frames
depend on each other through Silero's recurrent state) and scores the iteration in
a single batched run of the shared model, so an iteration costs one model call
whatever the number of sessions. An append's caller is woken in the iteration it
finishes: a long append only stays in more iterations, it never holds back a
shorter or a later one.

Smart Turn (``semantic_vad``) is too slow to run between iterations. Its work goes
to the session executor while that append sits the iterations out, and the append
rejoins them with the result.
"""

from __future__ import annotations

import asyncio
import threading
from collections import deque
from collections.abc import Callable
from concurrent.futures import Executor, Future
from dataclasses import dataclass

from vllm.logger import init_logger

from vllm_omni.engine.duplex.turn_detection import TurnDetectionSegment, TurnDetector
from vllm_omni.engine.duplex.vad import (
    FrameScoreRequest,
    FrameSteps,
    TurnScoreRequest,
    score_frames,
)

logger = init_logger(__name__)

#: :attr:`_Append.pending` of an append that has not started yet.
_START = object()


@dataclass(eq=False, slots=True)
class _Append:
    """One append's turn detection, from submission to its caller's future."""

    steps: FrameSteps[list[TurnDetectionSegment]]
    detector: TurnDetector
    future: asyncio.Future[list[TurnDetectionSegment]]
    #: What ``steps`` waits for: a frame to score or turn-scorer work; None once it has finished.
    request: FrameScoreRequest | TurnScoreRequest | None = None
    segments: list[TurnDetectionSegment] | None = None
    error: BaseException | None = None
    #: What to resume ``steps`` with next: :data:`_START`, or what it asked for.
    pending: object = _START

    def advance(self) -> None:
        """Resume ``steps`` with :attr:`pending` until its next request or its end.

        :data:`_START` starts it; an exception is thrown in.
        """
        value, self.pending = self.pending, None
        try:
            if value is _START:
                self.request = next(self.steps)
            elif isinstance(value, BaseException):
                self.request = self.steps.throw(value)
            else:
                self.request = self.steps.send(value)
        except StopIteration as stop:
            self.request = None
            self.segments = stop.value
        except Exception as exc:
            self.request = None
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
    orchestrator loop; that loop is bound by the first append, and each iteration's
    finished appends go back to it in a single hand-off. Turn-scorer work runs
    on ``executor``. The worker thread starts with the first append and is
    stopped by :meth:`close`.
    """

    def __init__(self, executor: Executor, *, thread_name: str = "duplex-vad") -> None:
        self._executor = executor
        self._thread_name = thread_name
        self._cond = threading.Condition()
        #: Submitted appends and appends back from turn scoring, in arrival order.
        self._inbox: deque[_Append] = deque()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._closed = False
        # Owned by the worker thread:
        #: Appends whose next request is a frame, scored in the next iteration.
        self._active: list[_Append] = []
        #: Appends whose turn-scorer work is still on the executor.
        self._away = 0

    async def process_segments(
        self, detector: TurnDetector, audio: str | bytes, *, fmt: str, sample_rate_hz: int | None
    ) -> list[TurnDetectionSegment]:
        """``detector.process_segments`` for one append, scored together with every other session's.

        ``audio`` is base64 or the already-decoded bytes. Raises what ``process_segments`` raises.
        """
        loop = asyncio.get_running_loop()
        append = _Append(
            steps=detector.segment_steps(audio, fmt=fmt, sample_rate_hz=sample_rate_hz),
            detector=detector,
            future=loop.create_future(),
        )
        with self._cond:
            if self._closed:
                raise RuntimeError("duplex turn detection batcher is closed")
            if self._loop is None:
                self._loop = loop
            elif loop is not self._loop:
                raise RuntimeError("duplex turn detection batcher serves one event loop (the orchestrator's)")
            self._inbox.append(append)
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
                while not self._inbox and not self._active:
                    if self._closed and not self._away:
                        return
                    self._cond.wait()
                inbox = list(self._inbox)
                self._inbox.clear()
            finished: list[_Append] = []
            try:
                self._iter(inbox, finished)
            except BaseException as exc:  # never strand a caller on its future
                logger.exception("Duplex turn detection iteration failed")
                self._abandon([*self._active, *inbox], exc, finished)
            if finished:
                try:
                    self._loop.call_soon_threadsafe(_resolve, finished)
                except RuntimeError:
                    pass  # the loop already closed; nobody is waiting

    def _iter(self, inbox: list[_Append], finished: list[_Append]) -> None:
        for append in inbox:
            if append.pending is not _START:
                # this append request is returned from SmartTurn detection
                self._away -= 1
            self._step(append, finished)
        if not self._active:
            return
        waiting, self._active = self._active, []
        try:
            scores: list[float | BaseException] = score_frames(
                [(append.detector.vad, append.request.frame) for append in waiting]
            )
        except Exception as exc:
            scores = [exc] * len(waiting)
        for append, score in zip(waiting, scores):
            append.pending = score
            self._step(append, finished)

    def _step(self, append: _Append, finished: list[_Append]) -> None:
        """Advance ``append`` with its pending value and file it by what it asks for next."""
        append.advance()
        request = append.request
        if isinstance(request, FrameScoreRequest):
            # Frame score request, submitted to _active and batcher will execute it
            self._active.append(append)
        elif isinstance(request, TurnScoreRequest):
            # Smart Turn loading or inference, offloaded to the session executor
            if request.is_loading:
                self._offload(append, append.detector.load_turn_scorer)
            else:
                self._offload(append, append.detector.score_turn, request.window)
        else:
            if request is not None:
                append.request = None
                append.error = TypeError(f"turn detection yielded an unknown request: {type(request).__name__}")
            finished.append(append)

    def _offload(self, append: _Append, work: Callable[..., object], *args: object) -> None:
        """Run Smart Turn ``work`` on the executor; ``append`` rejoins the iterations with its result."""

        def done(future: Future[object]) -> None:
            try:
                value = future.result()
            except BaseException as exc:  # failed, or cancelled by the executor's shutdown
                value = exc
            append.pending = value  # published to the worker by the lock below
            with self._cond:
                self._inbox.append(append)
                self._cond.notify()

        self._away += 1
        try:
            future = self._executor.submit(work, *args)
        except Exception as exc:  # the executor is shut down
            # A done future runs ``done`` at once, so this failure rejoins the same way.
            future = Future()
            future.set_exception(exc)
        future.add_done_callback(done)

    def _abandon(self, appends: list[_Append], exc: BaseException, finished: list[_Append]) -> None:
        """Fail ``appends`` that have not finished after an unexpected error."""
        self._active = []
        stranded = {id(append): append for append in appends if append.segments is None and append.error is None}
        for append in stranded.values():
            append.request = None
            append.error = exc
            finished.append(append)


__all__ = ["TurnDetectionBatcher"]
