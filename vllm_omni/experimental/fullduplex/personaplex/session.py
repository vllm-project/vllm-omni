# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PersonaPlex lockstep session driver — the runnable serving primitive.

Mirrors JoyVL's ``serving/session.py``: a thin, framework-free driver that owns a
``FrameStepper`` and turns arbitrary-length PCM chunks from a client into exact
80 ms frames, stepping the model one frame at a time. This is the path that
actually runs end to end (offline driver and any WS server build on it); the
``DuplexAdapter`` is a separate demonstration of the generic framework seam.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from vllm_omni.experimental.fullduplex.personaplex.config import PersonaPlexConfig
from vllm_omni.experimental.fullduplex.personaplex.engine import FrameOutput, FrameStepper

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Realtime queue bounds: 80 ms/frame, so 50 frames = 4 s of backlog. Anything
# beyond that means the peer cannot keep up and old frames are already useless.
MAX_PENDING_INPUT = 50
MAX_PENDING_OUTPUT = 50


def offer_realtime(queue: asyncio.Queue, item: Any) -> None:
    """Enqueue with a drop-oldest policy (realtime streams must never grow unbounded)."""
    while True:
        try:
            queue.put_nowait(item)
            return
        except asyncio.QueueFull:
            with contextlib.suppress(asyncio.QueueEmpty):
                queue.get_nowait()


@dataclass
class PersonaPlexServingSessionState:
    """Mutable per-connection serving state (the minicpmo45 ``session.py`` role).

    One instance per WebSocket connection, for both server modes: the
    single-session server uses ``pcm_queue`` (decoded mic PCM awaiting the
    engine); the batched server uses ``out_queue`` + ``slot``/``epoch`` (engine
    output awaiting send, and the claimed slot lease). ``closed`` tears both
    pump loops down.
    """

    pcm_queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=MAX_PENDING_INPUT))
    out_queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=MAX_PENDING_OUTPUT))
    closed: bool = False
    slot: Any = None  # batched mode: the claimed serving/batched.py Slot
    epoch: int = 0  # batched mode: slot epoch at claim time (stale-guard)
    # Single-session mode: completion event of the engine call currently running
    # on a worker thread. Task cancellation cannot stop that thread (and marks
    # the asyncio future cancelled while the thread runs on), so the server
    # waits on this event before releasing its lease (DuplexServer._engine_call).
    inflight: Any = None


class PersonaPlexSession:
    """Drive one PersonaPlex conversation in strict frame lockstep.

    A session buffers incoming PCM and emits exactly one :class:`FrameOutput` per
    completed ``frame_size`` window. Partial tails are held until enough samples
    arrive (or ``flush()`` pads them out at end of stream).
    """

    def __init__(self, stepper: FrameStepper, config: PersonaPlexConfig | None = None) -> None:
        self.stepper = stepper
        self.config = config or PersonaPlexConfig()
        self.frame_size = stepper.frame_size
        self._pending = np.zeros(0, dtype=np.float32)
        self._opened = False

    def open(self, voice_prompt: str | None = None, persona: str | None = None) -> None:
        """Inject the voice clone + persona and arm the conversation."""
        self.stepper.open_session(voice_prompt=voice_prompt, persona=persona)
        self._pending = np.zeros(0, dtype=np.float32)
        self._opened = True

    def feed(self, pcm: NDArray[np.float32]) -> list[FrameOutput]:
        """Append PCM and step every newly completed frame.

        Returns one :class:`FrameOutput` per frame consumed from the buffer this
        call (possibly empty if fewer than ``frame_size`` samples are buffered).
        """
        if not self._opened:
            raise RuntimeError("PersonaPlexSession.open() must be called before feed()")
        samples = np.ascontiguousarray(pcm, dtype=np.float32).reshape(-1)
        self._pending = np.concatenate([self._pending, samples]) if self._pending.size else samples

        outputs: list[FrameOutput] = []
        n = self.frame_size
        while self._pending.shape[0] >= n:
            frame, self._pending = self._pending[:n], self._pending[n:]
            outputs.append(self.stepper.step(frame))
        return outputs

    def flush(self) -> list[FrameOutput]:
        """Zero-pad and step any buffered tail; call once at end of input."""
        if self._pending.shape[0] == 0:
            return []
        pad = self.frame_size - self._pending.shape[0]
        frame = np.concatenate([self._pending, np.zeros(pad, dtype=np.float32)])
        self._pending = np.zeros(0, dtype=np.float32)
        return [self.stepper.step(frame)]

    def run(self, pcm: NDArray[np.float32]) -> tuple[NDArray[np.float32], str]:
        """Convenience for offline use: feed a whole utterance, return (audio, text).

        Concatenates every emitted agent frame and joins the inner-monologue text.
        """
        outputs = self.feed(pcm) + self.flush()
        chunks = [o.audio for o in outputs if o.audio is not None]
        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        text = "".join(o.text for o in outputs if o.text)
        return audio, text
