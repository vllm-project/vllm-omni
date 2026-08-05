# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PersonaPlex adapter driven through the model-owned lockstep runtime.

PersonaPlexDuplexRuntime starts ONE response for the whole session, streams one
agent frame per user frame, and drains on close (core stays verbatim upstream;
the lockstep lifecycle is model policy) — all with a GPU-free stub stepper.
"""

import asyncio

import numpy as np
import pytest

from vllm_omni.experimental.fullduplex.core import protocol as ev
from vllm_omni.experimental.fullduplex.core.session import (
    DuplexSession,
    DuplexSessionConfig,
    DuplexState,
)
from vllm_omni.experimental.fullduplex.personaplex.adapter import (
    PersonaPlexDuplexAdapter,
    PersonaPlexDuplexRuntime,
)
from vllm_omni.experimental.fullduplex.personaplex.config import FRAME_SIZE
from vllm_omni.experimental.fullduplex.personaplex.engine import FrameOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class StubStepper:
    """Deterministic FrameStepper: frame i -> audio filled with i, text 't{i}' on odds."""

    sample_rate = 24000
    frame_size = FRAME_SIZE

    def __init__(self) -> None:
        self.opened: tuple[str | None, str | None] | None = None
        self.steps = 0

    def open_session(self, voice_prompt=None, persona=None) -> None:
        self.opened = (voice_prompt, persona)
        self.steps = 0

    def step(self, user_pcm):
        assert user_pcm.shape[0] == self.frame_size
        self.steps += 1
        i = self.steps
        return FrameOutput(
            audio=np.full(self.frame_size, float(i), dtype=np.float32),
            text=(f"t{i}" if i % 2 else None),
        )


class RecordingAdapter(PersonaPlexDuplexAdapter):
    def __init__(self, stepper: StubStepper) -> None:
        super().__init__(stepper)
        self.closed = False

    async def on_close(self, session: DuplexSession) -> None:
        self.closed = True
        await super().on_close(session)


def _collector():
    out: list[dict] = []

    async def emit(event: dict) -> None:
        out.append(event)

    return out, emit


async def _feed(events):
    for e in events:
        yield e


async def _raising_feed(frame):
    yield {"type": ev.INPUT_APPEND, "modality": "audio", "data": frame}
    await asyncio.sleep(0)
    raise RuntimeError("input failed")


def _respond_tasks() -> list[asyncio.Task]:
    current = asyncio.current_task()
    return [
        task
        for task in asyncio.all_tasks()
        if task is not current and "_respond_once" in getattr(task.get_coro(), "__qualname__", "")
    ]


def _session():
    cfg = DuplexSessionConfig(
        input_modalities=("audio",),
        output_modalities=("audio", "text"),
    )
    return DuplexSession("s", cfg)


@pytest.mark.asyncio
async def test_lockstep_one_agent_frame_per_user_frame():
    stub = StubStepper()
    adapter = PersonaPlexDuplexAdapter(stub, voice_prompt="NATF2.pt", persona="be terse")
    rt = PersonaPlexDuplexRuntime(_session(), adapter)
    out, emit = _collector()

    frame = np.zeros(FRAME_SIZE, dtype=np.float32)
    events = [{"type": ev.INPUT_APPEND, "modality": "audio", "data": frame} for _ in range(3)]
    events.append({"type": ev.CLOSE})
    await rt.run(_feed(events), emit)

    types = [e["type"] for e in out]
    # exactly one response for the whole session, drained cleanly on close
    assert types.count(ev.RESPONSE_CREATED) == 1
    assert types.count(ev.RESPONSE_DONE) == 1
    audio = [e for e in out if e["type"] == ev.RESPONSE_DELTA and e["modality"] == "audio"]
    text = [e["data"] for e in out if e["type"] == ev.RESPONSE_DELTA and e["modality"] == "text"]
    assert len(audio) == 3  # one agent frame per user frame
    assert [int(e["data"][0]) for e in audio] == [1, 2, 3]  # lockstep order
    assert text == ["t1", "t3"]
    assert stub.steps == 3
    assert stub.opened == ("NATF2.pt", "be terse")


@pytest.mark.asyncio
async def test_unaligned_chunks_are_reframed_to_80ms():
    stub = StubStepper()
    adapter = PersonaPlexDuplexAdapter(stub)
    rt = PersonaPlexDuplexRuntime(_session(), adapter)
    out, emit = _collector()

    # 2.5 frames in one chunk + 0.5 frame in another => 3 whole frames
    events = [
        {"type": ev.INPUT_APPEND, "modality": "audio", "data": np.zeros(FRAME_SIZE * 5 // 2, dtype=np.float32)},
        {"type": ev.INPUT_APPEND, "modality": "audio", "data": np.zeros(FRAME_SIZE // 2, dtype=np.float32)},
        {"type": ev.CLOSE},
    ]
    await rt.run(_feed(events), emit)
    audio = [e for e in out if e["type"] == ev.RESPONSE_DELTA and e["modality"] == "audio"]
    assert len(audio) == 3
    assert stub.steps == 3


@pytest.mark.asyncio
async def test_close_without_input_is_clean():
    adapter = PersonaPlexDuplexAdapter(StubStepper())
    rt = PersonaPlexDuplexRuntime(_session(), adapter)
    out, emit = _collector()
    await rt.run(_feed([{"type": ev.CLOSE}]), emit)
    # no input -> response never started -> no created/done, no error
    assert not any(e["type"] == ev.ERROR for e in out)
    assert not any(e["type"] == ev.RESPONSE_CREATED for e in out)


@pytest.mark.asyncio
async def test_iterator_eof_without_close_drains_and_stops():
    # Transport drop: the input iterator ends WITHOUT an explicit CLOSE event.
    # The runtime must still signal on_close so the eternal response drains
    # instead of blocking on its inbox forever.
    stub = StubStepper()
    adapter = PersonaPlexDuplexAdapter(stub)
    rt = PersonaPlexDuplexRuntime(_session(), adapter)
    out, emit = _collector()
    frame = np.zeros(FRAME_SIZE, dtype=np.float32)
    events = [{"type": ev.INPUT_APPEND, "modality": "audio", "data": frame}]
    await asyncio.wait_for(rt.run(_feed(events), emit), timeout=5.0)
    types = [e["type"] for e in out]
    assert types.count(ev.RESPONSE_DONE) == 1
    assert stub.steps == 1


@pytest.mark.asyncio
async def test_input_exception_closes_response_and_runtime() -> None:
    stub = StubStepper()
    adapter = RecordingAdapter(stub)
    rt = PersonaPlexDuplexRuntime(_session(), adapter)
    out, emit = _collector()
    frame = np.zeros(FRAME_SIZE, dtype=np.float32)

    with pytest.raises(RuntimeError, match="input failed"):
        await rt.run(_raising_feed(frame), emit)

    leaked = _respond_tasks()
    for task in leaked:
        task.cancel()
    if leaked:
        await asyncio.gather(*leaked, return_exceptions=True)

    assert adapter.closed is True
    assert rt.session.state is DuplexState.CLOSED
    assert not leaked


@pytest.mark.asyncio
async def test_default_session_config_gets_lockstep_lifecycle():
    # The lifecycle comes from the model-owned runtime class, not a session
    # flag: a plain default-constructed session config runs lockstep.
    stub = StubStepper()
    adapter = PersonaPlexDuplexAdapter(stub)
    session = DuplexSession(
        "s",
        DuplexSessionConfig(input_modalities=("audio",), output_modalities=("audio", "text")),
    )
    rt = PersonaPlexDuplexRuntime(session, adapter)
    out, emit = _collector()
    frame = np.zeros(FRAME_SIZE, dtype=np.float32)
    events = [{"type": ev.INPUT_APPEND, "modality": "audio", "data": frame}, {"type": ev.CLOSE}]
    await rt.run(_feed(events), emit)
    types = [e["type"] for e in out]
    assert types.count(ev.RESPONSE_CREATED) == 1
    assert types.count(ev.RESPONSE_DONE) == 1
    assert stub.steps == 1


@pytest.mark.asyncio
async def test_partial_tail_is_padded_and_stepped_on_close():
    # A trailing sub-frame chunk is zero-padded and stepped at close (same
    # contract as PersonaPlexSession.flush) rather than silently dropped.
    stub = StubStepper()
    adapter = PersonaPlexDuplexAdapter(stub)
    rt = PersonaPlexDuplexRuntime(_session(), adapter)
    out, emit = _collector()
    events = [
        {"type": ev.INPUT_APPEND, "modality": "audio", "data": np.ones(FRAME_SIZE // 2, dtype=np.float32)},
        {"type": ev.CLOSE},
    ]
    await rt.run(_feed(events), emit)
    audio = [e for e in out if e["type"] == ev.RESPONSE_DELTA and e["modality"] == "audio"]
    assert len(audio) == 1
    assert stub.steps == 1


@pytest.mark.asyncio
async def test_single_eternal_response_across_barge_in():
    # Moved from the core contract tests when the lockstep lifecycle became
    # model-owned: ONE response for the whole session, RESPONSE_CANCEL bumps the
    # epoch fence (playback cursor) but the stream keeps flowing, and close
    # drains via on_close.
    stub = StubStepper()
    adapter = PersonaPlexDuplexAdapter(stub)
    session = _session()
    rt = PersonaPlexDuplexRuntime(session, adapter)
    out, emit = _collector()

    frame = np.zeros(FRAME_SIZE, dtype=np.float32)
    events = [
        {"type": ev.INPUT_APPEND, "modality": "audio", "data": frame},
        {"type": ev.RESPONSE_CANCEL},  # barge-in: epoch bumps, stream continues
        {"type": ev.INPUT_APPEND, "modality": "audio", "data": frame},
        {"type": ev.CLOSE},
    ]
    await rt.run(_feed(events), emit)

    types = [e["type"] for e in out]
    assert types.count(ev.RESPONSE_CREATED) == 1, "exactly one eternal response"
    assert types.count(ev.RESPONSE_DONE) == 1
    audio = [e for e in out if e["type"] == ev.RESPONSE_DELTA and e["modality"] == "audio"]
    assert len(audio) == 2, "stream keeps flowing across barge-in"
    assert session.epoch == 1, "cancel still advanced the epoch fence"
    assert stub.steps == 2
