# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lockstep-contract tests for the PersonaPlex session driver (no GPU / no moshi).

A stub :class:`FrameStepper` stands in for the real Moshi backend so we can assert
the framing + 1-frame-in / 1-frame-out invariants without weights.
"""

import numpy as np
import pytest

from vllm_omni.experimental.fullduplex.personaplex.config import FRAME_SIZE, PersonaPlexConfig
from vllm_omni.experimental.fullduplex.personaplex.engine import FrameOutput, FrameStepper
from vllm_omni.experimental.fullduplex.personaplex.session import PersonaPlexSession

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


def test_stub_satisfies_protocol():
    assert isinstance(StubStepper(), FrameStepper)


def test_feed_emits_one_output_per_complete_frame():
    s = PersonaPlexSession(StubStepper(), PersonaPlexConfig())
    s.open(voice_prompt="NATF2.pt", persona="be terse")
    out = s.feed(np.zeros(FRAME_SIZE * 3, dtype=np.float32))
    assert len(out) == 3
    # lockstep order preserved: frame i -> audio filled with i
    assert [int(o.audio[0]) for o in out] == [1, 2, 3]
    assert [o.text for o in out] == ["t1", None, "t3"]


def test_partial_frames_are_buffered_until_complete():
    stub = StubStepper()
    s = PersonaPlexSession(stub, PersonaPlexConfig())
    s.open()
    assert s.feed(np.zeros(FRAME_SIZE // 2, dtype=np.float32)) == []  # nothing yet
    assert stub.steps == 0
    out = s.feed(np.zeros(FRAME_SIZE, dtype=np.float32))  # 1.5 frames total -> 1 frame
    assert len(out) == 1
    assert stub.steps == 1


def test_flush_pads_the_tail():
    stub = StubStepper()
    s = PersonaPlexSession(stub, PersonaPlexConfig())
    s.open()
    s.feed(np.ones(FRAME_SIZE + 10, dtype=np.float32))  # 1 frame + 10-sample tail
    assert stub.steps == 1
    tail = s.flush()
    assert len(tail) == 1 and stub.steps == 2
    assert s.flush() == []  # nothing left


def test_run_concatenates_audio_and_joins_text():
    s = PersonaPlexSession(StubStepper(), PersonaPlexConfig())
    s.open()
    audio, text = s.run(np.zeros(FRAME_SIZE * 4, dtype=np.float32))
    assert audio.shape[0] == FRAME_SIZE * 4
    assert text == "t1t3"  # frames 1..4, text on odds


def test_open_forwards_voice_and_persona():
    stub = StubStepper()
    s = PersonaPlexSession(stub, PersonaPlexConfig())
    s.open(voice_prompt="NATM1.pt", persona="pirate")
    assert stub.opened == ("NATM1.pt", "pirate")


def test_feed_before_open_raises():
    s = PersonaPlexSession(StubStepper(), PersonaPlexConfig())
    with pytest.raises(RuntimeError):
        s.feed(np.zeros(FRAME_SIZE, dtype=np.float32))
