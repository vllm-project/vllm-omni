# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Per-frame work of a PersonaPlex session stays constant as the session grows.

The session runner, model channel and data plane run on the orchestrator
loop for every live session. These tests drive them through the runner
harness (recording stage port) and count what each frame and each Stage 1
chunk puts on the loop. The counts must not grow with the session's length,
and no frame may hide an executor round trip, a deepcopy or a JSON encode.
"""

from __future__ import annotations

import asyncio
import copy
import json
from dataclasses import dataclass
from typing import Any

import pytest

from tests.engine.duplex.test_session_runner import close_harness
from tests.engine.duplex.test_session_runner_personaplex import (
    PREFILL_SLOTS,
    code2wav_output,
    frame,
    open_personaplex_harness,
)
from vllm_omni.model_executor.models.personaplex.duplex import stage0
from vllm_omni.model_executor.models.personaplex.duplex.config import FRAME_SIZE

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_POLL_LIMIT = 10_000


@dataclass
class LoopWork:
    """Loop callbacks, tasks and executor jobs, deepcopies and JSON encodes, counted while active."""

    callbacks: int = 0
    tasks: int = 0
    executor_jobs: int = 0
    deepcopies: int = 0
    json_encodes: int = 0
    active: bool = False

    def install(self, monkeypatch: pytest.MonkeyPatch) -> LoopWork:
        loop = asyncio.get_running_loop()

        def counting(name: str, target: Any):
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                if self.active:
                    setattr(self, name, getattr(self, name) + 1)
                return target(*args, **kwargs)

            return wrapper

        for owner, attr, name in (
            (loop, "call_soon", "callbacks"),
            (loop, "call_soon_threadsafe", "callbacks"),
            (loop, "create_task", "tasks"),
            (loop, "run_in_executor", "executor_jobs"),
            (copy, "deepcopy", "deepcopies"),
            (json, "dumps", "json_encodes"),
        ):
            monkeypatch.setattr(owner, attr, counting(name, getattr(owner, attr)))
        return self

    def start(self) -> None:
        self.callbacks = self.tasks = self.executor_jobs = self.deepcopies = self.json_encodes = 0
        self.active = True

    def stop(self) -> LoopWork:
        self.active = False
        return self


@pytest.fixture(autouse=True)
def _fake_prefill(monkeypatch: pytest.MonkeyPatch) -> None:
    # Opening a session sizes the prefill from the voice prompt on disk; the
    # harness has no model directory.
    monkeypatch.setattr(stage0, "personaplex_prefill_slots", lambda model_path, voice, persona: PREFILL_SLOTS)


async def _drive_frames(h, work: LoopWork, frames: int) -> tuple[LoopWork, int]:
    """Submit *frames* appends and yield until all reached the stage; return the counts and our own yields."""
    expected = len(h.port.submissions) + frames
    work.start()
    for _ in range(frames):
        h.submit(frame())
    polls = 0
    while len(h.port.submissions) < expected and polls < _POLL_LIMIT:
        await asyncio.sleep(0)
        polls += 1
    # Let the finished append tasks run their done callbacks.
    for _ in range(4):
        await asyncio.sleep(0)
        polls += 1
    assert len(h.port.submissions) == expected
    return work.stop(), polls


async def _drive_chunks(h, work: LoopWork, first_chunk: int, chunks: int) -> tuple[LoopWork, int]:
    """Deliver cumulative Code2Wav chunks ``first_chunk..`` and yield until each was handled."""
    request_id = h.stage0_request_id(epoch=0)
    work.start()
    polls = 0
    for index in range(first_chunk, first_chunk + chunks):
        output = code2wav_output(request_id, samples=(index + 1) * 5 * FRAME_SIZE, text="x" * (index + 1))
        h.deliver(output, stage_id=1)
        while not h.runner._mailbox.empty() and polls < _POLL_LIMIT:
            await asyncio.sleep(0)
            polls += 1
        await asyncio.sleep(0)
        polls += 1
    return work.stop(), polls


def _per_unit(work: LoopWork, polls: int, units: int) -> dict[str, float]:
    # Each of our own sleep(0) yields adds one callback and one loop turn.
    return {
        "callbacks": (work.callbacks - polls) / units,
        "tasks": work.tasks / units,
        "executor_jobs": work.executor_jobs / units,
        "deepcopies": work.deepcopies / units,
        "json_encodes": work.json_encodes / units,
    }


@pytest.mark.asyncio
async def test_input_frames_cost_the_same_early_and_late_in_a_session(monkeypatch: pytest.MonkeyPatch) -> None:
    h = await open_personaplex_harness()
    try:
        work = LoopWork().install(monkeypatch)
        await h.run(frame())  # the first append carries the prefill; measure steady state
        early = _per_unit(*await _drive_frames(h, work, 8), 8)
        await _drive_frames(h, work, 64)
        late = _per_unit(*await _drive_frames(h, work, 8), 8)
    finally:
        await close_harness(h)

    assert early["executor_jobs"] == late["executor_jobs"] == 0
    assert early["deepcopies"] == late["deepcopies"] == 0
    assert early["json_encodes"] == late["json_encodes"] == 0
    # A constant number of tasks and loop callbacks per frame, however long the session.
    assert late["tasks"] == early["tasks"]
    assert late["callbacks"] <= early["callbacks"] + 0.5


@pytest.mark.asyncio
async def test_stage1_chunks_cost_the_same_early_and_late_in_a_response(monkeypatch: pytest.MonkeyPatch) -> None:
    h = await open_personaplex_harness()
    try:
        work = LoopWork().install(monkeypatch)
        await h.run(frame())
        await _drive_chunks(h, work, 0, 1)  # opens the response
        early = _per_unit(*await _drive_chunks(h, work, 1, 8), 8)
        await _drive_chunks(h, work, 9, 64)
        late = _per_unit(*await _drive_chunks(h, work, 73, 8), 8)
    finally:
        await close_harness(h)

    assert early["executor_jobs"] == late["executor_jobs"] == 0
    assert early["deepcopies"] == late["deepcopies"] == 0
    assert early["json_encodes"] == late["json_encodes"] == 0
    assert late["callbacks"] <= early["callbacks"] + 0.5
    assert late["tasks"] <= early["tasks"]
