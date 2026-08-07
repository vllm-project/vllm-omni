# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contract tests for the fullduplex core (PR #3907 alignment).

Covers the DuplexFence identity value, the session identity rules (epoch only
advances on barge-in, response_index once per begin_response), stale-output
dropping in the turn lifecycle. The lockstep (single-eternal-response)
lifecycle is model-owned (see
tests/e2e/features/fullduplex/test_personaplex_adapter.py).
"""

import asyncio
import dataclasses

import pytest

from vllm_omni.experimental.fullduplex.core import protocol as ev
from vllm_omni.experimental.fullduplex.core.adapter import (
    DuplexAdapter,
    DuplexCapability,
    OutputChunk,
)
from vllm_omni.experimental.fullduplex.core.runtime import DuplexRuntime
from vllm_omni.experimental.fullduplex.core.session import (
    DuplexSession,
    DuplexSessionConfig,
    DuplexState,
)
from vllm_omni.experimental.fullduplex.engine.messages import DuplexFence

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_fence_is_frozen_value_identity():
    f = DuplexFence(session_id="s1")
    assert (f.epoch, f.turn_id, f.response_seq) == (0, 0, 0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        f.epoch = 1  # type: ignore[misc]
    assert DuplexFence("s1", 1, 2, 3) == DuplexFence("s1", 1, 2, 3)
    assert DuplexFence("s1", 1, 2, 3) != DuplexFence("s1", 2, 2, 3)


def test_session_identity_rules():
    s = DuplexSession(session_id="test", config=DuplexSessionConfig())
    r1, e1 = s.begin_response()
    r2, e2 = s.begin_response()
    assert r2 == r1 + 1, "response index advances exactly once per begin"
    assert e1 == e2 == 0, "epoch does not move on response begin"
    s.end_response()
    assert s.epoch == 0, "epoch does not move on response end"
    new_epoch = s.barge_in()
    assert new_epoch == 1, "epoch advances only on barge-in"
    assert s.is_stale(e1) and not s.is_stale(new_epoch)


class _TurnAdapter(DuplexAdapter):
    """Emits chunks until cancelled; counts respond() lifecycles."""

    def __init__(self):
        self.responses = 0
        self.barge_ins = 0

    def capabilities(self) -> DuplexCapability:
        return DuplexCapability(
            input_modalities=frozenset({"audio"}),
            output_modalities=frozenset({"audio"}),
            proactive=True,
        )

    async def on_input(self, session, modality, data) -> None:
        pass

    def should_respond(self, session) -> bool:
        return True

    async def on_barge_in(self, session) -> None:
        self.barge_ins += 1

    async def respond(self, session):
        self.responses += 1
        for i in range(100):
            await asyncio.sleep(0.001)
            yield OutputChunk("audio", i)


async def _drive(runtime, events):
    async def gen():
        for e in events:
            await asyncio.sleep(0.005)
            yield e

    out = []

    async def emit(e):
        out.append(e)

    await runtime.run(gen(), emit)
    return out


def test_turn_mode_supersede_and_stale_drop():
    adapter = _TurnAdapter()
    session = DuplexSession(session_id="test", config=DuplexSessionConfig(proactive=True))
    runtime = DuplexRuntime(session, adapter)
    out = asyncio.run(
        _drive(
            runtime,
            [
                {"type": ev.RESPONSE_CREATE},
                {"type": ev.RESPONSE_CREATE},  # supersedes the first
                {"type": ev.CLOSE},
            ],
        )
    )
    assert adapter.responses == 2
    assert adapter.barge_ins == 1, "superseding an in-flight response barges in"
    created = [e for e in out if e["type"] == ev.RESPONSE_CREATED]
    done = [e for e in out if e["type"] == ev.RESPONSE_DONE]
    assert len(created) == 2
    assert len(done) == 1, "stale first response must not emit done"
    assert done[0]["response_index"] == created[1]["response_index"]
    assert session.state is DuplexState.CLOSED
