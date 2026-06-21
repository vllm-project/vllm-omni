# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm_omni.experimental.fullduplex.joyvl.decision.output_parser import parse_action
from vllm_omni.experimental.fullduplex.joyvl.memory.brain import InteractionBrain

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeSummarizer:
    def __init__(self):
        self.chunks = 0

    async def summarize_chunk(self, chunk_index, frame_range, frames):
        self.chunks += 1
        return f"summary-of-chunk-{chunk_index}"

    async def compress_to_long_term(self, existing, mids):
        return (existing + " | " if existing else "") + f"compressed({len(mids)})"


class _FakeDelegation:
    def __init__(self):
        self._ready = {}

    async def submit(self, question, note, frames):
        self._ready["t1"] = question
        return "t1"

    async def poll(self, task_id):
        from vllm_omni.experimental.fullduplex.joyvl.bridges.delegation import DelegationResult

        return DelegationResult(task_id, "ready", digest=f"answer to {self._ready[task_id]}")


def test_query_freshness_and_archive_on_change():
    b = InteractionBrain()
    assert b.update_query("count bottles") is True
    assert b.update_query("count bottles") is False
    b.record_response("1 bottle")
    assert b.update_query("now describe") is True
    assert b.memory.qa_history[-1].query == "count bottles"
    assert b.memory.qa_history[-1].responses == [("0.0s", "1 bottle")]


def test_query_in_frame_on_arrival_then_prefix_on_later_chunks():
    b = InteractionBrain(frame_seconds=1.0)
    b.tick(3)
    b.update_query("alert me if a fire breaks out")
    assert b.query_in_current_chunk is True
    assert "alert me if a fire breaks out" not in b.build_prefix()
    b.record_response("a fire is breaking out")
    b.close_chunk()
    assert b.query_in_current_chunk is False
    prefix = b.build_prefix()
    assert "alert me if a fire breaks out" in prefix
    assert "a fire is breaking out" in prefix


@pytest.mark.asyncio
async def test_flush_archives_qa_and_summarizes():
    summ = _FakeSummarizer()
    b = InteractionBrain(summarizer=summ, chunk_frames=3, long_term_every_n_chunks=2, frame_seconds=1.0)
    b.update_query("watch the room")
    b.tick(3)
    assert b.should_flush() is True
    b.record_response("a person enters")
    await b.flush([("0.0s", "u0"), ("2.0s", "u2")])
    assert summ.chunks == 1
    assert b.chunk_index == 2
    assert b.should_flush() is False
    assert b.memory.qa_history[-1].responses == [("2.0s", "a person enters")]
    assert len(b.memory.mid_term_summaries) == 1


@pytest.mark.asyncio
async def test_long_term_compression_after_n_chunks():
    summ = _FakeSummarizer()
    b = InteractionBrain(summarizer=summ, chunk_frames=1, long_term_every_n_chunks=2, frame_seconds=1.0)
    for _ in range(2):
        await b.flush([("0.0s", "u")])
    assert b.memory.long_term_memory.startswith("compressed(")
    assert b.memory.mid_term_summaries == []


@pytest.mark.asyncio
async def test_delegation_submit_and_fold():
    b = InteractionBrain(delegation=_FakeDelegation())
    action = parse_action("</response> hold on <delegation> solve this")
    info = await b.submit_delegation(action, [])
    assert info["status"] == "submitted"
    folded = await b.fold_delegations()
    assert folded["status"] == "ready"
    assert b.memory.qa_history[-1].responses[0][1] == "answer to solve this"
