# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qwen summary publication, source validity and bounded background planning."""

import asyncio
import threading
from dataclasses import dataclass, replace

import pytest

from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig
from vllm_omni.engine.duplex.session.context_ledger import DuplexContextLedger, DuplexContextLevel, DuplexSourceRef
from vllm_omni.engine.duplex.session.history_snapshot import DuplexHistorySnapshot
from vllm_omni.model_executor.models.qwen3_omni.duplex.context_budget import CONTEXT_OMITTED
from vllm_omni.model_executor.models.qwen3_omni.duplex.context_memory import QwenContextMemoryMixin
from vllm_omni.model_executor.models.qwen3_omni.duplex.session import QwenDuplexSessionState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class Processor:
    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(str(m["content"]) for m in messages)

    def tokenizer(self, *, text, **kwargs):
        return {"input_ids": [list(range(len(value))) for value in text]}


class Plugin(QwenContextMemoryMixin):
    def __init__(self, **config):
        self._context_config = DuplexSessionRuntimeConfig(qwen_context_policy="budget_summary", **config)
        self.processor = Processor()
        self.max_model_len = 16384


def snapshot(*texts):
    messages = tuple({"role": "user" if i % 2 == 0 else "assistant", "content": text} for i, text in enumerate(texts))
    return DuplexHistorySnapshot(messages, (None,) * len(texts), tuple(range(len(texts))), (0,) * len(texts), 1)


def observe(plugin, state, snap, omitted=None):
    memory = state.context_memory
    ledger = memory.ledger if memory.ledger is not None else DuplexContextLedger()
    refs = tuple(DuplexSourceRef(s, r) for s, r in zip(snap.source_ids, snap.source_revisions, strict=True))
    plugin.observe_history_context(
        snapshot=snap,
        ledger=ledger,
        state=state,
        runtime_config={CONTEXT_OMITTED: refs if omitted is None else omitted},
    )


async def prepare(plugin, state):
    return await plugin.prepare_auxiliary_request(history=(), runtime_config={}, state=state)


@dataclass(frozen=True)
class SummaryCompletion:
    text: str
    token_ids: tuple[int, ...]
    finish_reason: str


@dataclass(frozen=True)
class SummaryOutput:
    outputs: tuple[SummaryCompletion, ...]


def output(text="user likes tea", count=5, finish_reason="stop"):
    return SummaryOutput((SummaryCompletion(text, tuple(range(count)), finish_reason),))


def policy(plugin, state):
    return plugin.auxiliary_policy(state=state, history_revision=1, history=())


@pytest.mark.asyncio
async def test_summary_is_private_bounded_and_replaces_flattened_sources():
    plugin, state = Plugin(qwen_summary_timeout_s=0.25), QwenDuplexSessionState()
    snap = snapshot("tea", "noted", "current", "answer")
    refs = (DuplexSourceRef(0, 0), DuplexSourceRef(1, 0))
    observe(plugin, state, snap, refs)
    request = await prepare(plugin, state)
    assert request.stage_id == 0 and request.timeout_s == 0.25
    assert request.retry_delays_s == () and request.retry_cooldown_s is None
    assert request.sampling_params.temperature == 0 and request.sampling_params.max_tokens == 512
    assert plugin.apply_auxiliary_result(request=request, output=output(), state=state)
    old = state.context_memory.summary
    ledger = state.context_memory.ledger
    unrelated = ledger.add_node(
        snap, level=DuplexContextLevel.MEDIA_DROPPED, sources=(DuplexSourceRef(2, 0),), message=snap.messages[2]
    )
    observe(plugin, state, snap)
    next_request = await prepare(plugin, state)
    assert old.text in next_request.prompt["prompt"]
    assert plugin.apply_auxiliary_result(request=next_request, output=output("updated"), state=state)
    assert not ledger.valid(old.node_id) and ledger.valid(unrelated.node_id)
    summary = state.context_memory.summary
    assert summary.sources == tuple(DuplexSourceRef(i, 0) for i in range(4))
    assert [node.parents for node in ledger.nodes() if node.level == DuplexContextLevel.SUMMARY] == [()]
    assert not plugin.apply_auxiliary_result(request=request, output=output("stale"), state=state)
    assert policy(plugin, state) is None


@pytest.mark.parametrize("change", ["revision", "delete", "truncate"])
@pytest.mark.asyncio
async def test_source_changes_reject_delayed_results_and_invalidate_published_summary(change):
    plugin, state = Plugin(), QwenDuplexSessionState()
    snap = snapshot("tea", "noted", "next", "done")
    observe(plugin, state, snap)
    request = await prepare(plugin, state)
    assert plugin.apply_auxiliary_result(request=request, output=output(), state=state)
    node = state.context_memory.summary.node_id
    if change == "revision":
        revised = replace(snap, source_revisions=(1, 0, 0, 0), history_revision=2)
    else:
        indices = (1, 2, 3) if change == "delete" else (0, 1)
        revised = DuplexHistorySnapshot(
            tuple(snap.messages[i] for i in indices), (None,) * len(indices), indices, (0,) * len(indices), 2
        )
    observe(plugin, state, revised)
    assert state.context_memory.summary is None
    assert not state.context_memory.ledger.valid(node)
    assert not plugin.apply_auxiliary_result(request=request, output=output(), state=state)


@pytest.mark.asyncio
async def test_unrelated_append_preserves_source_and_failure_deduplication():
    plugin, state = Plugin(), QwenDuplexSessionState()
    snap = snapshot("tea", "noted")
    refs = (DuplexSourceRef(0, 0), DuplexSourceRef(1, 0))
    observe(plugin, state, snap)
    key = policy(plugin, state).source_key
    request = await prepare(plugin, state)
    observe(plugin, state, snapshot("tea", "noted", "new", "reply"), refs)
    assert plugin.auxiliary_source_matches(request=request, state=state, history=())
    assert not plugin.apply_auxiliary_result(request=request, output=output(""), state=state)
    assert policy(plugin, state) is None
    assert await prepare(plugin, state) is None
    observe(plugin, state, snapshot("tea", "noted", "new", "reply"))
    assert policy(plugin, state).source_key != key
    assert await prepare(plugin, state) is not None


@pytest.mark.parametrize("result", [output(""), output(count=513), output(finish_reason="length"), output(count=0)])
@pytest.mark.asyncio
async def test_invalid_outputs_leave_ledger_untouched(result):
    plugin, state = Plugin(), QwenDuplexSessionState()
    observe(plugin, state, snapshot("tea", "noted"))
    request = await prepare(plugin, state)
    assert not plugin.apply_auxiliary_result(request=request, output=result, state=state)
    assert state.context_memory.ledger.nodes() == ()


@pytest.mark.parametrize("text", ["OK", " \nOkay! \n", "收到。", "好的！", "NO_RELEVANT_MEMORY."])
@pytest.mark.parametrize("existing_summary", [False, True])
@pytest.mark.asyncio
async def test_acknowledgment_or_no_memory_result_cannot_publish_or_replace_facts(text, existing_summary):
    plugin, state = Plugin(), QwenDuplexSessionState()
    snap = snapshot(
        "The access code is 4271. Reply only OK.",
        "OK",
        "The meeting room is Cedar. Reply only OK.",
        "OK",
    )
    if existing_summary:
        observe(plugin, state, snap, (DuplexSourceRef(0, 0), DuplexSourceRef(1, 0)))
        first = await prepare(plugin, state)
        assert plugin.apply_auxiliary_result(request=first, output=output("Code: 4271."), state=state)
    observe(plugin, state, snap)
    request = await prepare(plugin, state)
    assert request is not None
    memory = state.context_memory
    previous_summary, previous_nodes = memory.summary, memory.ledger.nodes()

    assert not plugin.apply_auxiliary_result(request=request, output=output(text), state=state)

    assert memory.summary == previous_summary
    assert memory.ledger.nodes() == previous_nodes
    if previous_summary is not None:
        assert memory.ledger.valid(previous_summary.node_id)
    assert policy(plugin, state) is None


@pytest.mark.parametrize("text", ["Code: 4271.", "房间：Cedar。", "OK is the user's preferred status label."])
@pytest.mark.asyncio
async def test_short_factual_summaries_are_accepted(text):
    plugin, state = Plugin(), QwenDuplexSessionState()
    snap = snapshot("Access code 4271; room Cedar; preferred status label OK. Reply only OK.", "OK")
    observe(plugin, state, snap)
    request = await prepare(plugin, state)

    assert plugin.apply_auxiliary_result(request=request, output=output(text), state=state)

    memory = state.context_memory
    assert memory.summary.text == text
    assert memory.ledger.valid(memory.summary.node_id)
    assert memory.summary.sources == (DuplexSourceRef(0, 0), DuplexSourceRef(1, 0))


@pytest.mark.asyncio
async def test_skip_incomplete_and_textless_media_turns_but_accept_transcript():
    plugin, state = Plugin(), QwenDuplexSessionState()
    snap = snapshot("system", "ignored", "placeholder", "unseen", "last")
    messages = (
        {"role": "system", "content": "instructions"},
        {"role": "user", "content": [{"type": "audio_url", "audio_url": "media"}]},
        {"role": "assistant", "content": "must not summarize"},
        {"role": "user", "content": [{"type": "audio_url", "transcript": "heard fact"}]},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "unfinished"},
    )
    snap = replace(snap, messages=messages, source_ids=tuple(range(6)), source_revisions=(0,) * 6, item_ids=(None,) * 6)
    observe(plugin, state, snap)
    request = await prepare(plugin, state)
    assert request.model_source.sources == (DuplexSourceRef(3, 0), DuplexSourceRef(4, 0))
    prompt = request.prompt["prompt"]
    assert "heard fact" in prompt
    assert "must not summarize" not in prompt and "unfinished" not in prompt


@pytest.mark.asyncio
async def test_oversized_single_turn_is_not_truncated_and_next_turn_can_progress():
    plugin, state = Plugin(qwen_summary_input_max_tokens=1000), QwenDuplexSessionState()
    observe(plugin, state, snapshot("x" * 2000, "large reply", "small fact", "ack"))
    request = await prepare(plugin, state)
    assert len(request.prompt["prompt"]) <= 1000
    assert "small fact" in request.prompt["prompt"] and "large reply" not in request.prompt["prompt"]
    assert request.model_source.sources == (DuplexSourceRef(2, 0), DuplexSourceRef(3, 0))


@pytest.mark.parametrize("timeout", [False, True])
@pytest.mark.asyncio
async def test_cancelled_or_timed_out_preparation_cannot_mutate_ledger(timeout, monkeypatch):
    plugin, state = Plugin(), QwenDuplexSessionState()
    observe(plugin, state, snapshot("tea", "noted"))
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    original = plugin.processor.apply_chat_template

    def blocked(*args, **kwargs):
        started.set()
        try:
            assert release.wait(2)
            return original(*args, **kwargs)
        finally:
            finished.set()

    monkeypatch.setattr(plugin.processor, "apply_chat_template", blocked)
    task = asyncio.create_task(prepare(plugin, state))
    try:
        assert await asyncio.to_thread(started.wait, 2)
        if timeout:
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(task, 0.01)
        else:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
    finally:
        release.set()
        assert await asyncio.to_thread(finished.wait, 2)
    assert state.context_memory.ledger.nodes() == ()
    assert state.context_memory.summary is None
    assert policy(plugin, state) is None


@pytest.mark.asyncio
async def test_sessions_do_not_share_summary_state():
    plugin, first, second = Plugin(), QwenDuplexSessionState(), QwenDuplexSessionState()
    for state in (first, second):
        observe(plugin, state, snapshot("tea", "noted"))
    request = await prepare(plugin, first)
    assert plugin.apply_auxiliary_result(request=request, output=output(), state=first)
    assert second.context_memory.summary is None and second.context_memory.ledger.nodes() == ()
    assert await prepare(plugin, second) is not None


@pytest.mark.asyncio
async def test_edit_during_generation_rejects_unpublished_result():
    plugin, state = Plugin(), QwenDuplexSessionState()
    snap = snapshot("old fact", "noted", "current", "answer")
    observe(plugin, state, snap, (DuplexSourceRef(0, 0), DuplexSourceRef(1, 0)))
    request = await prepare(plugin, state)
    revised = replace(snap, source_revisions=(1, 0, 0, 0), history_revision=2)
    observe(plugin, state, revised)
    assert not plugin.auxiliary_source_matches(request=request, state=state, history=())
    assert not plugin.apply_auxiliary_result(request=request, output=output(), state=state)
    assert state.context_memory.ledger.nodes() == ()


@pytest.mark.asyncio
async def test_unrelated_revision_does_not_invalidate_selected_sources():
    plugin, state = Plugin(), QwenDuplexSessionState()
    snap = snapshot("old fact", "noted", "current", "answer")
    omitted = (DuplexSourceRef(0, 0), DuplexSourceRef(1, 0))
    observe(plugin, state, snap, omitted)
    request = await prepare(plugin, state)
    observe(plugin, state, replace(snap, source_revisions=(0, 0, 1, 0), history_revision=2), omitted)
    assert plugin.auxiliary_source_matches(request=request, state=state, history=())
    assert plugin.apply_auxiliary_result(request=request, output=output(), state=state)
    assert state.context_memory.summary.sources == omitted


@pytest.mark.asyncio
async def test_generation_reserve_limits_summary_input():
    plugin, state = Plugin(qwen_context_max_tokens=1000, qwen_summary_max_tokens=400), QwenDuplexSessionState()
    observe(plugin, state, snapshot("a" * 100, "b" * 100))
    assert await prepare(plugin, state) is None
    assert state.context_memory.ledger.nodes() == ()
