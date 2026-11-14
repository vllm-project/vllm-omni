# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qwen L2 lifecycle against real Session source identity and shared authority."""

import pytest

from tests.model_executor.models.qwen3_omni.test_history_multimodal import image_message
from tests.model_executor.models.qwen3_omni.test_history_policy import completion, runtime, sync_history, text_history
from vllm_omni.engine.duplex.config import DuplexContextViewPolicy, DuplexSessionConfig
from vllm_omni.engine.duplex.session.context_ledger import DuplexContextLevel, source_refs
from vllm_omni.engine.duplex.session.engine_session import DuplexEngineSession

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def setup_history(messages, *, owner):
    session = DuplexEngineSession(session_id="qwen-ledger", config=DuplexSessionConfig(model="test"))
    for index, message in enumerate(messages):
        session.register_history_item(str(index), message)
    state = owner.create_state()
    refresh(state, session)
    return state, session


def refresh(state, session):
    sync_history(state, session.history, session=session)


def summaries(session):
    return tuple(n for n in session.context_ledger.nodes() if n.level is DuplexContextLevel.SUMMARY)


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["edit", "delete_recreate", "truncate"])
async def test_rolling_summary_inherits_earliest_sources_and_rejects_late_result(mutation):
    owner = runtime(recent_messages=2, chunk_messages=2)
    state, session = setup_history(text_history(1) if mutation == "truncate" else text_history(10), owner=owner)
    if mutation == "truncate":
        response_id = session.begin_response()
        session.append_assistant_text("hello worlds")
        session.mark_audio_sent(duration_ms=2000, text_chars=12)
        session.acknowledge_playback(played_ms=2000)
        item_id = f"item_{response_id}"
        session.register_history_item(item_id, session.end_response())
        session.truncate_history_item(item_id, audio_end_ms=1999, hard=True)
        for index, message in enumerate(text_history(10)[2:], 2):
            session.register_history_item(str(index), message)
        refresh(state, session)
    for index in range(3):
        request = await owner.prepare(state)
        assert owner.apply(state, request, completion(f"memory {index}"))
        (node,) = summaries(session)
        assert node.node_id == state.summary_node_id and not node.parents
        assert set(node.sources) == set(
            source_refs(session.history_snapshot(), *session.history_snapshot().source_ids[: 2 * (index + 1)])
        )
    pending = await owner.prepare(state)
    if mutation == "edit":
        session.register_history_item("0", {"role": "user", "content": "corrected budget 5000"})
    elif mutation == "delete_recreate":
        session.delete_history_item("0")
        session.register_history_item("0", text_history(1)[0])
    else:
        before = session.history_snapshot()
        session.truncate_history_item(item_id, audio_end_ms=1998, hard=True)
        assert session.history_snapshot().messages == before.messages
        assert session.history_snapshot().source_revisions != before.source_revisions
    refresh(state, session)
    assert not summaries(session) and state.summary == "" and state.processed == 0
    assert not owner.apply(state, pending, completion("stale rolling result"))


@pytest.mark.asyncio
async def test_append_preserves_pending_sources_and_ledger_is_summary_authority():
    owner = runtime(recent_messages=2, chunk_messages=2)
    state, session = setup_history(text_history(6), owner=owner)
    request = await owner.prepare(state)
    session.register_history_item("tail", {"role": "user", "content": "new budget is 5000"})
    refresh(state, session)
    assert owner.apply(state, request, completion("budget 3000"))
    (node,) = summaries(session)
    assert node.sources == source_refs(session.history_snapshot(), *session.history_snapshot().source_ids[:2])
    context = owner.foreground(state, {})
    assert context.summary == node.message["content"]
    prompt = owner.build_foreground(context, {}, 128)["prompt"]
    assert "new budget is 5000" in prompt and "Later messages and corrections take precedence" in prompt
    session.context_ledger.remove(node.node_id)
    refresh(state, session)
    assert state.summary == "" and state.processed == 0
    assert owner.foreground(state, {}).summary == ""


@pytest.mark.asyncio
async def test_omitted_audio_is_not_claimed_as_summary_evidence():
    from tests.model_executor.models.qwen3_omni.test_history_policy import audio_message

    missing, _ = audio_message(0)
    owner = runtime(recent_messages=2, chunk_messages=2)
    state, session = setup_history([missing, {"role": "assistant", "content": "heard"}, *text_history(6)], owner=owner)
    request = await owner.prepare(state)
    assert state.processed == 2 and request.model_source.start == 2
    assert list(state.omissions) == [(0, 2, "source_unavailable")]
    assert "heard" not in request.prompt["prompt"]
    assert owner.apply(state, request, completion("text facts"))
    assert state.processed > 2
    (node,) = summaries(session)
    assert node.sources == source_refs(session.history_snapshot(), *session.history_snapshot().source_ids[2:4])


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["caption", "delete"])
async def test_visual_merge_inherits_sources_l1_coexists_and_old_nodes_are_cleaned(mutation):
    owner = runtime(recent_messages=2, visual_chunk_images=1, visual_mid_group=2, visual_max_long=1)
    messages = []
    for color in ("red", "blue", "green", "yellow"):
        messages.extend([image_message(color), {"role": "assistant", "content": color}])
    state, session = setup_history([*messages, *text_history(4)], owner=owner)
    session.render_history_view(DuplexContextViewPolicy(keep_media_user_turns=0))
    l1_ids = {n.node_id for n in session.context_ledger.nodes() if n.level is DuplexContextLevel.MEDIA_DROPPED}
    for index in range(2):
        for _ in range(2):
            request = await owner.prepare(state)
            assert request.model_source.source.kind == "frames"
            assert owner.apply(state, request, completion("visual mid"))
        request = await owner.prepare(state)
        assert request.model_source.source.kind == "long"
        assert owner.apply(state, request, completion(f"visual long {index}"))
        (node,) = summaries(session)
        expected = session.history_snapshot().source_ids[4 * index : 4 * index + 4 : 2]
        assert set(node.sources) == set(source_refs(session.history_snapshot(), *expected))
        assert not state.visual.memory.long_term[0].text  # No second authoritative summary string.
    assert l1_ids <= {n.node_id for n in session.context_ledger.nodes()}
    assert "visual long 1" in owner.foreground(state, {}).visual_summaries[0]
    if mutation == "caption":
        # Identical pixels with a changed caption still revoke derived memory.
        changed = image_message("green")
        changed["content"].append({"type": "text", "text": "corrected caption"})
        session.register_history_item("4", changed)
    else:
        session.delete_history_item("4")
    refresh(state, session)
    assert not summaries(session) and not state.visual.summaries()
    assert state.visual.covered == 0


@pytest.mark.asyncio
async def test_mailbox_checks_current_snapshot_before_registering_result():
    from tests.model_executor.models.joyai_vl_interaction.test_duplex_memory import start_summary, summary_output
    from tests.model_executor.models.qwen3_omni.test_duplex_plugin import open_qwen

    h = await open_qwen()
    try:
        h.runner.plugin._history_runtime = runtime(recent_messages=2, chunk_messages=2)
        h.runner.plugin.supports_auxiliary_requests = True
        for index, message in enumerate(text_history(6)):
            h.session.register_history_item(str(index), message)
        submission = await start_summary(h)
        # Mutation happens after preparation/submission, before mailbox writeback.
        h.session.register_history_item("0", {"role": "user", "content": "corrected budget 5000"})
        h.deliver(summary_output(submission.context.request_id, "stale budget 3000"), stage_id=0, segment_finished=True)
        await h.settle()
        state = h.runner.model_state.history_policy_state
        assert state.ledger is h.session.context_ledger
        assert state.snapshot.history_revision == h.session.history_revision
        assert not summaries(h.session) and not state.summary
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_foreground_refreshes_ledger_before_reading_summary():
    from tests.model_executor.models.qwen3_omni.test_duplex_plugin import open_qwen
    from vllm_omni.engine.duplex.commands import CreateItem, CreateResponse

    h = await open_qwen()
    try:
        owner = runtime(recent_messages=2, chunk_messages=2)
        h.runner.plugin._history_runtime = owner
        for index, message in enumerate(text_history(6)):
            h.session.register_history_item(str(index), message)
        state = h.runner.model_state.history_policy_state
        refresh(state, h.session)
        request = await owner.prepare(state)
        assert owner.apply(state, request, completion("obsolete unique summary"))
        h.session.register_history_item("0", {"role": "user", "content": "corrected budget 5000"})
        await h.run(
            CreateItem(
                item={
                    "id": "tail",
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "current question"}],
                }
            )
        )
        await h.run(CreateResponse())
        assert h.port.submissions
        prompt = h.port.submissions[-1].prompt["prompt"]
        assert "obsolete unique summary" not in prompt
        assert "current question" in prompt
        assert not summaries(h.session) and not state.summary
    finally:
        await h.manager.shutdown()
