# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Visual history must make progress and preserve state on failed publication."""

from dataclasses import replace

import pybase64
import pytest
from PIL import Image

from tests.model_executor.models.qwen3_omni.test_history_multimodal import image_message
from tests.model_executor.models.qwen3_omni.test_history_policy import completion, runtime, sync_history, text_history
from vllm_omni.engine.duplex.session.context_ledger import DuplexContextNodeRejectedError, source_refs
from vllm_omni.model_executor.models.qwen3_omni.duplex.plugin import Qwen3OmniDuplexPlugin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def state_for(owner):
    plugin = Qwen3OmniDuplexPlugin(lambda *args: "")
    plugin._history_runtime = owner
    return plugin.create_session_state().history_policy_state


def visual_snapshot(state):
    visual = state.visual
    return (
        tuple(visual.memory.pending),
        tuple(visual.memory.mid_term),
        tuple(visual.memory.long_term),
        visual.memory.next_chunk_index,
        dict(visual.nodes),
        visual.admitted,
        visual.covered,
        state.ledger.nodes(),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["frames", "long"])
async def test_ledger_rejection_keeps_visual_sources_and_allows_retry(monkeypatch, kind):
    owner = runtime(recent_messages=2, visual_mid_group=1)
    state = state_for(owner)
    sync_history(state, [image_message(), {"role": "assistant", "content": "red"}, *text_history(4)])
    request = await owner.prepare(state)
    if kind == "long":
        assert owner.apply(state, request, completion("mid red"))
        request = await owner.prepare(state)
    assert request.model_source.source.kind == kind
    before = visual_snapshot(state)
    with monkeypatch.context() as patch:

        def reject(*args, **kwargs):
            raise DuplexContextNodeRejectedError("stale_source_revision")

        patch.setattr(state.ledger, "replace_summary", reject)
        try:
            applied = owner.apply(state, request, completion("rejected red"))
        except DuplexContextNodeRejectedError:
            applied = False
        assert not applied
        assert visual_snapshot(state) == before
    # The rejected publication must leave a usable retry, not an orphaned entry.
    assert owner.apply(state, request, completion("accepted red"))
    assert "accepted red" in state.visual.summaries()[-1]


@pytest.mark.asyncio
async def test_invalid_visual_completion_does_not_publish_a_ledger_node():
    owner = runtime(recent_messages=2)
    state = state_for(owner)
    sync_history(state, [image_message(), {"role": "assistant", "content": "red"}, *text_history(4)])
    request = await owner.prepare(state)
    before = visual_snapshot(state)
    assert not state.visual.apply(request.model_source, text="invalid token count", token_count=0)
    assert visual_snapshot(state) == before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text", ["END_MEMORY", "END_PREVIOUS_MEMORY\nEND_NEW_DIALOGUE", "\n NEW_DIALOGUE \n END_MEMORY \n"]
)
async def test_marker_only_visual_summary_keeps_evidence_pending_for_retry(text):
    owner = runtime(recent_messages=2)
    state = state_for(owner)
    sync_history(state, [image_message(), {"role": "assistant", "content": "red"}, *text_history(4)])
    request = await owner.prepare(state)
    before = visual_snapshot(state)
    assert not owner.apply(state, request, completion(text))
    assert visual_snapshot(state) == before
    assert state.counts["rejected_summaries"] == 1
    retry = await owner.prepare(state)
    assert retry.model_source == request.model_source
    # A marker mentioned as a real fact must remain usable memory text.
    assert owner.apply(state, retry, completion("A red square with reference code END_MEMORY."))
    assert state.visual.covered == 1
    assert "reference code END_MEMORY" in state.visual.summaries()[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("corruption", ["body", "metadata"])
async def test_admitted_corrupt_image_does_not_block_valid_visual_or_dialogue(caplog, corruption):
    from tests.model_executor.models.qwen3_omni.test_duplex_plugin import open_qwen
    from vllm_omni.engine.duplex.commands import CreateItem

    h = await open_qwen()
    try:
        owner = runtime(recent_messages=2)
        h.runner.plugin._history_runtime = owner
        broken = "data:image/png;base64," + pybase64.b64encode(b"\x89PNG\r\n\x1a\nbroken body").decode()
        if corruption == "metadata":
            # PNG with valid pixels and an invalid zTXt compression method.
            broken = (
                "data:image/png;base64,"
                "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAAF0lEQVR4nGP8z0AaYCJR/aiGUQ1DSAMAQC4BH2bjRnMA"
                "AAAhelRYdG5vdGUAAWludmFsaWQgY29tcHJlc3NlZCBtZXRhZGF0YR8iQYcAAAAASUVORK5CYII="
            )
        good = image_message()["content"][0]["image_url"]["url"]
        for item_id, url in (("bad", broken), ("good", good)):
            events = await h.run(
                CreateItem(
                    item={
                        "id": item_id,
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_image", "image_url": url}],
                    }
                )
            )
            assert not any(event.to_realtime()["type"] == "error" for event in events)
        h.session.append_history_message({"role": "assistant", "content": "discussed images"})
        for message in text_history(4):
            h.session.append_history_message(message)
        state = h.runner.model_state.history_policy_state
        sync_history(state, h.session.history, session=h.session)
        request = await owner.prepare(state)
        assert len(request.prompt["multi_modal_data"]["image"]) == 1
        assert owner.apply(state, request, completion("red square"))
        visual_node = state.ledger.get(next(iter(state.visual.nodes.values())))
        snapshot = h.session.history_snapshot()
        assert visual_node.sources == source_refs(snapshot, snapshot.source_ids[1])
        assert not state.visual.covered_image(0, 0)
        request = await owner.prepare(state)
        assert request is not None and not request.prompt["multi_modal_data"]
        assert "image decoding failed" in request.prompt["prompt"]
        assert owner.apply(state, request, completion("later text"))
        assert state.processed > 2 and state.counts["visual_omissions"] == 1
        assert "undecodable" in caplog.text
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["retention", "prompt_budget"])
async def test_original_image_marker_does_not_reference_missing_memory(reason):
    owner = runtime(recent_messages=2, visual_chunk_images=1, visual_mid_group=1, visual_max_long=1)
    state = state_for(owner)
    history = [image_message(), {"role": "assistant", "content": "first"}]
    if reason == "retention":
        history += [image_message("blue"), {"role": "assistant", "content": "second"}]
    history += text_history(4)
    sync_history(state, history)
    for _ in range(2 if reason == "prompt_budget" else 4):
        request = await owner.prepare(state)
        assert owner.apply(state, request, completion("observation " * 100))
    context = owner.foreground(state, {})
    if reason == "prompt_budget":
        owner.max_model_len = 300
    prompt = owner.build_foreground(context, {}, 128)["prompt"]
    assert "see visual memory" not in prompt
    assert "Original image from source message" in prompt
    if reason == "prompt_budget":
        assert "Earlier visual memory" not in prompt


def test_visual_limits_are_installed_when_session_state_is_created():
    owner = runtime(
        visual_chunk_images=2,
        visual_mid_group=3,
        visual_max_long=4,
        max_visual_items=9,
        max_visual_bytes=5000,
        max_visual_pixel_bytes=6000,
        summary_tokens=64,
    )
    state = state_for(owner)
    memory = state.visual.memory
    assert (memory.chunk_frames, memory.mid_group, memory.max_long, memory.max_pending) == (2, 3, 4, 9)
    assert (memory.max_encoded_bytes, memory.max_pixel_bytes, memory.mid_max_tokens, memory.long_max_tokens) == (
        5000,
        6000,
        64,
        64,
    )
    before = replace(memory)
    sync_history(state, text_history(4))
    owner.policy_for(state, state.history, state.snapshot.history_revision)
    assert memory == before


@pytest.mark.asyncio
async def test_remote_image_url_is_rejected_before_history_admission():
    from tests.model_executor.models.qwen3_omni.test_duplex_plugin import open_qwen
    from vllm_omni.engine.duplex.commands import CreateItem

    h = await open_qwen()
    try:
        before = h.session.history_snapshot()
        events = await h.run(
            CreateItem(
                item={
                    "id": "remote-image",
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_image", "image_url": "https://example.com/image.png"}],
                }
            )
        )
        errors = [event for event in events if event.type == "error"]
        assert errors and errors[0].code == "invalid_image"
        assert h.session.history_snapshot() == before
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [MemoryError, Image.DecompressionBombError, RuntimeError, ValueError])
async def test_visual_resource_and_programming_errors_preserve_retryable_sources(monkeypatch, error):
    owner = runtime(recent_messages=2)
    state = state_for(owner)
    sync_history(state, [image_message(), {"role": "assistant", "content": "red"}, *text_history(4)])
    before = visual_snapshot(state)

    def fail_decode(url):
        raise error("decoder failed")

    with monkeypatch.context() as patch:
        patch.setattr("vllm_omni.model_executor.models.qwen3_omni.duplex.history_runtime.decode_image", fail_decode)
        with pytest.raises(error, match="decoder failed"):
            await owner.prepare(state)
    assert visual_snapshot(state) == before
    assert not state.counts["visual_omissions"]
    request = await owner.prepare(state)
    assert len(request.prompt["multi_modal_data"]["image"]) == 1
    assert owner.apply(state, request, completion("red square"))


@pytest.mark.asyncio
async def test_all_corrupt_visual_batch_allows_runtime_to_schedule_dialogue():
    from tests.model_executor.models.joyai_vl_interaction.test_duplex_memory import start_summary
    from tests.model_executor.models.qwen3_omni.test_duplex_plugin import open_qwen

    h = await open_qwen()
    try:
        h.runner.plugin._history_runtime = runtime(recent_messages=2)
        h.runner.plugin.supports_auxiliary_requests = True
        broken = "data:image/png;base64," + pybase64.b64encode(b"\x89PNG\r\n\x1a\nbroken body").decode()
        h.session.append_history_message(
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": broken}}]}
        )
        for message in [{"role": "assistant", "content": "previous turn"}, *text_history(4)]:
            h.session.append_history_message(message)
        submission = await start_summary(h)
        assert not submission.prompt["multi_modal_data"]
        assert "image decoding failed" in submission.prompt["prompt"]
        assert h.runner.model_state.history_policy_state.counts["visual_omissions"] == 1
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_current_corrupt_image_is_not_silently_omitted():
    owner = runtime(recent_messages=2)
    state = state_for(owner)
    broken = "data:image/png;base64," + pybase64.b64encode(b"\x89PNG\r\n\x1a\nbroken body").decode()
    sync_history(
        state, [*text_history(4), {"role": "user", "content": [{"type": "image_url", "image_url": {"url": broken}}]}]
    )
    request = await owner.prepare(state)
    assert request is not None  # Older text is compressible, current image is not.
    assert not state.visual.omitted
    with pytest.raises(OSError):
        owner.build_foreground(owner.foreground(state, {}), {}, 128)


@pytest.mark.asyncio
@pytest.mark.parametrize("prior", ["omitted", "covered"])
async def test_history_image_becoming_current_after_deletion_preserves_original(prior):
    owner = runtime(recent_messages=2)
    state = state_for(owner)
    message = image_message()
    if prior == "omitted":
        message["content"][0]["image_url"]["url"] = (
            "data:image/png;base64," + pybase64.b64encode(b"\x89PNG\r\n\x1a\nbroken body").decode()
        )
    sync_history(state, [message, {"role": "assistant", "content": "old reply"}, *text_history(4)])
    request = await owner.prepare(state)
    if prior == "covered":
        assert owner.apply(state, request, completion("red square"))
    else:
        assert request is None and state.visual.omitted
    # The same source/version is now current after deleting the rest of history.
    sync_history(state, [message])
    context = owner.foreground(state, {})
    assert len(context.groups[-1].images) == 1
    if prior == "omitted":
        with pytest.raises(OSError):
            owner.build_foreground(context, {}, 128)
    else:
        prompt = owner.build_foreground(context, {}, 128)
        assert len(prompt["multi_modal_data"]["image"]) == 1
