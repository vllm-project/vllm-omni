# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Public memory behavior for oversized text and hierarchical visual history."""

import asyncio
import threading
from io import BytesIO

import pybase64
import pytest
from PIL import Image

from tests.model_executor.models.qwen3_omni.test_history_compression import StructuredMemoryPolicy
from tests.model_executor.models.qwen3_omni.test_history_policy import (
    audio_message,
    completion,
    runtime,
    sync_history,
    text_history,
)
from vllm_omni.engine.duplex.plugin import DuplexRuntimeConfigError
from vllm_omni.engine.duplex.session.visual_history import VisualWork

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def image_message(color="red"):
    with BytesIO() as buffer:
        Image.new("RGB", (64, 64), color).save(buffer, format="PNG")
        url = "data:image/png;base64," + pybase64.b64encode(buffer.getvalue()).decode()
    return {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]}


def long_history():
    return [
        {"role": "user", "content": "budget 3000 " + "景点交通 " * 1800 + "USER_TAIL"},
        {"role": "assistant", "content": "建议 " * 1400 + "ASSISTANT_TAIL"},
        *text_history(8),
    ]


@pytest.mark.asyncio
async def test_large_text_turn_folds_all_sources_before_publishing():
    owner = runtime(summary_tokens=64, recent_messages=8)
    state = owner.create_state()
    owner.max_model_len = 1200
    sync_history(state, long_history())
    requests = []
    for _ in range(50):
        request = await owner.prepare(state)
        assert request is not None
        requests.append(request.prompt["prompt"])
        assert len(owner.processor.tokenizer(text=[request.prompt["prompt"]])["input_ids"][0]) + 64 <= 1200
        before = state.partial.offset if state.partial else 0
        assert request.model_source.end_offset > before
        assert owner.apply(state, request, completion("folded budget 3000"))
        if state.processed:
            break
        assert state.summary == ""
        assert owner.foreground(state, {}).summary == ""
        assert state.partial.offset > before
    assert state.processed == 2
    assert state.partial is None and not state.omissions
    assert state.summary == "folded budget 3000"
    assert len(requests) > 2
    assert any("USER_TAIL" in prompt for prompt in requests)
    assert any("ASSISTANT_TAIL" in prompt for prompt in requests)
    assert any("assistant: (continued)" in prompt for prompt in requests)
    assert all("folded budget 3000" in prompt for prompt in requests[1:])


@pytest.mark.asyncio
async def test_partial_text_reject_retry_source_edit_and_late_result():
    owner = runtime(summary_tokens=64)
    state = owner.create_state()
    owner.max_model_len = 1200
    history = long_history()
    sync_history(state, history)
    first = await owner.prepare(state)
    assert owner.apply(state, first, completion("first fold"))
    assert state.partial and not state.summary and not state.ledger.nodes()
    second = await owner.prepare(state)
    partial = state.partial
    assert not owner.apply(state, second, completion("truncated", "length"))
    assert state.partial == partial and state.processed == 0
    retry = await owner.prepare(state)
    assert retry.model_source.end_offset == second.model_source.end_offset
    sync_history(state, [*history, {"role": "user", "content": "another topic"}])
    assert owner.source_matches(state, second.model_source)
    history[0] = {"role": "user", "content": "corrected budget 5000"}
    sync_history(state, history)
    assert state.partial is None and not state.ledger.nodes()
    assert not owner.apply(state, second, completion("stale fold"))


@pytest.mark.asyncio
async def test_text_fold_preserves_custom_compressor_and_parser():
    owner = runtime(summary_tokens=64)
    state = owner.create_state()
    owner.policy = StructuredMemoryPolicy()
    owner.max_model_len = 800
    sync_history(state, long_history())
    request = await owner.prepare(state)
    assert "JSON object" in request.prompt["prompt"]
    assert not owner.apply(state, request, completion("invalid JSON"))
    assert owner.apply(state, request, completion('{"facts":["budget 3000"]}'))
    assert state.partial.text == '{"facts":["budget 3000"]}'
    assert state.summary == ""


@pytest.mark.asyncio
async def test_visual_frames_to_mid_to_long_and_foreground_reuse():
    owner = runtime(summary_tokens=128, recent_messages=2, visual_mid_group=2, visual_chunk_images=1)
    state = owner.create_state()
    history = [
        image_message(),
        {"role": "assistant", "content": "red"},
        image_message("blue"),
        {"role": "assistant", "content": "blue"},
        *text_history(4),
    ]
    sync_history(state, history)
    for color in ("red", "blue"):
        request = await owner.prepare(state)
        assert isinstance(request.model_source, VisualWork)
        assert request.model_source.source.kind == "frames"
        assert len(request.prompt["multi_modal_data"]["image"]) == 1
        assert owner.apply(state, request, completion(f"{color} object"))
    assert state.visual.covered == 2 and not state.visual.memory.pending
    request = await owner.prepare(state)
    assert request.model_source.source.kind == "long"
    assert "red object" in request.prompt["prompt"] and "blue object" in request.prompt["prompt"]
    assert not request.prompt["multi_modal_data"]
    assert owner.apply(state, request, completion("red then blue objects"))
    assert not state.visual.memory.mid_term and len(state.visual.memory.long_term) == 1
    foreground = owner.build_foreground(owner.foreground(state, {}), {}, 128)
    assert "red then blue objects" in foreground["prompt"]
    assert not foreground["multi_modal_data"]
    text_request = await owner.prepare(state)
    assert not isinstance(text_request.model_source, VisualWork)
    assert owner.apply(state, text_request, completion("discussed colors"))


@pytest.mark.asyncio
async def test_visual_failure_retains_sources_edit_revokes_and_append_does_not():
    owner = runtime(recent_messages=2)
    state = owner.create_state()
    history = [image_message(), {"role": "assistant", "content": "red"}, *text_history(4)]
    sync_history(state, history)
    request = await owner.prepare(state)
    assert not owner.apply(state, request, completion("", "length"))
    assert state.visual.covered == 0 and len(state.visual.memory.pending) == 1
    sync_history(state, [*history, {"role": "user", "content": "later"}])
    assert owner.source_matches(state, request.model_source)
    sync_history(state, [image_message("blue"), *history[1:]])
    assert not owner.apply(state, request, completion("stale red"))
    assert state.visual.covered == 0 and not state.visual.memory.pending
    replacement = await owner.prepare(state)
    assert replacement.prompt["multi_modal_data"]["image"][0].getpixel((0, 0)) == (0, 0, 255)


@pytest.mark.asyncio
async def test_visual_capacity_shrinks_batch_without_retiring_unprocessed_frames():
    owner = runtime(summary_tokens=64, recent_messages=2)
    state = owner.create_state()
    owner.processor.image_tokens = 1000
    owner.max_model_len = 1800
    sync_history(
        state,
        [
            image_message(),
            {"role": "assistant", "content": "red"},
            image_message("blue"),
            {"role": "assistant", "content": "blue"},
            *text_history(4),
        ],
    )
    first = await owner.prepare(state)
    assert len(first.prompt["multi_modal_data"]["image"]) == 1
    assert len(state.visual.memory.pending) == 2
    assert owner.apply(state, first, completion("red object"))
    assert state.visual.covered == 1 and len(state.visual.memory.pending) == 1
    second = await owner.prepare(state)
    assert second.prompt["multi_modal_data"]["image"][0].getpixel((0, 0)) == (0, 0, 255)


def test_current_mixed_audio_image_input_and_image_capacity():
    owner = runtime()
    state = owner.create_state()
    message, payload = audio_message(0)
    message["content"].extend(image_message()["content"])
    sync_history(state, [message])
    prompt = owner.build_foreground(owner.foreground(state, payload), {}, 128)
    assert set(prompt["multi_modal_data"]) == {"image", "audio"}
    owner.processor.image_tokens = 10000
    with pytest.raises(DuplexRuntimeConfigError, match="capacity"):
        owner.build_foreground(owner.foreground(state, payload), {}, 128)


def test_image_captions_remain_interleaved_in_foreground():
    owner = runtime()
    state = owner.create_state()
    content = [
        {"type": "text", "text": "before red"},
        *image_message("red")["content"],
        {"type": "text", "text": "before blue"},
        *image_message("blue")["content"],
    ]
    sync_history(state, [{"role": "user", "content": content}])
    group = owner.foreground(state, {}).groups[0]
    assert group.messages[0]["content"] == [
        {"type": "text", "text": "before red"},
        {"type": "image"},
        {"type": "text", "text": "before blue"},
        {"type": "image"},
    ]


@pytest.mark.asyncio
async def test_unfinished_text_fold_continues_after_unrelated_tail_is_deleted():
    owner = runtime(summary_tokens=64)
    state = owner.create_state()
    owner.max_model_len = 1200
    history = long_history()
    sync_history(state, history)
    first = await owner.prepare(state)
    assert owner.apply(state, first, completion("partial"))
    sync_history(state, history[:2])
    second = await owner.prepare(state)
    assert second is not None
    assert second.model_source.end_offset > state.partial.offset


@pytest.mark.asyncio
async def test_visual_preparation_cancellation_joins_worker(monkeypatch):
    owner = runtime(recent_messages=2)
    state = owner.create_state()
    sync_history(state, [image_message(), {"role": "assistant", "content": "red"}, *text_history(4)])
    entered, release = threading.Event(), threading.Event()
    original = owner._decode_visual_batch

    def blocked(work):
        entered.set()
        assert release.wait(10)
        return original(work)

    monkeypatch.setattr(owner, "_decode_visual_batch", blocked)
    task = asyncio.create_task(owner.prepare(state))
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert state.visual.admitted == 0
    assert not state.visual.memory.pending


@pytest.mark.asyncio
async def test_bounded_recent_uses_images_without_creating_memory():
    owner = runtime("bounded_recent")
    state = owner.create_state()
    sync_history(state, [image_message(), {"role": "assistant", "content": "red"}, *text_history(10)])
    assert await owner.prepare(state) is None
    prompt = owner.build_foreground(owner.foreground(state, {}), {}, 128)
    assert len(prompt["multi_modal_data"]["image"]) == 1
    assert state.visual.covered == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("policy,limit", [("recent", 8), ("incremental", 10)])
async def test_session_image_admission_uses_plugin_limits(policy, limit):
    from tests.model_executor.models.qwen3_omni.test_duplex_plugin import open_qwen
    from vllm_omni.engine.duplex.commands import CreateItem

    h = await open_qwen()
    try:
        h.runner.plugin._history_runtime = runtime(policy, max_session_images=limit)
        for index in range(limit + 1):
            url = image_message()["content"][0]["image_url"]["url"]
            events = await h.run(
                CreateItem(
                    item={
                        "id": f"image-{index}",
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_image", "image_url": url}],
                    }
                )
            )
            errors = [event for event in events if event.type == "error"]
            if index < limit:
                assert not errors
            else:
                assert errors and errors[0].code == "input_backpressure"
        assert len(h.session.history) == limit
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_visual_byte_limit_keeps_source_pending_without_omission():
    owner = runtime(recent_messages=2, max_visual_bytes=1)
    state = owner.create_state()
    sync_history(state, [image_message(), {"role": "assistant", "content": "red"}, *text_history(4)])
    assert await owner.prepare(state) is None
    assert state.visual.admitted == state.visual.covered == state.processed == 0
    assert not state.omissions


@pytest.mark.asyncio
async def test_visual_compression_uses_custom_builder_parser_and_reference_guard():
    from dataclasses import replace

    from vllm_omni.model_executor.models.qwen3_omni.duplex.history_policy import CompressionImageRef

    owner = runtime(recent_messages=2)
    state = owner.create_state()
    owner.policy = StructuredMemoryPolicy()
    sync_history(state, [image_message(), {"role": "assistant", "content": "red"}, *text_history(4)])
    request = await owner.prepare(state)
    assert "JSON object" in request.prompt["prompt"]
    assert not owner.apply(state, request, completion("not JSON"))
    assert state.visual.covered == 0
    assert owner.apply(state, request, completion('{"facts":["red square"]}'))
    assert state.visual.resolved(state.visual.memory.mid_term)[0].text == '{"facts":["red square"]}'

    class MissingImagePolicy(StructuredMemoryPolicy):
        def build_compression(self, batch, previous_memory, limits):
            prompt = super().build_compression(batch, previous_memory, limits)
            return replace(
                prompt, content=tuple(part for part in prompt.content if not isinstance(part, CompressionImageRef))
            )

    owner.policy = MissingImagePolicy()
    sync_history(state, [image_message("blue"), {"role": "assistant", "content": "blue"}, *text_history(4)])
    with pytest.raises(ValueError, match="image reference"):
        await owner.prepare(state)
    assert state.visual.covered == 0
    assert len(state.visual.memory.pending) == 1
