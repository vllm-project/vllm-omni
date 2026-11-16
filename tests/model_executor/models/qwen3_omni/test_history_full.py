# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Full-history defaults preserve raw sources or reject the complete request."""

import json
from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pybase64
import pytest
from vllm.config import ModelConfig
from vllm.sampling_params import SamplingParams

from tests.engine.duplex.test_session_runner import Harness
from tests.model_executor.models.qwen3_omni.test_history_multimodal import image_message
from tests.model_executor.models.qwen3_omni.test_history_policy import Processor, audio_message, runtime, text_history
from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig
from vllm_omni.engine.duplex.config import DuplexSessionConfig
from vllm_omni.engine.duplex.contracts import DuplexAppendPlan, DuplexStageSubmission
from vllm_omni.engine.duplex.plugin import DuplexInputCapacityError
from vllm_omni.engine.duplex_orchestrator import DuplexOrchestrator
from vllm_omni.model_executor.models.qwen3_omni.duplex.plugin import Qwen3OmniDuplexPlugin
from vllm_omni.protocol.duplex.events import ErrorEvent

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def plugin_for(policy=None, *, max_audio=8, max_images=12):
    plugin = Qwen3OmniDuplexPlugin(lambda *args: "")
    options = {} if policy is None else {"history_policy": policy}
    plugin.configure_server(DuplexSessionRuntimeConfig(model_custom_options=options))
    plugin.processor = Processor()
    media = Mock(get_limit_per_prompt=lambda kind: {"audio": max_audio, "image": max_images}[kind])
    plugin._history_runtime.bind(plugin.processor, Mock(spec=ModelConfig, max_model_len=4096, multimodal_config=media))
    return plugin


def validate_engine_capacity(value: DuplexAppendPlan | DuplexStageSubmission, processor: Processor) -> None:
    """Stand in for engine expansion; exercise the production capacity validator."""
    if value.input_capacity is None:
        return
    media = value.prompt["multi_modal_data"]
    processed = processor(
        text=[value.prompt["prompt"]],
        audio=[audio for audio, _ in media.get("audio", ())],
        images=media.get("image", ()),
        return_tensors="np",
    )
    DuplexOrchestrator._input_capacity_validator(value.input_capacity)(len(processed["input_ids"][0]))


def enable_engine_capacity(h: Harness) -> None:
    """Let the recording port reject expanded inputs like the real stage port."""
    submit = h.port.submit

    async def checked_submit(submission):
        validate_engine_capacity(submission, h.runner.plugin.processor)
        return await submit(submission)

    h.port.submit = checked_submit


def plan(plugin, history, *, state=None, payload=None, view=None, runtime=None, validate=True):
    if state is None:
        state = plugin.create_session_state()
    payload = payload or {"type": "conversation"}
    config = {"conversation": history, **plugin.extra_prompt_config(runtime or {})}
    if view is not None:
        config["conversation_view"] = view
    config = plugin.prepare_prompt_config(config, state=state, payload=payload)
    prepared = plugin.plan_append(
        request_id="full-history",
        fence=None,
        session_config=config,
        runtime_config=runtime or {},
        seq=1,
        turn_seq=1,
        payload=payload,
        final=True,
        sampling_params=SamplingParams(max_tokens=64),
    )
    if validate:
        validate_engine_capacity(prepared, plugin.processor)
    return prepared.prompt


@pytest.mark.parametrize("policy", [None, "full_history", "recent"])
def test_default_and_explicit_full_history_keep_early_messages_and_empty_assistant(policy):
    plugin = plugin_for(policy)
    history = [
        {"role": "assistant", "content": "assistant-preface"},
        {"role": "system", "content": "EARLY_SYSTEM"},
        {"role": "user", "content": "EARLY_FACT"},
        {"role": "assistant", "content": ""},
        *text_history(40),
    ]
    prompt = plan(plugin, history)["prompt"]
    if policy == "recent":
        assert "EARLY_FACT" not in prompt
    else:
        assert "EARLY_FACT" in prompt and "EARLY_SYSTEM" in prompt and "assistant-preface" in prompt
        assert {"role": "assistant", "content": ""} in json.loads(prompt)
    assert "message 39" in prompt
    assert not plugin.supports_auxiliary_requests


def test_default_keeps_all_audio_allocations_until_sources_are_deleted():
    plugin = plugin_for()
    state = plugin.create_session_state()
    history = []
    for index in range(6):
        message, payload = audio_message(index)
        payload["audio"] = pybase64.b64encode(np.full(300000, index + 1, dtype="<f4").tobytes()).decode()
        history.append(message)
        plugin.prepare_prompt_config({"conversation": history}, state=state, payload=payload)
        history.append({"role": "assistant", "content": f"heard {index}"})
    assert len(state.audio_history) == 6
    assert sum(len(payload["audio"]) for _, payload in state.audio_history) > 8 * 1024 * 1024
    prompt = plan(plugin, history, state=state)
    assert [audio[0][0] for audio in prompt["multi_modal_data"]["audio"]] == [1, 2, 3, 4, 5, 6]
    prompt = plan(plugin, history[2:], state=state)
    assert len(state.audio_history) == 5
    assert [audio[0][0] for audio in prompt["multi_modal_data"]["audio"]] == [2, 3, 4, 5, 6]
    assert plugin.create_session_state().audio_history == []


@pytest.mark.parametrize("image_limit", [8, 12])
def test_full_history_uses_original_images_and_rejects_media_overflow_without_trimming(image_limit):
    plugin = plugin_for(max_images=image_limit)
    history = []
    colors = ["red", "green", "blue", "yellow", "black", "white", "orange", "purple", "gray"]
    for color in colors:
        history.extend([image_message(color), {"role": "assistant", "content": color}])
    history.append({"role": "user", "content": "what was in the first image?"})
    view = [{**message, "content": []} if message["role"] == "user" else message for message in history]
    if image_limit == 8:
        with pytest.raises(DuplexInputCapacityError, match="image"):
            plan(plugin, history, view=view)
    else:
        prompt = plan(plugin, history, view=view)
        images = prompt["multi_modal_data"]["image"]
        assert len(images) == 9
        assert images[0].getpixel((0, 0)) == (255, 0, 0)
        assert images[-1].getpixel((0, 0)) == (128, 128, 128)
        assert prompt["prompt"].count('"type": "image"') == 9


@pytest.mark.parametrize("kind", ["text", "audio", "image"])
def test_full_history_checks_exact_expanded_input_and_reserved_output(kind):
    plugin = plugin_for()
    state = plugin.create_session_state()
    payload = None
    message = {"role": "user", "content": "keep all this text " * 40}
    if kind == "audio":
        message, payload = audio_message(0)
    elif kind == "image":
        message = image_message()
    history = [message]
    prompt = plan(plugin, history, state=state, payload=payload)
    media = prompt["multi_modal_data"]
    cost = plugin.processor(
        text=[prompt["prompt"]],
        audio=[audio for audio, _ in media.get("audio", ())],
        images=media.get("image", ()),
        return_tensors="np",
    )["input_ids"][0]
    plugin._history_runtime.max_model_len = len(cost) + 64
    assert plan(plugin, history, state=state)["prompt"] == prompt["prompt"]
    plugin._history_runtime.max_model_len -= 1
    with pytest.raises(DuplexInputCapacityError, match="context"):
        plan(plugin, history, state=state)
    assert history == [message] and not state.history_policy_state.summary


def test_full_history_keeps_tool_anchors_outside_the_recent_window():
    plugin = plugin_for()
    runtime = plugin.note_published_function_call({}, call_id="early-call", history_len=1)
    runtime = plugin.note_published_function_call(runtime, call_id="later-call", history_len=24)
    assert runtime["qwen_call_anchors"]["early-call"] == 1
    runtime = plugin.runtime_config_for_function_output(
        DuplexSessionConfig(model="qwen"),
        runtime,
        {"call_id": "early-call", "name": "weather", "arguments": "{}", "output": "18C"},
        history_len=24,
    )
    prompt = plan(plugin, text_history(24), runtime=runtime)
    messages = json.loads(prompt["prompt"])
    assert messages[1]["tool_calls"][0]["id"] == "early-call"
    assert messages[2] == {"role": "tool", "tool_call_id": "early-call", "content": "18C"}
    assert "message 0" in prompt["prompt"] and "message 23" in prompt["prompt"]


@pytest.mark.asyncio
@pytest.mark.parametrize("keep_question", [False, True])
@pytest.mark.parametrize("policy", ["full_history", "bounded_recent", "incremental"])
async def test_history_capacity_error_preserves_session_and_can_retry_after_deletion(keep_question, policy):
    from tests.model_executor.models.qwen3_omni.test_duplex_plugin import open_qwen
    from vllm_omni.engine.duplex.commands import CreateItem, CreateResponse, DeleteItem
    from vllm_omni.engine.duplex.config import DuplexSessionState

    h = await open_qwen()
    try:
        plugin = h.runner.plugin
        plugin._history_runtime = runtime(policy)
        plugin.processor = plugin._history_runtime.processor
        enable_engine_capacity(h)
        await h.run(
            CreateItem(
                item={
                    "id": "oversized-source",
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "x" * 20000}],
                }
            )
        )
        await h.run(
            CreateItem(
                item={
                    "id": "question",
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "original question"}],
                }
            )
        )
        await h.run(CreateResponse())
        assert not h.port.submissions
        assert h.session.state == DuplexSessionState.OPEN
        assert any(isinstance(event, ErrorEvent) and event.code == "input_capacity_exceeded" for event in h.events)
        assert "x" * 20000 in str(h.session.history[0]["content"])
        await h.run(DeleteItem(item_id="oversized-source"))
        if not keep_question:
            await h.run(DeleteItem(item_id="question"))
        await h.run(CreateResponse())
        assert len(h.port.submissions) == int(keep_question)
        if keep_question:
            assert "original question" in h.port.submissions[0].prompt["prompt"]
        else:
            assert any(
                isinstance(event, ErrorEvent) and event.code == "response_create_without_input" for event in h.events
            )
        assert h.session.state == DuplexSessionState.OPEN
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(("complete_first", "delete_all"), [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize("policy", ["full_history", "recent", "bounded_recent", "incremental"])
async def test_deleting_one_merged_commit_removes_only_its_pcm(complete_first, delete_all, policy):
    from tests.engine.duplex.test_session_runner import append_audio, tts_output
    from tests.model_executor.models.qwen3_omni.test_duplex_plugin import open_qwen
    from vllm_omni.engine.duplex.commands import AckPlayback, Commit, CreateItem, CreateResponse, DeleteItem

    h = await open_qwen()
    try:
        plugin = h.runner.plugin
        plugin._history_runtime = runtime(policy)
        plugin.processor = plugin._history_runtime.processor
        for value in (0.1, 0.2):
            await h.run(append_audio(value=value))
            await h.run(Commit(final=True, create_response=False))
        item_ids = h.session.history_snapshot().item_ids
        first_id, second_id = item_ids[0], item_ids[1]
        assert isinstance(first_id, str) and isinstance(second_id, str)
        if complete_first:
            await h.run(CreateResponse())
            submission = h.port.submissions[-1]
            response_id = h.session.active_response_id
            await h.deliver_and_settle(
                tts_output(submission.context.request_id, samples=0, text="heard", finished=True), stage_id=0
            )
            await h.deliver_and_settle(tts_output(submission.context.request_id, finished=True), stage_id=2)
            await h.run(AckPlayback(response_id=response_id, played_ms=1000, committed_ms=1000))
        await h.run(DeleteItem(item_id=first_id))
        if delete_all:
            await h.run(DeleteItem(item_id=second_id))
        await h.run(
            CreateItem(
                item={
                    "id": "follow-up",
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "what remains?"}],
                }
            )
        )
        await h.run(CreateResponse())
        audio = h.port.submissions[-1].prompt["multi_modal_data"].get("audio", [])
        assert len(audio) == int(not delete_all)
        if not delete_all:
            np.testing.assert_allclose(audio[0][0], np.full(16000, 0.2, dtype="<f4"))
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["bounded_recent", "incremental"])
@pytest.mark.parametrize("evict_before_delete", [False, True])
async def test_deleted_pcm_is_removed_from_pool_after_foreground_cache_eviction(policy, evict_before_delete):
    from tests.engine.duplex.test_session_runner import append_audio, tts_output
    from tests.model_executor.models.qwen3_omni.test_duplex_plugin import open_qwen
    from vllm_omni.engine.duplex.commands import AckPlayback, Commit, CreateItem, CreateResponse, DeleteItem

    h = await open_qwen()
    try:
        plugin = h.runner.plugin
        plugin._history_runtime = runtime(policy)
        # Two clips exceed the foreground cache, while the larger pool keeps
        # both. This exercises the same eviction without allocating 8 MiB.
        plugin._history_runtime.behavior = replace(plugin._history_runtime.behavior, audio_cache_bytes=100000)

        async def finish_response():
            submission = h.port.submissions[-1]
            response_id = h.session.active_response_id
            await h.deliver_and_settle(
                tts_output(submission.context.request_id, samples=0, text="heard", finished=True), stage_id=0
            )
            await h.deliver_and_settle(tts_output(submission.context.request_id, finished=True), stage_id=2)
            await h.run(AckPlayback(response_id=response_id, played_ms=1000, committed_ms=1000))

        for value in (0.1, 0.2):
            await h.run(append_audio(value=value))
            await h.run(Commit(final=True, create_response=False))
        first_id = h.session.history_snapshot().item_ids[0]
        await h.run(CreateResponse())
        await finish_response()
        if not evict_before_delete:
            await h.run(DeleteItem(item_id=first_id))
        await h.run(append_audio(value=0.3))
        await h.run(Commit(final=True, create_response=True))
        assert len(h.runner.model_state.audio_history) == 1
        if evict_before_delete:
            await finish_response()
            await h.run(DeleteItem(item_id=first_id))
            await h.run(
                CreateItem(
                    item={
                        "id": "recall-after-deletion",
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "recall"}],
                    }
                )
            )
            await h.run(CreateResponse())
        audios = h.port.submissions[-1].prompt["multi_modal_data"]["audio"]
        assert len(audios) == 2
        for (audio, _), value in zip(audios, (0.2, 0.3)):
            np.testing.assert_allclose(audio, np.full(16000, value, dtype="<f4"))
    finally:
        await h.manager.shutdown()


@pytest.mark.parametrize("same_pcm", [False, True])
def test_legacy_anonymous_audio_after_tool_result_is_new_input(same_pcm):
    from tests.model_executor.models.qwen3_omni.test_duplex_tools import _weather_turn

    plugin = plugin_for("recent")
    state = plugin.create_session_state()
    message, first = audio_message(0)
    _, second = audio_message(0 if same_pcm else 1)
    plan(plugin, [message], state=state, payload=first)
    prompt = plan(
        plugin,
        [message],
        state=state,
        payload=second,
        runtime={"qwen_tool_turns": [_weather_turn("call", "Paris", "18C", 1)]},
    )
    audios = prompt["multi_modal_data"]["audio"]
    assert len(audios) == 2
    for (audio, _), value in zip(audios, (1, 1 if same_pcm else 2)):
        np.testing.assert_allclose(audio, np.full(160, value, dtype="<f4"))


def test_legacy_merged_commits_do_not_duplicate_first_audio():
    plugin = plugin_for("recent")
    state = plugin.create_session_state()
    first_message, first = audio_message(0)
    second_message, second = audio_message(1)
    plan(plugin, [first_message], state=state, payload=first)
    merged = {
        **first,
        "audio": pybase64.b64encode(pybase64.b64decode(first["audio"]) + pybase64.b64decode(second["audio"])).decode(),
    }
    prompt = plan(plugin, [first_message, second_message], state=state, payload=merged)
    audios = prompt["multi_modal_data"]["audio"]
    assert len(audios) == 1
    np.testing.assert_allclose(audios[0][0], np.repeat(np.array([1, 2], dtype="<f4"), 160))


@pytest.mark.asyncio
async def test_full_history_new_audio_after_capacity_rejection_keeps_original_pcm():
    from tests.engine.duplex.test_session_runner import append_audio
    from tests.model_executor.models.qwen3_omni.test_duplex_plugin import open_qwen
    from vllm_omni.engine.duplex.commands import Commit, CreateItem, CreateResponse, DeleteItem

    h = await open_qwen()
    try:
        plugin = h.runner.plugin
        plugin.processor = Processor()
        plugin._history_runtime.bind(
            plugin.processor, Mock(spec=ModelConfig, max_model_len=4096, multimodal_config=None)
        )
        enable_engine_capacity(h)
        await h.run(append_audio(value=0.1))
        await h.run(Commit(final=True, create_response=False))
        await h.run(
            CreateItem(
                item={
                    "id": "large",
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "x" * 20000}],
                }
            )
        )
        await h.run(CreateResponse())
        assert not h.port.submissions
        await h.run(DeleteItem(item_id="large"))
        await h.run(append_audio(value=0.2))
        await h.run(Commit(final=True, create_response=True))
        audio = h.port.submissions[-1].prompt["multi_modal_data"]["audio"]
        assert len(audio) == 2
        np.testing.assert_allclose(audio[0][0], np.full(16000, 0.1, dtype="<f4"))
        np.testing.assert_allclose(audio[1][0], np.full(16000, 0.2, dtype="<f4"))
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("remaining", ["text", "audio"])
async def test_full_history_retry_tracks_interleaved_text_and_committed_audio(remaining):
    from tests.engine.duplex.test_session_runner import append_audio
    from tests.model_executor.models.qwen3_omni.test_duplex_plugin import open_qwen
    from vllm_omni.engine.duplex.commands import Commit, CreateItem, CreateResponse, DeleteItem

    h = await open_qwen()
    try:
        plugin = h.runner.plugin
        plugin.processor = Processor()
        plugin._history_runtime.bind(
            plugin.processor, Mock(spec=ModelConfig, max_model_len=4096, multimodal_config=None)
        )
        enable_engine_capacity(h)
        await h.run(
            CreateItem(
                item={
                    "id": "keep",
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "original question"}],
                }
            )
        )
        await h.run(append_audio(value=0.1))
        await h.run(Commit(final=True, create_response=False))
        audio_id = h.session.history_snapshot().item_ids[-1]
        assert isinstance(audio_id, str)
        await h.run(
            CreateItem(
                item={
                    "id": "large",
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "x" * 20000}],
                }
            )
        )
        await h.run(CreateResponse())
        assert not h.port.submissions
        await h.run(DeleteItem(item_id="large"))
        await h.run(DeleteItem(item_id=audio_id if remaining == "text" else "keep"))
        await h.run(CreateResponse())
        assert len(h.port.submissions) == 1
        prompt = h.port.submissions[0].prompt
        if remaining == "text":
            assert "original question" in prompt["prompt"]
            assert not prompt["multi_modal_data"].get("audio")
        else:
            assert "original question" not in prompt["prompt"]
            np.testing.assert_allclose(prompt["multi_modal_data"]["audio"][0][0], np.full(16000, 0.1, dtype="<f4"))
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_full_history_recreating_deleted_audio_id_never_replays_old_pcm():
    from tests.engine.duplex.test_session_runner import append_audio
    from tests.model_executor.models.qwen3_omni.test_duplex_plugin import open_qwen
    from vllm_omni.engine.duplex.commands import Commit, CreateItem, CreateResponse, DeleteItem

    h = await open_qwen()
    try:
        await h.run(append_audio(value=0.1))
        await h.run(Commit(final=True, create_response=False))
        old_id = h.session.history_snapshot().item_ids[-1]
        assert isinstance(old_id, str)
        await h.run(DeleteItem(item_id=old_id))
        assert h.session.pending_input_bytes == 0
        new_audio = pybase64.b64encode(np.full(16000, 0.9, dtype="<f4").tobytes()).decode()
        await h.run(
            CreateItem(
                item={
                    "id": old_id,
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "audio": new_audio,
                            "format": "pcm_f32le",
                            "sample_rate_hz": 16000,
                        }
                    ],
                }
            )
        )
        await h.run(CreateResponse())
        assert len(h.port.submissions) == 1
        audio = h.port.submissions[0].prompt["multi_modal_data"]["audio"]
        assert len(audio) == 1
        np.testing.assert_allclose(audio[0][0], np.full(16000, 0.9, dtype="<f4"))
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("keep_second", [False, True])
async def test_full_history_queued_append_keeps_ownership_when_sources_are_deleted(keep_second):
    import asyncio

    from tests.engine.duplex.test_session_runner import append_audio, tts_output
    from tests.model_executor.models.qwen3_omni.test_duplex_plugin import open_qwen
    from vllm_omni.engine.duplex.commands import AckPlayback, Commit, CreateResponse, DeleteItem
    from vllm_omni.engine.duplex.config import DuplexSessionState

    h = await open_qwen()
    release = asyncio.Event()

    async def predecessor():
        await release.wait()
        return True

    try:
        for value in (0.1, 0.2):
            await h.run(append_audio(value=value))
            await h.run(Commit(final=True, create_response=False))
        first_id, second_id = h.session.history_snapshot().item_ids
        assert isinstance(first_id, str) and isinstance(second_id, str)
        h.runner.tasks.append_tail = asyncio.create_task(predecessor())
        h.submit(CreateResponse())
        for _ in range(200):
            if h.runner.tasks.append_tasks:
                break
            await asyncio.sleep(0.005)
        assert h.runner.tasks.append_tasks and not h.runner.model_state.committed_audio_consume_started
        h.submit(DeleteItem(item_id=first_id))
        if not keep_second:
            h.submit(DeleteItem(item_id=second_id))
        for _ in range(200):
            if first_id not in h.session.history_item_ids and (
                keep_second or second_id not in h.session.history_item_ids
            ):
                break
            await asyncio.sleep(0.005)
        assert first_id not in h.session.history_item_ids
        release.set()
        await h.settle()
        assert h.runner.model_state.committed_audio_payload is None
        assert h.session.pending_input_bytes == 0
        assert h.session.state == DuplexSessionState.OPEN
        assert len(h.port.submissions) == int(keep_second)
        if keep_second:
            submission = h.port.submissions[0]
            np.testing.assert_allclose(
                submission.prompt["multi_modal_data"]["audio"][0][0], np.full(16000, 0.2, dtype="<f4")
            )
            response_id = h.session.active_response_id
            await h.deliver_and_settle(
                tts_output(submission.context.request_id, samples=0, text="heard", finished=True), stage_id=0
            )
            await h.deliver_and_settle(tts_output(submission.context.request_id, finished=True), stage_id=2)
            await h.run(AckPlayback(response_id=response_id, played_ms=1000, committed_ms=1000))
            await h.run(CreateResponse())
            assert len(h.port.submissions) == 1
    finally:
        release.set()
        await h.manager.shutdown()


def test_original_policy_identity_is_independent_of_the_configured_default(monkeypatch):
    from vllm_omni.model_executor.models.qwen3_omni.duplex import history_policy
    from vllm_omni.model_executor.models.qwen3_omni.duplex.history_runtime import QwenHistoryRuntime

    monkeypatch.setattr(history_policy, "DEFAULT_HISTORY_POLICY", "recent")
    full = QwenHistoryRuntime("full_history", {})
    recent = QwenHistoryRuntime("recent", {})
    assert isinstance(full.policy, history_policy.FullHistoryPolicy)
    assert isinstance(recent.policy, history_policy.RecentHistoryPolicy)
    assert full.behavior.overflow == "reject" and recent.behavior.overflow == "fit"
    assert not hasattr(full, "full_history")


def test_original_custom_policy_controls_selection_without_plugin_branches():
    from vllm_omni.model_executor.models.qwen3_omni.duplex.history_policy import FullHistoryPolicy

    class LastTurn(FullHistoryPolicy):
        def select(self, state, limits):
            return super().select(state, limits)[-1:]

    plugin = plugin_for()
    plugin._history_runtime.policy = LastTurn()
    prompt = plan(plugin, text_history(21))["prompt"]
    assert "message 0" not in prompt and "message 20" in prompt


def test_full_history_planning_does_not_extract_media_features(mocker):
    plugin = plugin_for()
    state = plugin.create_session_state()
    audio, payload = audio_message(0)
    history = [audio, image_message()]
    extract = mocker.patch.object(Processor, "__call__", side_effect=AssertionError("unexpected feature extraction"))
    prompt = plan(plugin, history, state=state, payload=payload, validate=False)
    assert len(prompt["multi_modal_data"]["audio"]) == 1
    assert len(prompt["multi_modal_data"]["image"]) == 1
    history.extend([{"role": "assistant", "content": "heard"}, {"role": "user", "content": "recall"}])
    assert "recall" in plan(plugin, history, state=state, validate=False)["prompt"]
    extract.assert_not_called()


def test_custom_recent_policy_reserves_its_only_audio_slot_for_current_input(monkeypatch):
    import sys
    from dataclasses import replace

    from vllm_omni.model_executor.models.qwen3_omni.duplex.history_policy import RecentHistoryPolicy

    class OneAudio(RecentHistoryPolicy):
        behavior = replace(RecentHistoryPolicy.behavior, audio_cache_items=1)

    monkeypatch.setattr(sys.modules[__name__], "OneAudio", OneAudio, raising=False)
    plugin = plugin_for(f"{__name__}:OneAudio")
    state = plugin.create_session_state()
    first, first_payload = audio_message(1)
    plan(plugin, [first], state=state, payload=first_payload)
    _, current_payload = audio_message(2)
    prompt = plan(plugin, [first, {"role": "assistant", "content": "heard"}], state=state, payload=current_payload)
    audio = prompt["multi_modal_data"]["audio"]
    assert len(audio) == 1 and not state.audio_history
    np.testing.assert_array_equal(audio[0][0], np.full(160, 3, dtype="<f4"))


@pytest.mark.parametrize("kind", ["original_bytes", "original_items", "bounded_without_items"])
def test_audio_retention_rejects_limits_that_would_be_ignored(kind):
    from dataclasses import replace

    from vllm_omni.model_executor.models.qwen3_omni.duplex.history_policy import FullHistoryPolicy, HistoryBehavior

    with pytest.raises(ValueError, match="audio"):
        if kind == "original_bytes":
            replace(FullHistoryPolicy.behavior, audio_cache_bytes=64)
        elif kind == "original_items":
            replace(FullHistoryPolicy.behavior, audio_cache_items=1)
        else:
            HistoryBehavior(original_messages=True, audio_cache_items=None, audio_cache_bytes=64)


@pytest.mark.parametrize("use_view", [False, True])
def test_original_audio_retention_rejects_missing_pcm_even_when_overflow_can_fit(monkeypatch, use_view):
    import sys
    from dataclasses import replace

    from vllm_omni.model_executor.models.qwen3_omni.duplex.history_policy import FullHistoryPolicy

    class OriginalFit(FullHistoryPolicy):
        behavior = replace(FullHistoryPolicy.behavior, overflow="fit", use_media_view=use_view)

    monkeypatch.setattr(sys.modules[__name__], "OriginalFit", OriginalFit, raising=False)
    plugin = plugin_for(f"{__name__}:OriginalFit")
    missing, payload = audio_message(0)
    history = [missing, {"role": "assistant", "content": "old answer"}, {"role": "user", "content": "recall"}]
    view = [{"role": "user", "content": "old transcript"}, *history[1:]] if use_view else None
    state = plugin.create_session_state()
    with pytest.raises(DuplexInputCapacityError, match="unavailable audio"):
        plan(plugin, history, state=state, view=view)
    plan(plugin, [missing], state=state, payload=payload)
    shown = plan(plugin, history, state=state, view=view)
    assert len(shown["multi_modal_data"].get("audio", ())) == int(not use_view)
    assert len(state.audio_history) == 1
    assert plan(plugin, history, state=state)["multi_modal_data"]["audio"][0][0][0] == 1
