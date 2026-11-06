# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Memory correctness without loading model weights or allocating a GPU."""

import json
from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pybase64
import pytest
from vllm.config import ModelConfig
from vllm.outputs import CompletionOutput, RequestOutput

from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig
from vllm_omni.engine.duplex.plugin import DuplexRuntimeConfigError
from vllm_omni.model_executor.models.qwen3_omni.duplex.history_policy import (
    HistoryLimits,
    IncrementalHistoryPolicy,
    QwenHistoryPolicy,
    QwenHistoryState,
    load_history_policy,
)
from vllm_omni.model_executor.models.qwen3_omni.duplex.history_runtime import HISTORY_PROMPT, QwenHistoryRuntime
from vllm_omni.model_executor.models.qwen3_omni.duplex.plugin import Qwen3OmniDuplexPlugin
from vllm_omni.model_executor.models.qwen3_omni.duplex.session import QwenDuplexSessionState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class Processor:
    """Count expanded audio as expensive, even when its placeholder is short."""

    audio_tokens = 100

    def apply_chat_template(self, messages, **kwargs):
        return json.dumps(messages, ensure_ascii=False)

    def tokenizer(self, *, text):
        return {"input_ids": [list(range(max(1, len(value) // 4))) for value in text]}

    def __call__(self, *, text, audio, return_tensors):
        count = len(self.tokenizer(text=text)["input_ids"][0]) + len(audio) * self.audio_tokens
        return {"input_ids": [list(range(count))]}


def runtime(name="incremental", **limits):
    result = QwenHistoryRuntime(name, limits)
    result.bind(Processor(), Mock(spec=ModelConfig, max_model_len=4096, multimodal_config=None))
    return result


def text_history(n=20):
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": f"message {i}"} for i in range(n)]


def audio_message(i):
    encoded = pybase64.b64encode(np.full(160, i + 1, dtype="<f4").tobytes()).decode()
    message = {"role": "user", "content": [{"type": "audio_url", "audio_url": {"url": f"source-{i}"}}]}
    return message, {"audio": encoded, "format": "pcm_f32le", "sample_rate_hz": 16000}


def completion(text="user budget is 3000", reason="stop"):
    return RequestOutput(
        request_id="summary",
        prompt="",
        prompt_token_ids=[],
        prompt_logprobs=None,
        finished=True,
        outputs=[
            CompletionOutput(
                index=0, text=text, token_ids=[], cumulative_logprob=0.0, logprobs=None, finish_reason=reason
            )
        ],
    )


class IncompleteHistoryPolicy(QwenHistoryPolicy):
    def select(self, state, limits):
        return []


class IdleHistoryPolicy(IncrementalHistoryPolicy):
    """Example extension that defers work while preparing or after rejection."""

    def next_batch(self, state, limits):
        if state.preparing or state.counts["rejected_summaries"]:
            return None
        return super().next_batch(state, limits)


def test_incomplete_external_policy_is_rejected_at_load():
    with pytest.raises(TypeError):
        load_history_policy(__name__ + ":IncompleteHistoryPolicy")


def test_unavailable_current_audio_cannot_silently_fall_back_to_old_question():
    owner, state = runtime("bounded_recent"), QwenHistoryState()
    missing, _ = audio_message(0)
    state.reconcile([*text_history(2), missing, {"role": "user", "content": "current budget is 5000"}])
    with pytest.raises(DuplexRuntimeConfigError, match="Current.*audio"):
        owner.foreground(state, {"type": "conversation"})


def test_current_mixed_input_survives_too_small_history_media_pool():
    plugin = Qwen3OmniDuplexPlugin(lambda *args: "")
    plugin._history_runtime = runtime("bounded_recent", max_audio_bytes=100)
    message, payload = audio_message(0)
    message["content"].append({"type": "text", "text": "current budget is 5000"})
    state = QwenDuplexSessionState()
    config = plugin.prepare_prompt_config({"conversation": [message]}, state=state, payload=payload)
    prompt = plugin._history_runtime.build_foreground(config[HISTORY_PROMPT], {}, 128)
    assert "current budget is 5000" in prompt["prompt"]
    assert len(prompt["multi_modal_data"]["audio"]) == 1
    assert state.history_policy_state.audio_bytes <= 100


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", ["items", "bytes", "bytes_between_item_sizes"])
async def test_configured_media_capacity_triggers_before_sources_are_lost(limit):
    plugin = Qwen3OmniDuplexPlugin(lambda *args: "")
    _, payload = audio_message(0)
    limits = {"max_audio_items": 2} if limit == "items" else {"max_audio_bytes": 2 * len(payload["audio"])}
    if limit == "bytes_between_item_sizes":
        limits["max_audio_bytes"] = 3 * len(payload["audio"]) // 2
    plugin._history_runtime = runtime(recent_messages=100, **limits)
    state, history = QwenDuplexSessionState(), []
    planned = []
    for index in range(6):
        message, payload = audio_message(index)
        history.append(message)
        plugin.prepare_prompt_config({"conversation": history}, state=state, payload=payload)
        history.append({"role": "assistant", "content": f"heard {index}"})
        policy = plugin.auxiliary_policy(state=state, history_revision=index, history=history)
        if policy:
            request = await plugin.prepare_auxiliary_request(state=state, history=(), runtime_config={})
            assert request is not None
            planned.append(index)
            assert plugin.apply_auxiliary_result(state=state, request=request, output=completion())
    assert planned[0] <= 1  # Compress no later than the second completed turn.
    assert not state.history_policy_state.omissions
    assert state.history_policy_state.processed >= 8


@pytest.mark.asyncio
async def test_background_uses_available_foreground_source_within_media_limits():
    plugin = Qwen3OmniDuplexPlugin(lambda *args: "")
    plugin._history_runtime = runtime(recent_messages=2, max_audio_items=1)
    state = QwenDuplexSessionState()
    first, payload = audio_message(0)
    later, later_payload = audio_message(1)
    history = [
        first,
        {"role": "assistant", "content": "first"},
        later,
        {"role": "assistant", "content": "later"},
        *text_history(4),
    ]
    state.audio_history = [((first["content"],), payload), ((later["content"],), later_payload)]
    memory = state.history_policy_state
    memory.reconcile(history)
    memory.retain(state.audio_history, plugin._history_runtime.limits)
    assert memory.audio_for(first) is None
    request = await plugin.prepare_auxiliary_request(state=state, history=(), runtime_config={})
    assert request.model_source.start == 0
    assert request.model_source.end == 2
    assert not memory.omissions
    np.testing.assert_array_equal(request.prompt["multi_modal_data"]["audio"][0][0], np.ones(160))
    assert len(memory.held_audio) <= 1
    assert memory.counts["peak_audio_bytes"] <= plugin._history_runtime.limits.max_audio_bytes


def test_readding_old_cached_audio_does_not_change_eviction_order():
    state, limits = QwenHistoryState(), HistoryLimits(max_audio_items=2)
    entries, history = [], []
    for i in range(4):
        message, payload = audio_message(i)
        history.extend([message, {"role": "assistant", "content": "heard"}])
        entries.append(((message["content"],), payload))
    state.reconcile(history)
    state.retain(entries[:3], limits)
    state.retain(entries, limits)
    assert state.audio_for(history[0]) is None
    assert state.audio_for(history[2]) is None
    assert state.audio_for(history[4]) is not None
    assert state.audio_for(history[6]) is not None


@pytest.mark.asyncio
async def test_unchanged_revision_skips_history_work_but_edits_and_progress_recompute(mocker):
    plugin = Qwen3OmniDuplexPlugin(lambda *args: "")
    plugin._history_runtime = runtime()
    state, history = QwenDuplexSessionState(), text_history()
    owner = plugin._history_runtime
    initial = plugin.auxiliary_policy(state=state, history_revision=1, history=history)
    reconcile = mocker.spy(state.history_policy_state, "reconcile")
    validate = mocker.spy(owner, "validate_history")
    work = mocker.spy(owner, "work")
    for _ in range(100):
        assert plugin.auxiliary_policy(state=state, history_revision=1, history=history) == initial
    assert reconcile.call_count == validate.call_count == work.call_count == 0
    request = await owner.prepare(state.history_policy_state)
    assert owner.apply(state.history_policy_state, request, completion())
    following = plugin.auxiliary_policy(state=state, history_revision=1, history=history)
    assert following.source_key != initial.source_key
    history[0] = {"role": "user", "content": "corrected budget is 5000"}
    plugin.auxiliary_policy(state=state, history_revision=2, history=history)
    assert state.history_policy_state.summary == ""


def test_no_pending_work_is_cached_and_new_runtime_revalidates(mocker):
    plugin = Qwen3OmniDuplexPlugin(lambda *args: "")
    plugin._history_runtime = runtime()
    state, history = QwenDuplexSessionState(), text_history(6)
    assert plugin.auxiliary_policy(state=state, history_revision=1, history=history) is None
    reconcile = mocker.spy(state.history_policy_state, "reconcile")
    work = mocker.spy(plugin._history_runtime, "work")
    for _ in range(100):
        assert plugin.auxiliary_policy(state=state, history_revision=1, history=history) is None
    assert reconcile.call_count == work.call_count == 0
    plugin._history_runtime = runtime(recent_messages=2)
    assert plugin.auxiliary_policy(state=state, history_revision=1, history=history) is not None
    assert reconcile.call_count == 1


def test_unchanged_history_reconsiders_new_media_pressure():
    plugin = Qwen3OmniDuplexPlugin(lambda *args: "")
    _, payload = audio_message(0)
    plugin._history_runtime = runtime(recent_messages=100, max_audio_bytes=len(payload["audio"]))
    state = QwenDuplexSessionState()
    message, payload = audio_message(0)
    history = [message, {"role": "assistant", "content": "heard"}]
    assert plugin.auxiliary_policy(state=state, history_revision=1, history=history) is None
    state.history_policy_state.retain([((message["content"],), payload)], plugin._history_runtime.limits)
    assert plugin.auxiliary_policy(state=state, history_revision=1, history=history) is not None


def test_foreground_clipping_records_one_degradation_event(caplog):
    owner, state = runtime(summary_tokens=32), QwenHistoryState()
    state.reconcile(text_history(20))
    context = owner.foreground(state, {})
    # Fit the latest turn, but not its predecessor or the summary.
    latest = replace(context, groups=context.groups[-1:], summary="")
    rendered = owner.build_foreground(latest, {}, 128)
    owner.max_model_len = len(owner.processor.tokenizer(text=[rendered["prompt"]])["input_ids"][0]) + 128
    context = replace(context, summary="budget 3000 " * 20)
    with caplog.at_level("DEBUG", logger="vllm_omni.model_executor.models.qwen3_omni.duplex.history_runtime"):
        prompt = owner.build_foreground(context, {}, 128)
    events = [r for r in caplog.records if "Qwen history foreground" in r.message]
    assert len(events) == 1
    assert "degraded=1" in events[0].message
    assert "trimmed_turns=9" in events[0].message
    assert "summary_dropped=1" in events[0].message
    assert "message 18" in prompt["prompt"]


@pytest.mark.asyncio
async def test_incremental_summary_reuses_previous_and_does_not_repeat_covered_sources():
    owner, state = runtime(), QwenHistoryState()
    state.reconcile(text_history())
    first = await owner.prepare(state)
    assert first.history_length == 8
    assert first.stage_id == 0
    assert first.sampling_params.max_tokens == 512
    assert owner.apply(state, first, completion())
    second = await owner.prepare(state)
    assert second.history_length == 12
    assert "user budget is 3000" in second.prompt["prompt"]
    assert "message 0" not in second.prompt["prompt"]
    assert owner.apply(state, second, completion("updated memory"))
    assert await owner.prepare(state) is None
    foreground = owner.foreground(state, {})
    assert foreground.summary == "updated memory"
    assert "message 12" in str(foreground.groups)
    assert "message 0'" not in str(foreground.groups)


@pytest.mark.asyncio
async def test_audio_pressure_precedes_text_window_and_uses_original_pcm_once():
    owner, state = runtime(recent_messages=100), QwenHistoryState()
    history, retained = [], []
    for i in range(4):
        message, payload = audio_message(i)
        history += [message, {"role": "assistant", "content": f"heard {i}"}]
        retained.append(((message["content"],), payload))
    state.reconcile(history)
    state.retain(retained, owner.limits)
    request = await owner.prepare(state)
    assert request is not None
    assert request.history_length == 2
    assert len(request.prompt["multi_modal_data"]["audio"]) == 1
    np.testing.assert_array_equal(request.prompt["multi_modal_data"]["audio"][0][0], np.ones(160))
    assert "heard 0" in request.prompt["prompt"]


@pytest.mark.asyncio
async def test_permanently_missing_audio_does_not_block_following_text():
    owner, state = runtime(recent_messages=2), QwenHistoryState()
    message, _ = audio_message(0)
    state.reconcile([message, {"role": "assistant", "content": "answer without available evidence"}, *text_history(8)])
    request = await owner.prepare(state)
    assert list(state.omissions) == [(0, 2, "source_unavailable")]
    assert "answer without available evidence" not in request.prompt["prompt"]
    assert request.model_source.start == 2
    assert owner.apply(state, request, completion())
    assert state.processed > 2


@pytest.mark.asyncio
async def test_transient_failure_retries_same_source_and_preserves_summary():
    owner, state = runtime(), QwenHistoryState()
    state.reconcile(text_history())
    request = await owner.prepare(state)
    assert not owner.apply(state, request, completion("", "length"))
    assert state.processed == 0 and not state.omissions
    again = await owner.prepare(state)
    assert again.model_source == request.model_source
    assert request.retry_delays_s


@pytest.mark.asyncio
async def test_cached_decision_reconsiders_rejected_summary_for_custom_policy():
    owner, state = runtime(), QwenHistoryState()
    owner.policy = IdleHistoryPolicy()
    history = text_history()
    state.reconcile(history)
    request = await owner.prepare(state)
    assert owner.policy_for(state, history, 1) is not None
    assert not owner.apply(state, request, completion("", "length"))
    assert owner.policy_for(state, history, 1) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("edit", ["truncate", "delete", "replace_audio"])
async def test_source_edits_reject_late_results_and_invalidate_derived_memory(edit):
    owner, state = runtime(), QwenHistoryState()
    history = text_history()
    if edit == "replace_audio":
        m, p = audio_message(1)
        history[0] = m
        state.retain([((m["content"],), p)], owner.limits)
    state.reconcile(history)
    request = await owner.prepare(state)
    assert owner.apply(state, request, completion())
    late = await owner.prepare(state)
    if edit == "truncate":
        history[1] = {"role": "assistant", "content": ""}
    elif edit == "delete":
        history.pop(0)
    else:
        history[0], _ = audio_message(2)
    state.reconcile(history)
    assert state.summary == "" and state.processed == 0
    assert not owner.apply(state, late, completion("stale text must never return"))


def test_media_pool_deduplicates_bindings_and_enforces_single_item_limit():
    state, limits = QwenHistoryState(), HistoryLimits(max_audio_bytes=1500, max_audio_items=2)
    for i in range(4):
        m, p = audio_message(i)
        state.retain([((m["content"],), p)] * 2, limits)
        assert state.audio_bytes <= 1500
        assert len(state.audio) == 1
    m, p = audio_message(7)
    state.retain([((m["content"],), {**p, "audio": p["audio"] * 3})], limits)
    assert state.counts["oversize_audio"] == 1
    assert state.audio_bytes <= 1500


def test_external_policy_is_loadable_without_modifying_plugin():
    policy = load_history_policy(__name__ + ":IncrementalHistoryPolicy")
    assert isinstance(policy, IncrementalHistoryPolicy)
    with pytest.raises(ValueError):
        load_history_policy("misspelled-policy")


def test_policy_rejects_images_tools_and_leaves_default_capabilities_intact():
    plugin = Qwen3OmniDuplexPlugin(lambda *args: "")
    assert plugin.capabilities(max_sessions=1).supports_image_input
    plugin.configure_server(DuplexSessionRuntimeConfig(model_options={"history_policy": "incremental"}))
    assert not plugin.capabilities(max_sessions=1).supports_image_input
    with pytest.raises(DuplexRuntimeConfigError, match="tools"):
        plugin.validate_client_extra_body({"realtime_tools": [{"name": "refund"}]})
    with pytest.raises(DuplexRuntimeConfigError, match="text and audio"):
        plugin._history_runtime.validate_history([{"role": "user", "content": [{"type": "image_url"}]}])


@pytest.mark.asyncio
async def test_expanded_audio_context_limit_and_oversize_batch_progress():
    owner, state = runtime(summary_tokens=64, recent_messages=2), QwenHistoryState()
    owner.max_model_len = 400
    owner.processor.audio_tokens = 500
    message, payload = audio_message(0)
    state.reconcile([message, {"role": "assistant", "content": "heard"}, *text_history(8)])
    state.retain([((message["content"],), payload)], owner.limits)
    request = await owner.prepare(state)
    assert state.omissions[0] == (0, 2, "summary_capacity")
    assert request is not None and request.model_source.start == 2
    state.reconcile([message])
    context = owner.foreground(state, payload)
    with pytest.raises(DuplexRuntimeConfigError, match="capacity"):
        owner.build_foreground(context, {}, 64)


@pytest.mark.asyncio
async def test_new_tail_does_not_revoke_valid_summary_or_old_source_request():
    owner, state = runtime(), QwenHistoryState()
    history = text_history()
    state.reconcile(history)
    request = await owner.prepare(state)
    state.reconcile([*history, {"role": "user", "content": "new budget is 5000"}])
    assert owner.apply(state, request, completion())
    context = owner.foreground(state, {})
    prompt = owner.build_foreground(context, {}, 512)["prompt"]
    assert "new budget is 5000" in prompt and "Later messages and corrections take precedence" in prompt


def test_configuration_rejects_nonintegral_and_unbounded_values():
    for value in (0, -1, True, 0.5):
        with pytest.raises(ValueError):
            replace(HistoryLimits(), max_audio_bytes=value)


@pytest.mark.asyncio
async def test_real_session_runtime_keeps_audio_summary_private_and_stops_at_thinker():
    from tests.model_executor.models.joyai_vl_interaction.test_duplex_memory import start_summary, summary_output
    from tests.model_executor.models.qwen3_omni.test_duplex_plugin import open_qwen

    h = await open_qwen()
    try:
        owner = runtime(recent_messages=2)
        h.runner.plugin._history_runtime = owner
        h.runner.plugin.supports_auxiliary_requests = True
        message, payload = audio_message(0)
        h.session.append_history_message(message)
        h.session.append_history_message({"role": "assistant", "content": "heard prefix"})
        for entry in text_history(4):
            h.session.append_history_message(entry)
        h.runner.model_state.history_policy_state.retain([((message["content"],), payload)], owner.limits)
        submission = await start_summary(h)
        assert submission.context.stage_id == submission.context.final_stage_id == 0
        assert not submission.resumable
        assert len(submission.prompt["multi_modal_data"]["audio"]) == 1
        h.deliver(summary_output(submission.context.request_id, "user budget 3000"), stage_id=0, segment_finished=True)
        events = await h.settle()
        assert h.runner.model_state.history_policy_state.summary == "user budget 3000"
        assert not any(e.to_realtime()["type"].startswith("response.") for e in events)
        assert submission.context.request_id not in h.session.resource_request_ids()
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_real_session_close_cancels_private_summary_and_rejects_late_output():
    from tests.model_executor.models.joyai_vl_interaction.test_duplex_memory import start_summary, summary_output
    from tests.model_executor.models.qwen3_omni.test_duplex_plugin import open_qwen

    h = await open_qwen()
    owner = runtime(recent_messages=2)
    h.runner.plugin._history_runtime = owner
    h.runner.plugin.supports_auxiliary_requests = True
    for entry in text_history(8):
        h.session.append_history_message(entry)
    submission = await start_summary(h)
    task = h.runner.memory._task
    await h.manager.shutdown()
    assert task.done()
    h.deliver(summary_output(submission.context.request_id), stage_id=0, segment_finished=True)
    assert not h.runner.model_state.history_policy_state.summary


@pytest.mark.asyncio
async def test_real_foreground_append_uses_stage_sampling_object_and_preserves_current_input():
    from tests.model_executor.models.qwen3_omni.test_duplex_plugin import open_qwen
    from vllm_omni.engine.duplex.commands import CreateItem, CreateResponse

    h = await open_qwen()
    try:
        h.runner.plugin._history_runtime = runtime()
        h.runner.plugin.supports_auxiliary_requests = True
        await h.run(
            CreateItem(
                item={
                    "id": "current-input",
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "current budget is 5000"}],
                }
            )
        )
        await h.run(CreateResponse())
        assert h.port.submissions
        assert "current budget is 5000" in h.port.submissions[-1].prompt["prompt"]
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["retained", "foreground"])
async def test_cancelled_preparation_keeps_media_charged_until_cpu_worker_finishes(source):
    import asyncio
    import threading

    owner, state = runtime(recent_messages=2, max_audio_items=1), QwenHistoryState()
    owner.policy = IdleHistoryPolicy()
    message, payload = audio_message(0)
    state.reconcile([message, {"role": "assistant", "content": "heard"}, *text_history(4)])
    cache = [((message["content"],), payload)]
    if source == "retained":
        state.retain(cache, owner.limits)
    entered, release = threading.Event(), threading.Event()
    render = owner._render

    def delayed(*args):
        entered.set()
        assert release.wait(5)
        return render(*args)

    owner._render = delayed
    task = asyncio.create_task(owner.prepare(state, lambda: cache))
    try:
        for _ in range(200):
            if entered.is_set():
                break
            await asyncio.sleep(0.005)
        assert entered.is_set()
        cache.clear()  # Foreground eviction must not release the worker's charged source.
        task.cancel()
        await asyncio.sleep(0.01)
        assert not task.done() and state.preparing
        newer, new_payload = audio_message(1)
        state.retain([((newer["content"],), new_payload)], owner.limits)
        assert len(state.held_audio) == 1
        assert state.held_audio[0].encoded is payload["audio"]
        assert state.audio_bytes <= owner.limits.max_audio_bytes
        assert owner.policy_for(state, state.history, 1) is None
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert not state.preparing
    assert owner.policy_for(state, state.history, 1) is not None


def test_identical_pcm_in_distinct_turns_is_not_confused_with_current_input():
    owner, state = runtime(), QwenHistoryState()
    old, payload = audio_message(0)
    state.reconcile([old, {"role": "assistant", "content": "old response"}])
    state.retain([((old["content"],), payload)], owner.limits)
    context = owner.foreground(state, payload)
    assert len(context.groups) == 2
    assert sum(len(group.audio) for group in context.groups) == 2


def test_rebinding_audio_counts_source_still_held_by_preparation():
    state = QwenHistoryState()
    message, payload = audio_message(0)
    _, replacement = audio_message(1)
    limits = HistoryLimits(max_audio_items=1, max_audio_bytes=len(payload["audio"]))
    state.reconcile([message])
    state.retain([((message["content"],), payload)], limits)
    original = state.audio[0]
    state.preparing = {id(original): original}
    state.retain([((message["content"],), replacement)], limits)
    # The old worker still owns its source. The replacement cannot consume
    # another slot, and the old binding must not masquerade as current audio.
    assert state.audio_for(message) is None
    assert state.audio_bytes == len(payload["audio"])
    state.preparing.clear()
    assert state.audio_bytes == 0
    state.retain([((message["content"],), replacement)], limits)
    assert state.audio_for(message).encoded is replacement["audio"]
