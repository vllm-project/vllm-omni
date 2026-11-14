# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Token-triggered history compression and independently bounded source batches."""

import asyncio
import threading
from pathlib import Path
from unittest.mock import Mock

import pytest
from vllm.config import ModelConfig

from tests.model_executor.models.qwen3_omni.test_history_multimodal import image_message
from tests.model_executor.models.qwen3_omni.test_history_policy import (
    IdleHistoryPolicy,
    Processor,
    audio_message,
    completion,
    runtime,
    sync_history,
)
from vllm_omni.config.stage_config import load_deploy_config
from vllm_omni.model_executor.models.qwen3_omni.duplex.history_policy import HistoryLimits
from vllm_omni.model_executor.models.qwen3_omni.duplex.plugin import Qwen3OmniDuplexPlugin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def turn(width):
    return [
        {"role": "user", "content": [{"type": "text", "text": "x" * width}]},
        {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
    ]


def test_token_options_preserve_positional_message_limit_arguments():
    limits = HistoryLimits(2, 4, 128, recent_tokens=1000, chunk_tokens=256)
    assert (limits.recent_messages, limits.chunk_messages, limits.summary_tokens) == (2, 4, 128)
    assert (limits.recent_tokens, limits.chunk_tokens) == (1000, 256)


@pytest.mark.asyncio
async def test_deployment_overlay_loads_token_targets_into_compression(tmp_path):
    config_path = tmp_path / "tokens.yaml"
    base = Path(__file__).resolve().parents[4] / "vllm_omni/deploy/qwen3_omni_duplex.yaml"
    config_path.write_text(
        f"base_config: {base}\n"
        "duplex_session:\n  model_custom_options:\n    history_policy: incremental\n"
        "    history_limits:\n      recent_messages: 1\n      chunk_messages: 1\n"
        "      recent_tokens: 1000\n      chunk_tokens: 230\n      summary_tokens: 64\n"
    )
    plugin = Qwen3OmniDuplexPlugin(lambda *args: "")
    plugin.configure_server(load_deploy_config(config_path).duplex_session)
    plugin._history_runtime.bind(Processor(), Mock(spec=ModelConfig, max_model_len=4096, multimodal_config=None))
    state = plugin.create_session_state()
    history = [message for _ in range(10) for message in turn(300)]
    sync_history(state.history_policy_state, history)
    plugin.prepare_prompt_config({"conversation": history}, state=state, payload={})
    request = await plugin.prepare_auxiliary_request(state=state, history=(), runtime_config={})
    assert request is not None and request.model_source.end == 4


@pytest.mark.asyncio
@pytest.mark.parametrize(("turns", "width", "compress"), [(12, 1, False), (1, 2400, True)])
async def test_token_trigger_distinguishes_many_short_and_few_long_messages(turns, width, compress):
    owner = runtime(recent_messages=1, recent_tokens=512)
    state = owner.create_state()
    sync_history(state, [message for _ in range(turns) for message in turn(width)])
    request = await owner.prepare(state)
    assert (request is not None) is compress
    assert state.processed == 0 and not state.omissions and not state.ledger.nodes()


@pytest.mark.asyncio
@pytest.mark.parametrize("margin", [-1, 0, 1])
async def test_token_trigger_counts_rendered_originals_and_requires_exceeding_target(margin):
    owner = runtime()
    history = turn(200)
    original = owner.processor.apply_chat_template(history)
    count = len(owner.processor.tokenizer(text=[original])["input_ids"][0])
    owner = runtime(recent_messages=1, recent_tokens=count + margin)
    state = owner.create_state()
    sync_history(state, history)
    assert (await owner.prepare(state) is not None) is (margin < 0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("limits", "end", "more"),
    [
        ({"recent_messages": 1000, "recent_tokens": 1000, "chunk_messages": 1, "chunk_tokens": 230}, 4, False),
        ({"recent_messages": 2, "chunk_messages": 1, "chunk_tokens": 230}, 4, True),
        ({"recent_messages": 1000, "recent_tokens": 200, "chunk_messages": 2}, 2, True),
    ],
)
async def test_token_targets_independently_replace_their_message_targets(limits, end, more):
    owner = runtime(summary_tokens=64, **limits)
    state = owner.create_state()
    sync_history(state, [message for _ in range(10) for message in turn(300)])
    request = await owner.prepare(state)
    assert request is not None and request.model_source.end == end
    assert owner.apply(state, request, completion("remember the original facts"))
    assert state.processed == end
    assert len(state.ledger.get(state.summary_node_id).sources) == end
    assert (await owner.prepare(state) is not None) is more


@pytest.mark.asyncio
async def test_token_batch_folds_a_single_turn_that_fits_the_model_context():
    owner = runtime(recent_tokens=128, chunk_tokens=160, summary_tokens=64)
    state = owner.create_state()
    history = turn(1800)
    history[0]["content"][0]["text"] += " USER_TAIL"
    history[1]["content"][0]["text"] = "ASSISTANT_TAIL"
    sync_history(state, history)
    prompts = []
    for _ in range(30):
        request = await owner.prepare(state)
        assert request is not None
        prompts.append(request.prompt["prompt"])
        assert owner.apply(state, request, completion("folded original facts"))
        if state.processed:
            break
        assert state.partial is not None and not state.summary and not state.ledger.nodes()
    assert len(prompts) > 1 and state.processed == 2 and state.partial is None
    assert any("USER_TAIL" in prompt for prompt in prompts)
    assert any("ASSISTANT_TAIL" in prompt for prompt in prompts)
    assert state.summary == "folded original facts" and not state.omissions


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["audio", "image"])
@pytest.mark.parametrize(("target", "compress"), [(80, True), (10000, False)])
async def test_token_trigger_counts_expanded_media_without_early_message_compression(kind, target, compress):
    owner = runtime(recent_messages=1, recent_tokens=target, chunk_tokens=64)
    state = owner.create_state()
    if kind == "audio":
        message, payload = audio_message(0)
        state.retain([((message["content"],), payload)], owner.limits)
    else:
        message = image_message()
    sync_history(state, [message, {"role": "assistant", "content": "heard or seen"}])
    request = await owner.prepare(state)
    assert (request is not None) is compress
    if request is not None:
        assert len(request.prompt["multi_modal_data"][kind]) == 1
        assert owner.apply(state, request, completion("original media fact"))
    assert not state.omissions


@pytest.mark.asyncio
async def test_source_edit_during_token_counting_discards_the_old_plan(monkeypatch):
    owner = runtime(recent_tokens=128, chunk_tokens=256)
    state = owner.create_state()
    sync_history(state, turn(1600))
    entered, release = threading.Event(), threading.Event()
    count = owner._source_tokens

    def delayed(group):
        entered.set()
        assert release.wait(5)
        return count(group)

    monkeypatch.setattr(owner, "_source_tokens", delayed)
    task = asyncio.create_task(owner.prepare(state))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        replacement = turn(1800)
        replacement[0]["content"][0]["text"] += " corrected"
        sync_history(state, replacement)
    finally:
        release.set()
    assert await task is None
    assert state.processed == 0 and not state.ledger.nodes() and not state.preparing
    request = await owner.prepare(state)
    assert request is not None
    assert request.model_source.sources == state.sources


@pytest.mark.asyncio
async def test_visual_batch_shrinks_to_token_target_before_the_image_count_limit():
    owner = runtime(recent_tokens=1, chunk_tokens=200)
    state = owner.create_state()
    sync_history(
        state,
        [*(image_message(color) for color in ("red", "green", "blue")), {"role": "assistant", "content": "seen"}],
    )
    for _ in range(2):
        key = owner.policy_for(state, state.history).source_key
        request = await owner.prepare(state)
        assert request is not None
        assert len(request.prompt["multi_modal_data"]["image"]) == 1
        assert len(request.model_source.source.covered_ids) == 1
        assert owner.policy_for(state, state.history).source_key == key
        assert owner.apply(state, request, completion("visual fact"))
    assert state.visual.covered == 2 and not state.omissions


@pytest.mark.asyncio
async def test_fitted_token_batch_keeps_retry_identity_and_rejects_changed_evidence():
    owner = runtime(recent_tokens=1000, chunk_tokens=230, summary_tokens=64)
    state = owner.create_state()
    history = [message for _ in range(10) for message in turn(300)]
    sync_history(state, history)
    key = owner.policy_for(state, history, 1).source_key
    request = await owner.prepare(state)
    assert request is not None and request.model_source.end == 4
    assert not owner.apply(state, request, completion("truncated", "length"))
    history.append({"role": "system", "content": "unrelated later instruction"})
    sync_history(state, history)
    assert owner.policy_for(state, history, 2).source_key == key
    retry = await owner.prepare(state)
    assert retry.model_source == request.model_source
    history[0] = {"role": "user", "content": "corrected original fact " * 30}
    sync_history(state, history)
    assert not owner.apply(state, retry, completion("stale"))
    assert owner.policy_for(state, history, 3).source_key != key
    assert state.token_plan is None and not state.ledger.nodes()


@pytest.mark.asyncio
async def test_cached_token_plan_respects_custom_policy_deferral_after_rejection():
    owner = runtime(recent_tokens=128, chunk_tokens=256)
    owner.policy = IdleHistoryPolicy()
    state = owner.create_state()
    sync_history(state, turn(1600))
    request = await owner.prepare(state)
    assert request is not None
    assert not owner.apply(state, request, completion("truncated", "length"))
    assert owner.policy_for(state, state.history, 1) is None
    assert await owner.prepare(state) is None
    assert state.processed == 0 and not state.ledger.nodes()


@pytest.mark.asyncio
async def test_counting_preserves_retained_audio_when_the_selected_batch_is_text_only():
    owner = runtime(recent_tokens=10000, chunk_tokens=200, max_audio_items=2)
    state = owner.create_state()
    history, entries = turn(500), []
    for index in range(3):
        message, payload = audio_message(index)
        history.extend([message, {"role": "assistant", "content": "heard"}])
        entries.append(((message["content"],), payload))
    sync_history(state, history)
    state.retain(entries[-2:], owner.limits)
    retained = tuple(state.audio)
    request = await owner.prepare(state, lambda: entries)
    assert request is not None and request.model_source.end == 2
    assert not request.prompt["multi_modal_data"].get("audio")
    assert tuple(state.audio) == retained


@pytest.mark.asyncio
async def test_ready_text_batch_does_not_count_unrelated_later_media(monkeypatch):
    owner = runtime(recent_tokens=1000, chunk_tokens=400)
    state = owner.create_state()
    history = [message for _ in range(10) for message in turn(1200)]
    history.extend([image_message(), {"role": "assistant", "content": "image"}])
    sync_history(state, history)
    count = owner._source_tokens

    def fail_for_later_image(group):
        if group.images:
            raise MemoryError("later image counting failed")
        return count(group)

    monkeypatch.setattr(owner, "_source_tokens", fail_for_later_image)
    request = await owner.prepare(state)
    assert request is not None and request.model_source.end == 2
    assert not request.prompt["multi_modal_data"].get("image")


@pytest.mark.asyncio
@pytest.mark.parametrize("chunk_tokens", [None, 256])
async def test_session_rechecks_token_trigger_after_growth_without_recounting_unchanged_history(chunk_tokens):
    from tests.model_executor.models.joyai_vl_interaction.test_duplex_memory import start_summary
    from tests.model_executor.models.qwen3_omni.test_duplex_plugin import open_qwen
    from vllm_omni.engine.duplex.commands import CreateItem, Heartbeat

    h = await open_qwen()
    try:
        owner = runtime(recent_tokens=512, chunk_tokens=chunk_tokens)
        h.runner.plugin._history_runtime = owner
        h.runner.plugin.supports_auxiliary_requests = True
        for _ in range(4):
            for message in turn(1):
                h.session.append_history_message(message)
        await h.run(Heartbeat())
        assert not h.port.submissions
        revision = h.runner.model_state.history_policy_state.media_revision
        await h.run(Heartbeat())
        assert not h.port.submissions
        assert h.runner.model_state.history_policy_state.media_revision == revision
        h.submit(
            CreateItem(
                item={
                    "id": "later-system",
                    "type": "message",
                    "role": "system",
                    "content": [{"type": "input_text", "text": "x" * 2400}],
                }
            )
        )
        submission = await start_summary(h)
        assert submission.context.stage_id == 0 and not submission.resumable
        assert not h.runner.model_state.history_policy_state.omissions
    finally:
        await h.manager.shutdown()
