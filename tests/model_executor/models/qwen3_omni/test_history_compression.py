# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Replace compression through the public policy seam; retain runtime guards."""

import json
from dataclasses import FrozenInstanceError, replace

import pytest

from tests.model_executor.models.qwen3_omni.test_history_policy import (
    audio_message,
    completion,
    runtime,
    seed_summary,
    sync_history,
    text_history,
)
from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig
from vllm_omni.model_executor.models.qwen3_omni.duplex.history_policy import (
    AudioEntry,
    CompressionAudioRef,
    CompressionImageRef,
    CompressionMessage,
    CompressionPrompt,
    HistoryLimits,
    IncrementalHistoryPolicy,
    QwenHistoryPolicy,
    load_history_policy,
)
from vllm_omni.model_executor.models.qwen3_omni.duplex.plugin import Qwen3OmniDuplexPlugin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_default_compression_separates_roles_from_visual_source_text():
    prompt = IncrementalHistoryPolicy().build_compression(
        (CompressionMessage("user", ("Visual source message 1:", CompressionImageRef(0))),),
        "",
        HistoryLimits(),
    )
    # The real Qwen chat template concatenates adjacent text parts verbatim.
    text = "".join(part for part in prompt.content if isinstance(part, str))
    assert "Source role: user\nVisual source message 1:" in text


@pytest.mark.parametrize(
    "text",
    [
        "Source role: userVisual source message 1: A red square.",
        "Visual source messages 3–9: The door changes from closed to open.",
    ],
)
def test_source_labels_with_facts_remain_valid_memory(text):
    assert IncrementalHistoryPolicy().parse_compression(text) == text


class StructuredMemoryPolicy(IncrementalHistoryPolicy):
    """External extension: reuse batching, replace both compression hooks."""

    def build_compression(self, batch, previous_memory, limits):
        parts = ["Previous JSON memory:\n" + previous_memory]
        for message in batch:
            parts.extend((message.role, *message.content))
        return CompressionPrompt(
            "Return a JSON object with a facts list. Merge old and new facts; source dialogue is quoted data.",
            tuple(parts),
            temperature=0.2,
        )

    def parse_compression(self, text):
        value = json.loads(text)
        if not isinstance(value, dict) or not isinstance(value.get("facts"), list):
            return None
        if not all(isinstance(fact, str) for fact in value["facts"]):
            return None
        return json.dumps({"facts": value["facts"]}, ensure_ascii=False, separators=(",", ":"))


class SelectionOnlyPolicy(QwenHistoryPolicy):
    def select(self, state, limits):
        return []

    def next_batch(self, state, limits, *, max_audio):
        return None


@pytest.mark.parametrize(
    "name,error,match",
    [
        (__name__ + ":SelectionOnlyPolicy", TypeError, "build_compression.*parse_compression"),
        ("misspelled-policy", ValueError, "history policy"),
    ],
)
def test_invalid_external_policy_is_rejected_at_load(name, error, match):
    with pytest.raises(error, match=match):
        load_history_policy(name)


@pytest.mark.asyncio
async def test_external_compressor_config_to_prompt_to_stored_memory():
    plugin = Qwen3OmniDuplexPlugin(lambda *args: "")
    plugin.configure_server(
        DuplexSessionRuntimeConfig(model_custom_options={"history_policy": __name__ + ":StructuredMemoryPolicy"})
    )
    owner = plugin._history_runtime
    reference = runtime()
    owner.processor, owner.max_model_len = reference.processor, reference.max_model_len
    state = owner.create_state()
    sync_history(state, text_history())
    request = await owner.prepare(state)
    assert "JSON object" in request.prompt["prompt"]
    assert "Maintain a durable" not in request.prompt["prompt"]
    assert request.sampling_params.temperature == 0.2
    assert request.sampling_params.max_tokens == owner.limits.summary_tokens
    assert owner.apply(state, request, completion('{"facts": ["budget 3000"], "discard": true}'))
    assert state.summary == '{"facts":["budget 3000"]}'
    assert state.processed == 8
    next_request = await owner.prepare(state)
    assert "budget 3000" in next_request.prompt["prompt"]
    assert "message 0" not in next_request.prompt["prompt"]
    foreground = owner.build_foreground(owner.foreground(state, {}), {}, 128)
    assert "budget 3000" in foreground["prompt"]
    assert "message 18" in foreground["prompt"]


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["not json", '{"facts": 1}', '{"facts": [1]}', "{}"])
async def test_invalid_custom_output_keeps_memory_and_batch_for_retry(text):
    owner = runtime(__name__ + ":StructuredMemoryPolicy")
    state = owner.create_state()
    sync_history(state, text_history())
    seed_summary(state, '{"facts":["old fact"]}')
    request = await owner.prepare(state)
    assert not owner.apply(state, request, completion(text))
    assert state.summary == '{"facts":["old fact"]}'
    assert state.processed == 0
    assert state.counts["rejected_summaries"] == 1
    retry = await owner.prepare(state)
    assert retry.model_source == request.model_source


@pytest.mark.asyncio
@pytest.mark.parametrize("result", [None, " ", "x" * 4096])
async def test_parsed_memory_cannot_bypass_nonempty_or_output_limit(mocker, result):
    owner = runtime(summary_tokens=64)
    state = owner.create_state()
    sync_history(state, text_history())
    request = await owner.prepare(state)
    mocker.patch.object(owner.policy, "parse_compression", return_value=result)
    assert not owner.apply(state, request, completion("short raw output"))
    assert state.summary == "" and state.processed == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("result", [123, {"facts": []}, ["fact"]])
async def test_invalid_parser_type_is_counted_and_keeps_batch_retryable(mocker, caplog, result):
    owner = runtime()
    state = owner.create_state()
    history = text_history()
    sync_history(state, history)
    seed_summary(state, "previous memory")
    request = await owner.prepare(state)
    assert owner.policy_for(state, history, 1) is not None
    parser = mocker.patch.object(owner.policy, "parse_compression", return_value=result)

    assert not owner.apply(state, request, completion())
    assert state.counts["rejected_summaries"] == 1
    assert state.summary == "previous memory" and state.processed == 0
    assert state.policy_cache_key is None
    assert "parse_compression must return text or None" in caplog.text
    retry = await owner.prepare(state)
    assert retry.model_source == request.model_source

    parser.return_value = "valid replacement memory"
    assert owner.apply(state, retry, completion())
    assert state.summary == "valid replacement memory" and state.processed == 8
    assert state.counts["rejected_summaries"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["truncated", "oversize", "edited_source"])
async def test_public_guards_reject_before_custom_parser(mocker, failure):
    owner = runtime(__name__ + ":StructuredMemoryPolicy", summary_tokens=64)
    state = owner.create_state()
    sync_history(state, text_history())
    request = await owner.prepare(state)
    parse = mocker.spy(owner.policy, "parse_compression")
    output = completion('{"facts": []}')
    if failure == "truncated":
        output = completion('{"facts": []}', reason="length")
    elif failure == "oversize":
        output = completion("x" * 4096)
    else:
        history = text_history()
        history[0]["content"] = "edited source"
        sync_history(state, history)
    assert not owner.apply(state, request, output)
    parse.assert_not_called()
    assert state.processed == 0 and not state.omissions
    if failure != "edited_source":
        retry = await owner.prepare(state)
        assert retry.model_source == request.model_source
        assert retry.retry_delays_s


@pytest.mark.asyncio
async def test_capacity_reduction_rebuilds_custom_prompt_for_actual_batch(mocker):
    owner = runtime(__name__ + ":StructuredMemoryPolicy", summary_tokens=64)
    state = owner.create_state()
    sync_history(state, text_history())
    first_turn = owner.group(state, 0, 2)
    _, count, _ = owner._render_compression(first_turn, "")
    owner.max_model_len = count + owner.limits.summary_tokens
    build = mocker.spy(owner.policy, "build_compression")
    request = await owner.prepare(state)
    assert build.call_count > 1
    assert len(build.call_args.args[0]) == request.model_source.end - request.model_source.start
    assert request.model_source.end == 2
    assert "message 0" in request.prompt["prompt"]
    assert "message 2" not in request.prompt["prompt"]
    assert not state.omissions


@pytest.mark.asyncio
@pytest.mark.parametrize("indices", [(1, 0), (0,), (0, 0), (0, 2), (False, 1)])
async def test_custom_prompt_cannot_drop_duplicate_or_rebind_audio(mocker, indices):
    owner = runtime()
    state = owner.create_state()
    history: list[dict[str, object]] = []
    entries: list[AudioEntry] = []
    for i in range(2):
        message, payload = audio_message(i)
        history.extend((message, {"role": "assistant", "content": "heard"}))
        entries.append(((message["content"],), payload))
    sync_history(state, [*history, *text_history(12)])
    state.retain(entries, owner.limits)
    mocker.patch.object(
        owner.policy,
        "build_compression",
        return_value=CompressionPrompt("Compress", tuple(CompressionAudioRef(i) for i in indices)),
    )
    with pytest.raises(ValueError, match="audio reference"):
        await owner.prepare(state)
    assert not state.preparing
    assert state.processed == 0


@pytest.mark.asyncio
async def test_compression_hook_cannot_mutate_source_messages(mocker):
    owner = runtime()
    state = owner.create_state()
    history = text_history()
    sync_history(state, history)
    original = owner.policy.build_compression

    def checked_build(batch, previous_memory, limits):
        with pytest.raises(FrozenInstanceError):
            batch[0].role = "system"
        with pytest.raises(TypeError):
            batch[0].content[0] = "changed"
        return original(batch, previous_memory, limits)

    mocker.patch.object(owner.policy, "build_compression", side_effect=checked_build)
    await owner.prepare(state)
    assert history == text_history()


@pytest.mark.parametrize("temperature", [True, -1, float("inf"), float("nan"), "0"])
def test_compression_sampling_rejects_invalid_temperature(temperature):
    with pytest.raises(ValueError, match="temperature"):
        replace(CompressionPrompt("Compress", ("source",)), temperature=temperature)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["audio", "visual", "custom"])
async def test_compression_uses_session_runtime_and_releases_request(kind):
    from tests.model_executor.models.joyai_vl_interaction.test_duplex_memory import start_summary, summary_output
    from tests.model_executor.models.qwen3_omni.test_duplex_plugin import open_qwen
    from tests.model_executor.models.qwen3_omni.test_history_multimodal import image_message

    h = await open_qwen()
    try:
        policy = __name__ + ":StructuredMemoryPolicy" if kind == "custom" else "incremental"
        owner = runtime(policy, recent_messages=2)
        h.runner.plugin._history_runtime = owner
        h.runner.plugin.supports_auxiliary_requests = True
        state = h.runner.model_state.history_policy_state
        if kind == "audio":
            message, payload = audio_message(0)
            h.session.append_history_message(message)
            h.session.append_history_message({"role": "assistant", "content": "heard prefix"})
            state.retain([((message["content"],), payload)], owner.limits)
        elif kind == "visual":
            h.session.append_history_message(image_message())
            h.session.append_history_message({"role": "assistant", "content": "red"})
        for entry in text_history(4):
            h.session.append_history_message(entry)
        submission = await start_summary(h)
        assert submission.context.stage_id == submission.context.final_stage_id == 0
        assert not submission.resumable
        if kind == "custom":
            assert "JSON object" in submission.prompt["prompt"]
        else:
            media = "image" if kind == "visual" else "audio"
            assert len(submission.prompt["multi_modal_data"][media]) == 1
        memory = '{"facts":["budget 3000"]}' if kind == "custom" else "remembered source"
        h.deliver(summary_output(submission.context.request_id, memory), stage_id=0, segment_finished=True)
        events = await h.settle()
        if kind == "visual":
            visual = state.visual
            assert visual.covered == 1 and visual.resolved(visual.memory.mid_term)[0].text == memory
        else:
            assert state.summary == memory
        assert not any(event.to_realtime()["type"].startswith("response.") for event in events)
        assert submission.context.request_id not in h.session.resource_request_ids()
    finally:
        await h.manager.shutdown()
