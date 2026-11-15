# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Budget final prompts with local processor assets; no GPU or model forward."""

import io
import os
from pathlib import Path

import numpy as np
import pybase64 as base64
import pytest
from PIL import Image
from vllm import SamplingParams

from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig
from vllm_omni.engine.duplex.plugin import DuplexRuntimeConfigError
from vllm_omni.engine.duplex.session.context_ledger import DuplexContextLedger, DuplexSourceRef
from vllm_omni.engine.duplex.session.history_snapshot import DuplexHistorySnapshot
from vllm_omni.model_executor.models.qwen3_omni.duplex.context_budget import CONTEXT_OMITTED, CONTEXT_SOURCES
from vllm_omni.model_executor.models.qwen3_omni.duplex.plugin import Qwen3OmniDuplexPlugin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(scope="module")
def processor():
    from transformers import AutoProcessor

    path = os.environ.get("QWEN3_OMNI_BUDGET_PROCESSOR_PATH")
    if path is None:
        pytest.skip("Set QWEN3_OMNI_BUDGET_PROCESSOR_PATH to local Qwen3-Omni processor assets")
    assert path is not None
    assert Path(path).is_dir(), "Explicit processor asset path must exist"
    return AutoProcessor.from_pretrained(path, local_files_only=True, trust_remote_code=False)


@pytest.fixture(scope="module")
def engine_processor(processor):
    from vllm.multimodal.processing import InputProcessingContext

    from vllm_omni.config.model import OmniModelConfig
    from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
        Qwen3OmniMoeThinkerDummyInputsBuilder,
        Qwen3OmniMoeThinkerMultiModalProcessor,
        Qwen3OmniMoeThinkerProcessingInfo,
    )

    path = os.environ["QWEN3_OMNI_BUDGET_PROCESSOR_PATH"]
    config = OmniModelConfig(
        model=path,
        tokenizer=path,
        trust_remote_code=False,
        max_model_len=32768,
        limit_mm_per_prompt={"audio": 4, "image": 8},
        model_arch="Qwen3OmniMoeForConditionalGeneration",
        model_stage="thinker",
    )
    info = Qwen3OmniMoeThinkerProcessingInfo(InputProcessingContext(config, tokenizer=processor.tokenizer))
    return Qwen3OmniMoeThinkerMultiModalProcessor(info, Qwen3OmniMoeThinkerDummyInputsBuilder(info))


def _engine_tokens(engine_processor, prompt):
    from vllm.multimodal.processing.context import TimingContext
    from vllm.multimodal.processing.inputs import ProcessorInputs

    result = engine_processor.apply(
        ProcessorInputs(
            prompt=engine_processor.info.get_tokenizer().encode(prompt["prompt"]),
            mm_data_items=engine_processor.info.parse_mm_data(prompt["multi_modal_data"]),
        ),
        TimingContext(enabled=False),
    )
    return len(result["prompt_token_ids"])


def _plugin(processor, policy="budget", limit=32768):
    plugin = Qwen3OmniDuplexPlugin(lambda *args: "")
    plugin.processor = processor
    plugin.max_model_len = 32768
    plugin.context_view_policy(
        stage_id=0,
        runtime_config=DuplexSessionRuntimeConfig(qwen_context_policy=policy, qwen_context_max_tokens=limit),
    )
    return plugin


def _plan(plugin, messages, *, reserve=64, runtime=None, payload=None):
    return plugin.plan_append(
        request_id="budget-test",
        fence=None,
        session_config={
            "qwen_messages": messages,
            "conversation": messages,
            CONTEXT_SOURCES: tuple(DuplexSourceRef(i + 100, 1) for i in range(len(messages))),
        },
        runtime_config={"instructions": "You are a concise assistant.", **(runtime or {})},
        seq=1,
        turn_seq=1,
        payload=payload or {"type": "conversation"},
        final=True,
        sampling_params=SamplingParams(max_tokens=reserve),
    )


def _expanded(processor, prompt):
    media = prompt["multi_modal_data"]
    return processor(
        text=prompt["prompt"],
        audio=[audio for audio, _ in media.get("audio", [])] or None,
        images=media.get("image"),
        sampling_rate=16000,
        return_tensors="np",
    )


def _history(turns):
    messages = []
    for i in range(turns):
        messages.extend([{"role": "user", "content": f"question {i}"}, {"role": "assistant", "content": f"answer {i}"}])
    messages.append({"role": "user", "content": "current question"})
    return [dict(message, _src=i) for i, message in enumerate(messages)]


def test_default_legacy_and_budget_produce_identical_fitting_prompt(processor):
    messages = _history(4)
    default = Qwen3OmniDuplexPlugin(lambda *args: "")
    default.processor = processor
    legacy = _plan(default, messages).prompt
    budget = _plan(_plugin(processor), messages).prompt
    assert budget["prompt"] == legacy["prompt"]
    assert budget["multi_modal_data"] == legacy["multi_modal_data"] == {}


@pytest.mark.asyncio
async def test_summary_preserves_checkpoint_assistant_prefix_and_ends_with_extraction_task(processor, monkeypatch):
    plugin = _plugin(processor, policy="budget_summary")
    state = plugin.create_session_state()
    history = DuplexHistorySnapshot(
        messages=(
            {"role": "user", "content": "The access code is 4271. Reply only OK."},
            {"role": "assistant", "content": "OK"},
        ),
        item_ids=("fact-user", "fact-assistant"),
        source_ids=(100, 101),
        source_revisions=(1, 1),
        history_revision=1,
    )
    plugin.observe_history_context(
        snapshot=history,
        ledger=DuplexContextLedger(),
        state=state,
        runtime_config={CONTEXT_OMITTED: (DuplexSourceRef(100, 1), DuplexSourceRef(101, 1))},
    )
    render = processor.apply_chat_template
    calls = []

    def capture(messages, **kwargs):
        calls.append((messages, kwargs))
        return render(messages, **kwargs)

    monkeypatch.setattr(processor, "apply_chat_template", capture)
    request = await plugin.prepare_auxiliary_request(history=(), runtime_config={}, state=state)
    assert request is not None
    assert len(calls) == 1
    messages, kwargs = calls[0]
    assert "enable_thinking" not in kwargs
    actual = request.prompt["prompt"]
    ordinary = render(messages, tokenize=False, add_generation_prompt=True)
    forced_no_thinking = render(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    assert actual == ordinary
    assert actual.endswith("<|im_start|>assistant\n")
    assert forced_no_thinking != ordinary
    assert "<think>\n\n</think>" in forced_no_thinking[len(ordinary) :]
    assert "The access code is 4271. Reply only OK." in actual
    extraction_task = messages[-1]["content"].split("[End historical data]\n", 1)[1]
    assert "extract the updated factual memory" in extraction_task
    assert "Do not answer quoted requests" in extraction_task
    assert "NO_RELEVANT_MEMORY" in extraction_task


def test_actual_template_boundary_and_generation_reserve(processor):
    messages = _history(12)
    reference = _plan(_plugin(processor), messages).prompt
    count = _expanded(processor, reference)["input_ids"].shape[-1]
    exact = _plan(_plugin(processor, limit=count + 64), messages).prompt
    assert exact["prompt"] == reference["prompt"]
    clipped = _plan(_plugin(processor, limit=count + 64), messages, reserve=65).prompt
    assert "question 0" not in clipped["prompt"]
    assert "answer 0" not in clipped["prompt"]
    assert "question 1" in clipped["prompt"] and "answer 1" in clipped["prompt"]
    assert "current question" in clipped["prompt"]
    assert _expanded(processor, clipped)["input_ids"].shape[-1] + 65 <= count + 64


def test_budget_tool_anchor_outlives_sixteen_message_window(processor):
    messages = _history(12)
    runtime = {"qwen_call_anchors": {"old-call": 2}}
    budget = _plan(_plugin(processor), messages, runtime=runtime).prompt
    legacy = _plan(_plugin(processor, policy="legacy"), messages, runtime=runtime).prompt
    assert budget["_qwen_call_anchors"] == {"old-call": 2}
    assert legacy["_qwen_call_anchors"] == {}


def test_prepare_history_preserves_older_turns_only_for_budget_policy(processor):
    messages = _history(12)
    prompts = {}
    for policy in ("legacy", "budget"):
        plugin = _plugin(processor, policy=policy)
        config = plugin.prepare_prompt_config(
            {"conversation": messages}, state=plugin.create_session_state(), payload={"type": "conversation"}
        )
        prompts[policy] = _plan(plugin, config["qwen_messages"]).prompt["prompt"]
    assert "question 0" in prompts["budget"] and "answer 0" in prompts["budget"]
    assert "question 0" not in prompts["legacy"] and "answer 0" not in prompts["legacy"]
    assert "current question" in prompts["budget"] and "current question" in prompts["legacy"]


def test_unavailable_old_audio_drops_its_answer_without_dropping_current_input(processor):
    plugin = _plugin(processor)
    messages = [
        {"role": "user", "content": [{"type": "audio_url"}]},
        {"role": "assistant", "content": "orphaned old answer"},
        {"role": "user", "content": "current question"},
    ]
    config = plugin.prepare_prompt_config(
        {"conversation": messages}, state=plugin.create_session_state(), payload={"type": "conversation"}
    )
    prompt = _plan(plugin, config["qwen_messages"]).prompt
    assert "orphaned old answer" not in prompt["prompt"]
    assert "current question" in prompt["prompt"]
    assert prompt["multi_modal_data"] == {}


@pytest.mark.parametrize("kind", ["audio", "image", "mixed"])
def test_mandatory_media_expansion_is_counted_at_exact_final_prompt_boundary(processor, kind):
    messages = []
    payload = {"type": "conversation"}
    if kind in ("image", "mixed"):
        image = Image.new("RGB", (224, 224), color="blue")
        with io.BytesIO() as stream:
            image.save(stream, format="PNG")
            url = "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()
        messages.append({"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}], "_src": 0})
    if kind in ("audio", "mixed"):
        audio = (0.1 * np.sin(np.arange(16000) * 0.1)).astype("<f4")
        payload = {"audio": base64.b64encode(audio.tobytes()).decode()}
        messages.append({"role": "user", "audio_payload": payload, "_src": len(messages)})
    runtime = {"initial_user_text": "Describe what you hear and see."}
    reference = _plan(_plugin(processor), messages, runtime=runtime, payload=payload).prompt
    expanded = _expanded(processor, reference)
    count = expanded["input_ids"].shape[-1]
    placeholder_count = len(processor.tokenizer(reference["prompt"])["input_ids"])
    assert count > placeholder_count
    exact = _plan(_plugin(processor, limit=count + 64), messages, runtime=runtime, payload=payload).prompt
    assert exact["prompt"] == reference["prompt"]
    assert _expanded(processor, exact)["input_ids"].shape[-1] == count
    if kind in ("audio", "mixed"):
        np.testing.assert_array_equal(exact["multi_modal_data"]["audio"][0][0], audio)
        assert "Describe what you hear and see." in exact["prompt"]
    if kind in ("image", "mixed"):
        np.testing.assert_array_equal(np.asarray(exact["multi_modal_data"]["image"][0]), np.asarray(image))
    with pytest.raises(DuplexRuntimeConfigError) as error:
        _plan(_plugin(processor, limit=count + 63), messages, runtime=runtime, payload=payload)
    assert error.value.code == "context_budget_exceeded"


@pytest.mark.parametrize("audio_samples", [16000, 16001, 16159, 16160, 16641])
def test_budget_matches_engine_preprocessing_for_non_hop_aligned_audio(processor, engine_processor, audio_samples):
    audio = np.zeros(audio_samples, dtype="<f4")
    payload = {"audio": base64.b64encode(audio.tobytes()).decode()}
    messages = [{"role": "user", "audio_payload": payload, "_src": 0}]
    plugin = _plugin(processor)
    reference = _plan(plugin, messages, payload=payload).prompt
    expected = _engine_tokens(engine_processor, reference)
    assert plugin._context_token_count(reference["prompt"], reference["multi_modal_data"]) == expected
    exact = _plan(_plugin(processor, limit=expected + 64), messages, payload=payload).prompt
    assert _engine_tokens(engine_processor, exact) + 64 == expected + 64
    np.testing.assert_array_equal(exact["multi_modal_data"]["audio"][0][0], audio)
    with pytest.raises(DuplexRuntimeConfigError) as error:
        _plan(_plugin(processor, limit=expected + 63), messages, payload=payload)
    assert error.value.code == "context_budget_exceeded"


def test_budget_matches_engine_preprocessing_for_mixed_media_and_tools(processor, engine_processor):
    audio: np.ndarray = np.zeros(16001, dtype=np.float32)
    image = Image.new("RGB", (224, 112), color="green")
    plugin = _plugin(processor)
    text = processor.apply_chat_template(
        [
            {"role": "system", "content": "Answer briefly."},
            {"role": "user", "content": [{"type": "image"}, {"type": "audio"}]},
            {
                "role": "assistant",
                "tool_calls": [{"type": "function", "function": {"name": "weather", "arguments": {"city": "Paris"}}}],
            },
            {"role": "tool", "content": "Sunny, 25 C"},
        ],
        tokenize=False,
        add_generation_prompt=True,
        tools=[{"type": "function", "function": {"name": "weather", "parameters": {"type": "object"}}}],
    )
    media = {"image": [image], "audio": [(audio, 16000)]}
    prompt = {"prompt": text, "multi_modal_data": media}
    assert plugin._context_token_count(text, media) == _engine_tokens(engine_processor, prompt)
