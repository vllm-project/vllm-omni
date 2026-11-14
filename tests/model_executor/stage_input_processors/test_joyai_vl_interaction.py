# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for the JoyAI stage input processors."""

from types import SimpleNamespace

import pytest

from vllm_omni.engine.stage_engine_core_client import StageEngineCoreClient
from vllm_omni.experimental.fullduplex.joyvl.decision.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    USER_QUERY_HEADER,
)
from vllm_omni.model_executor.stage_input_processors import (
    joyai_vl_interaction as joyai_bridge,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_IMAGE = "<|vision_start|><|image_pad|><|vision_end|>"
_VIDEO = "<|vision_start|><|video_pad|><|vision_end|>"


@pytest.mark.parametrize("salt", [None, "tenant-a", "tenant-b"])
def test_asr_to_joyai_preserves_cache_salt(salt):
    prompt = {"cache_salt": salt} if salt is not None else {}
    output = SimpleNamespace(outputs=[SimpleNamespace(text="hello")])
    (result,) = joyai_bridge.asr_to_joyai([output], prompt)
    if salt is None:
        assert "cache_salt" not in result
    else:
        assert result["cache_salt"] == salt


def test_asr_to_joyai_combines_transcript_with_deferred_visuals() -> None:
    """The cleaned transcript becomes JoyAI's user query, followed by the deferred visuals."""
    client = object.__new__(StageEngineCoreClient)
    client.custom_process_input_func = joyai_bridge.asr_to_joyai
    client.requires_multimodal_data = True
    client.vllm_config = SimpleNamespace(model_config=None)

    def joyai_prompt(system_prompt: str, user_turn: str) -> str:
        return (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_turn}<|im_end|>\n<|im_start|>assistant\n"
        )

    cases = [
        (
            SimpleNamespace(text="", cumulative_text="language English<asr_text>Is the stove  still on?"),
            {
                "multi_modal_data": {"audio": ("wave", 16000)},
                "additional_information": {"deferred_multi_modal_data": {"image": ["frame-0", "frame-1"]}},
            },
            {
                "prompt": joyai_prompt(
                    DEFAULT_SYSTEM_PROMPT, f"{USER_QUERY_HEADER}\nIs the stove still on?{_IMAGE}{_IMAGE}"
                ),
                "multi_modal_data": {"image": ["frame-0", "frame-1"]},
            },
        ),
        (
            SimpleNamespace(text="What is burning?"),
            {
                "mm_processor_kwargs": {"fps": 1},
                "additional_information": {
                    "deferred_multi_modal_data": {"video": [("frames", {"fps": 1})]},
                    "joyai_system_prompt": ["Watch the feed."],
                },
            },
            {
                "prompt": joyai_prompt("Watch the feed.", f"{USER_QUERY_HEADER}\nWhat is burning?{_VIDEO}"),
                "multi_modal_data": {"video": [("frames", {"fps": 1})]},
                "mm_processor_kwargs": {"fps": 1},
            },
        ),
    ]
    for completion, prompt, expected in cases:
        assert client.process_engine_inputs([SimpleNamespace(outputs=[completion])], prompt=prompt) == [expected]


def test_asr_handoff_preserves_session_prompt_and_checks_final_budget(monkeypatch):
    monkeypatch.setattr(
        joyai_bridge,
        "cached_processor_from_config",
        lambda config: SimpleNamespace(
            apply_chat_template=lambda messages, **kwargs: repr(messages),
            tokenizer=lambda **kwargs: {"input_ids": [list(kwargs["text"][0])]},
        ),
    )
    prompt = {
        "additional_information": {
            "joyai_prompt_context": {
                "system": "standing instruction",
                "history": [{"role": "assistant", "content": "played history"}],
                "query": "",
            },
            "joyai_prompt_budget": 500,
            "joyai_image_token_counts": [21],
            "deferred_multi_modal_data": {"image": ["original-frame"]},
        }
    }
    output = SimpleNamespace(outputs=[SimpleNamespace(text="language English<asr_text>Watch the pot.")])
    (result,) = joyai_bridge.asr_to_joyai([output], prompt)
    assert all(text in result["prompt"] for text in ("standing instruction", "played history", "Watch the pot."))
    assert result["multi_modal_data"] == {"image": ["original-frame"]}
    prompt["additional_information"]["joyai_prompt_budget"] = 30
    with pytest.raises(joyai_bridge.OmniClientError, match="context budget"):
        joyai_bridge.asr_to_joyai([output], prompt)


@pytest.mark.parametrize(
    "metadata",
    [
        {"joyai_prompt_context": {}},
        {"joyai_prompt_context": None},
        {"joyai_prompt_context": []},
        {"joyai_prompt_context": {"system": "policy"}},
        {"joyai_prompt_context": {"system": 1, "history": []}},
        {"joyai_prompt_context": {"system": "policy", "history": {}}},
        {"joyai_prompt_context": {"system": "policy", "history": [], "instruction": []}},
        {"joyai_prompt_context": {"system": "policy", "history": [], "summary": None}},
        {"joyai_prompt_context": {"system": "policy", "history": [], "visual_summaries": None}},
        {"joyai_prompt_context": {"system": "policy", "history": [], "visual_summaries": "scene"}},
        {"joyai_prompt_context": {"system": "policy", "history": [], "visual_summaries": [None]}},
        {"joyai_prompt_context": {"system": "policy", "history": [None]}},
        {"joyai_prompt_context": {"system": "policy", "history": [{"content": "hello"}]}},
        {"joyai_prompt_context": {"system": "policy", "history": [{"role": [], "content": "hello"}]}},
        {"joyai_prompt_context": {"system": "policy", "history": [{"role": "user", "content": None}]}},
        {"joyai_prompt_budget": None},
        {"joyai_prompt_budget": "invalid"},
        {"joyai_prompt_budget": True},
        {"joyai_prompt_budget": 0},
        {"joyai_image_token_counts": None},
        {"joyai_image_token_counts": ["21"]},
        {"joyai_image_token_counts": [-1]},
        {"joyai_image_token_counts": []},
    ],
)
def test_asr_handoff_rejects_malformed_session_metadata(monkeypatch, metadata):
    # No tokenizer is needed to reject invalid client-supplied metadata.
    monkeypatch.setattr(joyai_bridge, "cached_processor_from_config", lambda _: None)
    info = {
        "joyai_prompt_context": {"system": "policy", "history": []},
        "joyai_prompt_budget": 500,
        "joyai_image_token_counts": [21],
        "deferred_multi_modal_data": {"image": ["frame"]},
        **metadata,
    }
    output = SimpleNamespace(outputs=[SimpleNamespace(text="What is this?")])
    with pytest.raises(joyai_bridge.OmniClientError, match="joyai_") as error:
        joyai_bridge.asr_to_joyai([output], {"additional_information": info})
    assert error.value.status_code == 400


def test_joyai_action_to_tts_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    """Convert each JoyAI action through the real stage-client entry point."""
    target_model_config = SimpleNamespace(
        hf_config=SimpleNamespace(talker_config=SimpleNamespace(spk_id={"vivian": 1, "ryan": 2}))
    )
    prompt_length_calls: list[tuple[dict[str, list[str]], object]] = []
    prompt_lengths = {
        "Colored squares are visible.": 12,
        "I am checking this request now.": 18,
        "Why don't scientists trust atoms? Because they make up everything!": 24,
    }

    def compute_prompt_length(metadata, model_config):
        prompt_length_calls.append((metadata, model_config))
        return prompt_lengths[metadata["text"][0]]

    monkeypatch.setattr(
        joyai_bridge,
        "_compute_talker_prompt_length",
        compute_prompt_length,
    )

    client = object.__new__(StageEngineCoreClient)
    client.custom_process_input_func = joyai_bridge.joyai_action_to_tts
    client.requires_multimodal_data = False
    client.vllm_config = SimpleNamespace(model_config=target_model_config)

    cases = [
        (
            "</response> Why don't scientists trust atoms?\n\nBecause they make up everything!",
            None,
            {},
            ("Why don't scientists trust atoms? Because they make up everything!", "Auto", "Vivian"),
        ),
        (
            "</response> Colored squares are visible.",
            None,
            {
                "tts_task_type": "Base",
                "language": ["Chinese"],
                "speaker": ["Ryan"],
            },
            ("Colored squares are visible.", "Chinese", "Ryan"),
        ),
        ("</silence>", None, {}, None),
        (
            "",
            (
                "</response> I am checking this request now. "
                "</delegation> Why is the purple elephant dancing beside a volcano?"
            ),
            {
                "language": "Chinese",
                "speaker": "Ryan",
                "tts_language": "English",
                "tts_speaker": "Vivian",
            },
            ("I am checking this request now.", "English", "Vivian"),
        ),
    ]

    for text, cumulative_text, additional_information, expected in cases:
        completion = SimpleNamespace(text=text)
        if cumulative_text is not None:
            completion.cumulative_text = cumulative_text

        call_count = len(prompt_length_calls)
        talker_inputs = client.process_engine_inputs(
            [SimpleNamespace(outputs=[completion])],
            prompt={"additional_information": additional_information},
        )

        if expected is None:
            assert talker_inputs == []
            assert len(prompt_length_calls) == call_count
            continue

        spoken_text, language, speaker = expected
        [talker_input] = talker_inputs
        metadata = talker_input["additional_information"]
        assert talker_input["prompt_token_ids"] == [0] * prompt_lengths[spoken_text]
        assert metadata == {
            "task_type": ["CustomVoice"],
            "language": [language],
            "speaker": [speaker],
            "instruct": [""],
            "text": [spoken_text],
        }
        assert talker_input["model_intermediate_buffer"] == metadata
        assert prompt_length_calls[-1] == (metadata, target_model_config)


def test_joyai_action_to_tts_takes_the_whole_utterance() -> None:
    """JoyAI→Talker stays orchestrator-fed; there is no thinker2talker async_chunk."""
    assert not hasattr(joyai_bridge, "joyai_action_to_tts_async_chunk")


def test_joyai_action_to_tts_close_only_keeps_a_one_token_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_omni.model_executor.models.joyai_vl_interaction.duplex.sentence_tts import (
        JOYAI_TTS_CLOSE_ONLY_KEY,
        JOYAI_TTS_PARTIAL_KEY,
        SentenceTtsOutput,
    )

    target_model_config = SimpleNamespace(
        hf_config=SimpleNamespace(talker_config=SimpleNamespace(spk_id={"vivian": 1}))
    )

    def fail_prompt_length(*_args, **_kwargs):
        raise AssertionError("close_only must not estimate a Talker prompt")

    monkeypatch.setattr(joyai_bridge, "_compute_talker_prompt_length", fail_prompt_length)
    talker_inputs = joyai_bridge.joyai_action_to_tts(
        [SentenceTtsOutput("req-1", "")],
        prompt={
            "additional_information": {
                JOYAI_TTS_PARTIAL_KEY: True,
                JOYAI_TTS_CLOSE_ONLY_KEY: True,
            }
        },
        target_model_config=target_model_config,
    )
    [talker_input] = talker_inputs
    assert talker_input["prompt_token_ids"] == [0]
    assert talker_input["additional_information"]["text"] == [""]
    assert "meta" not in talker_input["additional_information"]


def test_joyai_action_to_tts_partial_keeps_the_sentence_view(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_omni.model_executor.models.joyai_vl_interaction.duplex.sentence_tts import (
        JOYAI_TTS_CLOSE_ONLY_KEY,
        JOYAI_TTS_PARTIAL_KEY,
        SentenceTtsOutput,
    )

    target_model_config = SimpleNamespace(
        hf_config=SimpleNamespace(talker_config=SimpleNamespace(spk_id={"vivian": 1}))
    )
    prompt_lengths: dict[str, int] = {}

    def fake_prompt_length(additional_information, talker_model_config):
        del talker_model_config
        spoken = additional_information["text"][0]
        prompt_lengths[spoken] = 4
        return 4

    monkeypatch.setattr(joyai_bridge, "_compute_talker_prompt_length", fake_prompt_length)
    talker_inputs = joyai_bridge.joyai_action_to_tts(
        [SentenceTtsOutput("req-1", "水开了。")],
        prompt={
            "additional_information": {
                JOYAI_TTS_PARTIAL_KEY: True,
                JOYAI_TTS_CLOSE_ONLY_KEY: False,
                "speaker": ["Vivian"],
            }
        },
        target_model_config=target_model_config,
    )
    [talker_input] = talker_inputs
    assert talker_input["additional_information"]["text"] == ["水开了。"]
    assert talker_input["prompt_token_ids"] == [0] * 4
    assert talker_input["additional_information"]["meta"]["replace_streaming_prompt"] is True
    assert talker_input["additional_information"]["meta"]["replace_runtime_additional_information"] is True
    assert "joyai_finish_on_next_stop" not in talker_input["additional_information"]["meta"]
    assert talker_input["additional_information"]["embed"]["prefill"] == []
    assert talker_input["additional_information"]["hidden_states"]["trailing_text"] == []
    assert talker_input["model_intermediate_buffer"] == talker_input["additional_information"]
    from vllm_omni.core.sched.omni_ar_scheduler import _streaming_update_replaces_prompt
    from vllm_omni.data_entry_keys import serialize_payload

    payload = serialize_payload(talker_input["additional_information"])
    assert _streaming_update_replaces_prompt((None, payload)) is True
    assert _streaming_update_replaces_prompt((None, serialize_payload({"text": ["水开了。"]}))) is False


def test_joyai_action_to_tts_stamps_finish_on_next_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_omni.core.sched.omni_scheduler_mixin import _request_finishes_on_next_stop
    from vllm_omni.model_executor.models.joyai_vl_interaction.duplex.sentence_tts import (
        JOYAI_TTS_CLOSE_ONLY_KEY,
        JOYAI_TTS_FINISH_ON_NEXT_STOP_KEY,
        JOYAI_TTS_PARTIAL_KEY,
        SentenceTtsOutput,
    )

    target_model_config = SimpleNamespace(
        hf_config=SimpleNamespace(talker_config=SimpleNamespace(spk_id={"vivian": 1}))
    )
    monkeypatch.setattr(joyai_bridge, "_compute_talker_prompt_length", lambda *_args, **_kwargs: 4)
    talker_inputs = joyai_bridge.joyai_action_to_tts(
        [SentenceTtsOutput("req-1", "关火。")],
        prompt={
            "additional_information": {
                JOYAI_TTS_PARTIAL_KEY: True,
                JOYAI_TTS_CLOSE_ONLY_KEY: False,
                JOYAI_TTS_FINISH_ON_NEXT_STOP_KEY: True,
                "speaker": ["Vivian"],
            }
        },
        target_model_config=target_model_config,
    )
    [talker_input] = talker_inputs
    assert talker_input["additional_information"]["meta"][JOYAI_TTS_FINISH_ON_NEXT_STOP_KEY] is True
    request = SimpleNamespace(
        additional_information=talker_input["additional_information"],
        model_intermediate_buffer=talker_input["model_intermediate_buffer"],
    )
    assert _request_finishes_on_next_stop(request) is True
    from vllm_omni.data_entry_keys import serialize_payload, to_struct

    # to_struct used to reject this unknown meta key; the last-sentence 2150
    # then never reached Code2Wav as a request terminal. Convert only the
    # known MetaStruct subset: talker_text_offset is Talker-local and is not
    # on the struct (flatten_payload still carries it on the wire).
    structured = to_struct(
        {
            "meta": {
                "replace_streaming_prompt": True,
                "replace_runtime_additional_information": True,
                "talker_prefill_offset": 0,
                JOYAI_TTS_FINISH_ON_NEXT_STOP_KEY: True,
            }
        }
    )
    assert structured.meta is not None
    assert structured.meta.joyai_finish_on_next_stop is True
    wire = SimpleNamespace(
        additional_information=serialize_payload(talker_input["additional_information"]),
        model_intermediate_buffer=None,
    )
    assert _request_finishes_on_next_stop(wire) is True
    top_level = SimpleNamespace(
        additional_information={JOYAI_TTS_FINISH_ON_NEXT_STOP_KEY: True},
        model_intermediate_buffer=None,
    )
    assert _request_finishes_on_next_stop(top_level) is True
    absent = SimpleNamespace(additional_information={"meta": {}}, model_intermediate_buffer=None)
    assert _request_finishes_on_next_stop(absent) is False


def test_talker2code2wav_empty_finished_preserves_finish_on_next_stop() -> None:
    from collections import defaultdict

    import torch

    from vllm_omni.data_entry_keys import JOYAI_FINISH_ON_NEXT_STOP_KEY
    from vllm_omni.model_executor.stage_input_processors.qwen3_tts import talker2code2wav_async_chunk

    extra = {
        "codec_chunk_frames": 25,
        "codec_left_context_frames": 25,
        "initial_codec_chunk_frames": 1,
    }
    tm = SimpleNamespace(
        code_prompt_token_ids=defaultdict(list),
        scheduler_max_num_seqs=1,
        put_req_chunk=defaultdict(int),
        ramp_chunk_count=defaultdict(int),
        _adaptive_states={},
        request_payload={},
        connector=SimpleNamespace(config={"extra": extra}),
    )
    request = SimpleNamespace(
        external_req_id="req-1",
        is_finished=lambda: False,
        additional_information=None,
        output_token_ids=[],
    )
    setattr(request, f"_{JOYAI_FINISH_ON_NEXT_STOP_KEY}", True)
    first = talker2code2wav_async_chunk(
        tm,
        {"codes": {"audio": torch.tensor([[1, 2, 3, 4]], dtype=torch.long)}},
        request,
    )
    assert first is not None
    tm.put_req_chunk["req-1"] += 1
    request.is_finished = lambda: True
    terminal = talker2code2wav_async_chunk(
        tm,
        {"codes": {"audio": torch.zeros((0,))}},
        request,
        is_finished=True,
    )
    assert terminal is not None
    assert terminal.codes.audio.numel() == 0
    assert terminal.meta.joyai_finish_on_next_stop is True
