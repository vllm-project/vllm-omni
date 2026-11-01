# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Logic-equivalent checks adapted from AURA_026_p0exp stage processor tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_omni.model_executor.models.aura_omni.duplex.history import (
    drop_session_history,
    get_or_create_session_history,
)
from vllm_omni.model_executor.stage_input_processors.aura_omni import (
    SILENT_TEXT,
    asr2aura,
    aura2tts,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _source_output(text: str, request_id: str = "r0") -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id,
        outputs=[SimpleNamespace(text=text, token_ids=[1, 2, 3])],
    )


def test_asr2aura_carries_video_and_transcript() -> None:
    prompt = {
        "additional_information": {
            "aura_system_prompt": "sys",
            "deferred_multi_modal_data": {"image": ["frame"]},
        },
        "multi_modal_data": {},
    }
    [next_input] = asr2aura([_source_output("What is happening now?")], prompt=[prompt])
    assert "What is happening now?" in next_input["prompt"]
    assert next_input["multi_modal_data"].get("image") == ["frame"]


def test_asr2aura_supports_video_only_observation() -> None:
    prompt = {
        "additional_information": {
            "deferred_multi_modal_data": {"image": ["frame"]},
        },
        "multi_modal_data": {},
    }
    [next_input] = asr2aura([_source_output("")], prompt=[prompt])
    assert next_input["multi_modal_data"].get("image") == ["frame"]


def test_asr2aura_duplex_empty_transcript_is_not_a_vision_user_turn() -> None:
    drop_session_history("duplex-vision")
    prompt = {
        "additional_information": {
            "aura_duplex": True,
            "session_id": "duplex-vision",
            "is_speech": False,
            "deferred_multi_modal_data": {"image": ["frame"]},
        },
        "multi_modal_data": {},
    }
    [next_input] = asr2aura([_source_output("noise")], prompt=[prompt])
    assert "[vision]" not in next_input["prompt"]
    history = get_or_create_session_history("duplex-vision")
    history.commit_turn("saw a book")
    assert all(message.get("content") != "[vision]" for message in history.messages)
    assert [message["role"] for message in history.messages] == ["assistant"]
    drop_session_history("duplex-vision")


def test_asr2aura_duplex_history_videos_match_prompt_pads() -> None:
    """Retained clips then this clip: one <|video_pad|> each, same order as mm data."""
    drop_session_history("duplex-pads")
    history = get_or_create_session_history("duplex-pads")
    history.begin_user_turn("看着手", video=("clip-a", {"fps": 2.0}))
    history.commit_turn("好的")
    prompt = {
        "additional_information": {
            "aura_duplex": True,
            "session_id": "duplex-pads",
            "is_speech": False,
            "deferred_multi_modal_data": {"video": [("clip-b", {"fps": 2.0})]},
        },
        "multi_modal_data": {},
    }
    [next_input] = asr2aura([_source_output("")], prompt=[prompt])
    text = next_input["prompt"]
    assert text.count("<|video_pad|>") == 2
    assert "clip-a" not in text and "clip-b" not in text
    assert next_input["multi_modal_data"]["video"] == [
        ("clip-a", {"fps": 2.0}),
        ("clip-b", {"fps": 2.0}),
    ]
    assert text.index("<|video_pad|>") < text.rindex("<|im_start|>user")
    drop_session_history("duplex-pads")


def test_asr2aura_duplex_uses_session_history_prefix() -> None:
    drop_session_history("duplex-hist")
    history = get_or_create_session_history("duplex-hist")
    history.begin_user_turn("prev")
    history.commit_turn("prev-answer")
    prompt = {
        "additional_information": {
            "aura_duplex": True,
            "session_id": "duplex-hist",
            "deferred_multi_modal_data": {"image": ["frame"]},
        },
        "multi_modal_data": {},
    }
    [next_input] = asr2aura([_source_output("next question")], prompt=[prompt])
    assert "prev-answer" in next_input["prompt"]
    assert "next question" in next_input["prompt"]
    drop_session_history("duplex-hist")


def test_asr2aura_vision_follow_keeps_previous_assistant_in_prefix() -> None:
    drop_session_history("duplex-watch")
    history = get_or_create_session_history("duplex-watch")
    history.begin_user_turn("盯着滑鼠")
    history.commit_turn("好的，出现的时候我会说")
    prompt = {
        "additional_information": {
            "aura_duplex": True,
            "session_id": "duplex-watch",
            "is_speech": False,
            "deferred_multi_modal_data": {"image": ["frame"]},
        },
        "multi_modal_data": {},
    }
    [next_input] = asr2aura([_source_output("noise")], prompt=[prompt])
    assert "好的，出现的时候我会说" in next_input["prompt"]
    drop_session_history("duplex-watch")


def test_aura2tts_drops_silent_response() -> None:
    prompt = {"additional_information": {"tts_task_type": "CustomVoice"}}
    assert aura2tts([_source_output(SILENT_TEXT)], prompt=[prompt]) == []


def test_aura2tts_duplex_does_not_commit_history() -> None:
    """Stage1 history has one owner: commit_model_context. aura2tts must not write a second row."""
    drop_session_history("duplex-silent")
    history = get_or_create_session_history("duplex-silent")
    history.begin_user_turn("look")
    prompt = {
        "additional_information": {
            "aura_duplex": True,
            "session_id": "duplex-silent",
            "tts_task_type": "CustomVoice",
        }
    }
    assert aura2tts([_source_output(SILENT_TEXT)], prompt=[prompt]) == []
    assert all(message.get("role") != "assistant" for message in history.messages)
    from vllm_omni.model_executor.models.aura_omni.duplex.plugin import AuraDuplexPlugin

    AuraDuplexPlugin(encode_audio=lambda *_a, **_k: None).commit_model_context(
        session_id="duplex-silent",
        assistant_text="只記一次",
    )
    assert [message["role"] for message in history.messages] == ["user", "assistant"]
    assert history.messages[-1]["content"] == "只記一次"
    drop_session_history("duplex-silent")


def test_aura2tts_skips_history_on_sentence_partial() -> None:
    drop_session_history("duplex-partial")
    history = get_or_create_session_history("duplex-partial")
    history.begin_user_turn("look")
    prompt = {
        "additional_information": {
            "aura_duplex": True,
            "aura_tts_partial": True,
            "session_id": "duplex-partial",
            "tts_task_type": "CustomVoice",
            "tts_speaker": "Vivian",
        }
    }
    assert len(aura2tts([_source_output("你好，这是一句测试。")], prompt=[prompt])) == 1
    assert all(message.get("role") != "assistant" for message in history.messages)
    drop_session_history("duplex-partial")


def test_aura2tts_close_only_does_not_repeat_the_sentence() -> None:
    prompt = {
        "additional_information": {
            "aura_tts_partial": True,
            "aura_tts_close_only": True,
            "tts_task_type": "CustomVoice",
            "tts_speaker": "Vivian",
        }
    }
    [request] = aura2tts([_source_output("上一句不該再送。")], prompt=[prompt])
    info = request["additional_information"]
    assert info["text"] == [""]
    assert info["max_new_tokens"] == [1]
    assert request["prompt_token_ids"] == [0]


def test_duplex_interception_commits_silent_history_without_aura2tts() -> None:
    """Silent Stage1 uses DIRECT_RESPONSE; history must commit on the data-plane path."""
    from vllm.outputs import CompletionOutput

    from vllm_omni.engine.duplex.contracts import (
        DuplexFence,
        DuplexOutputAction,
        DuplexOutputDecision,
        duplex_ephemeral_stage_request_id,
    )
    from vllm_omni.model_executor.models.aura_omni.duplex.data_plane import AuraDataPlaneSession
    from vllm_omni.outputs import OmniRequestOutput
    from vllm_omni.outputs.duplex import attach_duplex_output_decision

    drop_session_history("duplex-silent-dp")
    history = get_or_create_session_history("duplex-silent-dp")
    history.begin_user_turn("look")
    fence = DuplexFence("duplex-silent-dp", epoch=0, turn_id=1)
    request_id = duplex_ephemeral_stage_request_id(fence, stage_id=1)
    output = OmniRequestOutput(
        request_id=request_id,
        finished=True,
        stage_id=1,
        outputs=[
            CompletionOutput(
                index=0,
                text=SILENT_TEXT,
                token_ids=[151669],
                cumulative_logprob=None,
                logprobs=None,
            )
        ],
    )
    attach_duplex_output_decision(
        output,
        DuplexOutputDecision(
            action=DuplexOutputAction.DIRECT_RESPONSE,
            metadata={"model_listen": True, "duplex_direct_response": True},
        ),
    )
    plane = AuraDataPlaneSession(encode_audio=lambda *_a, **_k: None)
    events = list(plane.project_output(output))
    assert events and events[0].get("silent") is True
    assert events[0].get("model_context_text") == SILENT_TEXT
    assert events[0].get("abort_data_plane_request") is True
    assert history.pending_user == "look"
    from vllm_omni.model_executor.models.aura_omni.duplex.plugin import AuraDuplexPlugin

    AuraDuplexPlugin(encode_audio=lambda *_a, **_k: None).commit_model_context(
        session_id="duplex-silent-dp",
        assistant_text=str(events[0]["model_context_text"]),
    )
    assert history.pending_user is None
    assert history.messages[-1]["content"] == SILENT_TEXT
    drop_session_history("duplex-silent-dp")


def test_proactive_silent_ticks_do_not_evict_a_real_turn() -> None:
    drop_session_history("duplex-silent-ticks")
    history = get_or_create_session_history("duplex-silent-ticks")
    history.begin_user_turn("真正的問題")
    history.commit_turn("真正的回答")
    for _ in range(20):
        history.begin_user_turn("")
        history.commit_turn("<|silent|>")
    assert history.messages[0]["content"] == "真正的問題"
    assert history.messages[1]["content"] == "真正的回答"
    assert len(history.messages) == 2
    drop_session_history("duplex-silent-ticks")
