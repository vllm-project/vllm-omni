# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Logic-equivalent checks adapted from AURA_026_p0exp stage processor tests."""

from __future__ import annotations

from types import SimpleNamespace

from vllm_omni.model_executor.models.aura_omni.duplex.history import (
    drop_session_history,
    get_or_create_session_history,
)
from vllm_omni.model_executor.stage_input_processors.aura_omni import (
    SILENT_TEXT,
    asr2aura,
    aura2tts,
)


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


def test_aura2tts_drops_silent_response() -> None:
    prompt = {"additional_information": {"tts_task_type": "CustomVoice"}}
    assert aura2tts([_source_output(SILENT_TEXT)], prompt=[prompt]) == []


def test_aura2tts_duplex_commits_silent_into_history() -> None:
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
    assert history.messages[-1]["content"] == SILENT_TEXT
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
    assert history.pending_user is None
    assert history.messages[-1]["content"] == SILENT_TEXT
    drop_session_history("duplex-silent-dp")
