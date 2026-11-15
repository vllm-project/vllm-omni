# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from vllm_omni.engine.duplex.realtime_events import (
    RealtimeProjectionState,
    project_internal_event,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_input_transcribed_is_a_user_transcript_event() -> None:
    events = project_internal_event(
        RealtimeProjectionState(session_id="s"),
        {"type": "input.transcribed", "transcript": "  你好  "},
    )
    assert len(events) == 1
    assert events[0].wire_type == "conversation.item.input_audio_transcription.completed"
    assert events[0].transcript == "你好"


def test_input_transcribed_updates_the_committed_item() -> None:
    state = RealtimeProjectionState(session_id="s")
    project_internal_event(
        state,
        {
            "type": "input.committed",
            "realtime_item_id": "item_user_1",
            "is_speech": True,
            "message": {"role": "user"},
        },
    )
    events = project_internal_event(
        state,
        {"type": "input.transcribed", "transcript": "你好", "realtime_item_id": "item_user_1"},
    )
    assert events[0].item_id == "item_user_1"
    item = state.conversation_items["item_user_1"]
    assert item["content"][0]["transcript"] == "你好"


def test_blank_input_transcribed_emits_nothing() -> None:
    events = project_internal_event(
        RealtimeProjectionState(session_id="s"),
        {"type": "input.transcribed", "transcript": "  "},
    )
    assert events == []
