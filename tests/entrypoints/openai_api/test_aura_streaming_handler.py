# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for AuraStreamingVideoHandler."""

from __future__ import annotations

import base64
import io
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from vllm_omni.model_executor.stage_input_processors.aura_cross_turn_penalty import (
    merge_penalty_sampling_params,
)
from vllm_omni.model_executor.stage_input_processors.aura_session_history import (
    AuraSessionState,
    SessionHistory,
    clear_all_sessions,
    get_session_history,
    register_session,
)
from vllm_omni.entrypoints.openai.serving_video_stream import (
    AuraStreamingVideoHandler,
    AuraStreamingVideoSessionConfig,
)
from vllm_omni.entrypoints.openai.video_stream_base import VideoStreamTurnTrigger

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def _clear_aura_session_store():
    clear_all_sessions()
    yield
    clear_all_sessions()


def _make_jpeg(r: int = 128, g: int = 128, b: int = 128) -> bytes:
    img = Image.new("RGB", (16, 16), (r, g, b))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _session_state() -> AuraSessionState:
    history = SessionHistory()
    session_id = "aura-test-session"
    register_session(session_id, history)
    return AuraSessionState(history=history, turn_frame_arrays=[], session_id=session_id)


def test_aura_disables_manual_query_and_interrupt():
    handler = AuraStreamingVideoHandler(chat_service=object())
    assert handler.supports_manual_query_turn() is False
    assert handler.supports_query_interrupt() is False
    assert handler.releases_turn_after_text_done() is True


def test_aura_streaming_session_config_native_aligned_defaults():
    config = AuraStreamingVideoSessionConfig(model="test")
    assert config.cross_turn_penalty == 1.0
    assert config.cross_turn_lookback == 10


def test_should_trigger_turn_respects_auto_trigger_gate():
    handler = AuraStreamingVideoHandler(chat_service=object())
    config = AuraStreamingVideoSessionConfig(model="test", auto_trigger=True, auto_trigger_min_frames=2)

    assert (
        handler.should_trigger_turn(
            VideoStreamTurnTrigger(frame_count=1, is_generating=False, is_turn_locked=False, config=config)
        )
        is False
    )
    assert (
        handler.should_trigger_turn(
            VideoStreamTurnTrigger(frame_count=2, is_generating=False, is_turn_locked=False, config=config)
        )
        is True
    )
    assert (
        handler.should_trigger_turn(
            VideoStreamTurnTrigger(frame_count=3, is_generating=True, is_turn_locked=True, config=config)
        )
        is False
    )
    assert (
        handler.should_trigger_turn(
            VideoStreamTurnTrigger(frame_count=3, is_generating=True, is_turn_locked=False, config=config)
        )
        is True
    )

    disabled = AuraStreamingVideoSessionConfig(model="test", auto_trigger=False)
    assert (
        handler.should_trigger_turn(
            VideoStreamTurnTrigger(frame_count=5, is_generating=False, is_turn_locked=False, config=disabled)
        )
        is False
    )


def test_auto_trigger_frame_count_uses_turn_frame_arrays():
    handler = AuraStreamingVideoHandler(chat_service=object())
    state = _session_state()
    session_buffer = [_b64(_make_jpeg()) for _ in range(5)]
    state.turn_frame_arrays = [
        np.zeros((8, 8, 3), dtype=np.uint8),
        np.zeros((8, 8, 3), dtype=np.uint8),
    ]
    assert handler.auto_trigger_frame_count(session_buffer, state) == 2


def test_per_turn_auto_trigger_not_cumulative_session_buffer():
    """13 frames @ min=2 should yield 6 turns, not 12 from uncleared frame_buffer."""
    handler = AuraStreamingVideoHandler(chat_service=object())
    config = AuraStreamingVideoSessionConfig(model="test", auto_trigger=True, auto_trigger_min_frames=2)
    state = _session_state()
    session_buffer: list[str] = []
    triggers = 0

    for i in range(13):
        raw = _make_jpeg(i, i, i)
        b64 = _b64(raw)
        session_buffer.append(b64)
        handler.on_frame_buffered(raw, b64, state, config)
        if handler.should_trigger_turn(
            VideoStreamTurnTrigger(
                frame_count=handler.auto_trigger_frame_count(session_buffer, state),
                is_generating=False,
                is_turn_locked=False,
                config=config,
            )
        ):
            triggers += 1
            state.turn_frame_arrays.clear()

    assert triggers == 6
    assert len(session_buffer) == 13


def test_on_turn_complete_persists_user_video_and_assistant():
    from vllm_omni.model_executor.stage_input_processors.aura_omni import (
        record_turn_transcript,
    )

    handler = AuraStreamingVideoHandler(chat_service=object())
    state = _session_state()
    frames = np.array(
        [
            [[[1, 0, 0], [0, 1, 0]], [[0, 0, 1], [1, 1, 0]]],
            [[[2, 0, 0], [0, 2, 0]], [[0, 0, 2], [2, 2, 0]]],
        ],
        dtype=np.uint8,
    )
    metadata = {
        "fps": 2.0,
        "duration": 1.0,
        "total_num_frames": 2,
        "frames_indices": [0, 1],
        "video_backend": "opencv",
        "do_sample_frames": False,
    }
    state.pending_turn_video = {"video": [(frames, metadata)]}
    record_turn_transcript("req-1", "画面有什么？")

    handler.on_turn_complete(state, {"role": "user", "content": []}, "好的。", request_id="req-1")

    inputs = state.history.get_vllm_inputs()
    assert "画面有什么？" in inputs["prompt"]
    assert "好的。" in inputs["prompt"]
    assert "<|video_pad|>" in inputs["prompt"]
    assert len(inputs["multi_modal_data"]["video"]) == 1
    assert state.pending_turn_video is None


def test_build_engine_prompt_stores_audio_and_session_payload():
    handler = AuraStreamingVideoHandler(chat_service=object())
    config = AuraStreamingVideoSessionConfig(model="test", aura_system_prompt="system-a")
    state = _session_state()
    state.turn_frame_arrays = [
        np.zeros((8, 8, 3), dtype=np.uint8),
        np.ones((8, 8, 3), dtype=np.uint8),
    ]

    messages, user_message = handler.build_engine_prompt(
        config,
        [_b64(_make_jpeg())],
        bytearray(b"\x00\x01"),
        state,
        "",
        {},
    )

    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    content_types = [part["type"] for part in messages[0]["content"]]
    assert content_types == ["input_audio"]

    additional = user_message["_aura_additional_information"]
    assert additional["aura_session_id"] == state.session_id
    assert "aura_session_state" not in additional
    assert additional["aura_system_prompt"] == ["system-a"]
    deferred = additional["deferred_multi_modal_data"]
    assert deferred["video"][0][1]["total_num_frames"] == 2
    assert deferred["video"][0][0].shape == (2, 8, 8, 3)
    assert additional["tts_ref_audio"]
    assert additional["tts_ref_text"]


def test_build_engine_prompt_omni_skip_stages():
    handler = AuraStreamingVideoHandler(chat_service=object())
    state = _session_state()
    state.turn_frame_arrays = [np.zeros((8, 8, 3), dtype=np.uint8)]

    _, text_only = handler.build_engine_prompt(
        AuraStreamingVideoSessionConfig(model="test", modalities=["text"]),
        [],
        bytearray(),
        state,
        "",
        {},
    )
    assert text_only["_aura_additional_information"]["omni_skip_stages"] == [0]

    _, with_audio = handler.build_engine_prompt(
        AuraStreamingVideoSessionConfig(model="test"),
        [_b64(_make_jpeg())],
        bytearray(b"\x00\x01"),
        state,
        "",
        {},
    )
    assert with_audio["_aura_additional_information"]["omni_skip_stages"] == []
    assert "tts_ref_audio" not in text_only["_aura_additional_information"]


def test_create_message_history_registers_server_side_store():
    handler = AuraStreamingVideoHandler(chat_service=object())
    config = AuraStreamingVideoSessionConfig(model="test")

    state = handler.create_message_history(config)

    assert state.session_id
    assert get_session_history(state.session_id) is state.history


def test_on_session_end_unregisters_server_side_store():
    handler = AuraStreamingVideoHandler(chat_service=object())
    state = _session_state()

    handler.on_session_end(state)

    assert get_session_history(state.session_id) is None


@pytest.mark.asyncio
async def test_process_query_merges_cross_turn_penalty_sampling_params():
    captured_requests: list[Any] = []

    class CapturingChatService:
        chat_template = None
        chat_template_content_format = "string"

        class _Renderer:
            pass

        renderer = _Renderer()

        async def _preprocess_chat(self, request, messages, **kwargs):
            captured_requests.append(request)
            return messages, [{"prompt": "engine-prompt"}]

    class _FakeTokenizer:
        all_special_ids = [0, 1]

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            return [ord(c) for c in text if c.isalnum() or ord(c) > 127]

        def decode(self, token_ids: list[int]) -> str:
            return "".join(chr(tid) for tid in token_ids)

    from vllm_omni.model_executor.stage_input_processors.aura_cross_turn_penalty import CrossTurnPenalty

    mock_engine = MagicMock()

    async def _get_tokenizer():
        return _FakeTokenizer()

    mock_engine.get_tokenizer = _get_tokenizer

    handler = AuraStreamingVideoHandler(chat_service=CapturingChatService(), engine_client=mock_engine)
    config = AuraStreamingVideoSessionConfig(model="test", cross_turn_penalty=2.0)
    state = _session_state()
    state.turn_frame_arrays = [
        np.zeros((8, 8, 3), dtype=np.uint8),
        np.zeros((8, 8, 3), dtype=np.uint8),
    ]
    penalty = CrossTurnPenalty(_FakeTokenizer(), window=2, logit_penalty=2.0)
    penalty.record("hello world")
    penalty.record("hello again")
    state.cross_turn_penalty = penalty

    async def _noop_generation(*_args, **_kwargs):
        return None

    handler._run_engine_generation = _noop_generation  # type: ignore[method-assign]

    await handler._process_query_engine(
        websocket=MagicMock(),
        config=config,
        frame_buffer=[_b64(_make_jpeg())],
        audio_buffer=bytearray(),
        message_history=state,
        query_text="",
        request_id="req-aura-penalty",
        interrupt_event=MagicMock(),
        prewarmed_frames={},
    )

    assert captured_requests
    sampling = getattr(captured_requests[0], "sampling_params_list", None)
    assert sampling is not None
    assert len(sampling) >= 2
    assert sampling[1].get("logit_bias") or sampling[1].get("bad_words")

    merged = merge_penalty_sampling_params(
        [{"temperature": 0.7}, {"top_p": 0.9}],
        {"logit_bias": {42: -1.5}, "bad_words": ["foo"]},
    )
    assert merged[0] == {"temperature": 0.7}
    assert merged[1]["logit_bias"] == {42: -1.5}
    assert merged[1]["bad_words"] == ["foo"]
