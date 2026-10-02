# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""History depth selects completed text turns without changing media lifetime."""

import copy
import json

import pytest
from fastapi import WebSocket
from pydantic import ValidationError

from vllm_omni.entrypoints.openai.serving_video_stream import (
    QwenOmniStreamingVideoHandler,
    StreamingVideoSessionConfig,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def history(turns):
    return [
        message
        for i in range(turns)
        for message in (
            {"role": "user", "content": f"question {i}"},
            {"role": "assistant", "content": f"answer {i}"},
        )
    ]


def build(config, retained):
    return QwenOmniStreamingVideoHandler(chat_service=object()).build_engine_prompt(
        config, [], bytearray(), retained, "now", {}
    )[0]


@pytest.mark.parametrize("turns", [0, 1, 2, 3, 7, 100])
@pytest.mark.parametrize("depth", [1, 2, 3, 200, None])
def test_history_window_matches_completed_turns(turns, depth):
    retained = history(turns)
    before = copy.deepcopy(retained)
    actual = build(StreamingVideoSessionConfig(max_history_turns=depth, system_prompt="voice"), retained)
    first = 0 if depth is None else max(0, turns - depth)
    expected = [
        message
        for i in range(first, turns)
        for message in (
            {"role": "user", "content": f"question {i}"},
            {"role": "assistant", "content": f"answer {i}"},
        )
    ]
    assert actual == [
        {"role": "system", "content": "voice"},
        *expected,
        {"role": "user", "content": [{"type": "text", "text": "now"}]},
    ]
    assert retained == before


@pytest.mark.parametrize("turns", [0, 1, 2, 10])
def test_default_is_explicit_one_and_legacy_last_pair(turns):
    retained = history(turns)
    default = build(StreamingVideoSessionConfig(), retained)
    assert default == build(StreamingVideoSessionConfig(max_history_turns=1), retained)
    assert default[:-1] == retained[-2:]


@pytest.mark.parametrize("depth", [1, 2, None])
def test_history_media_removed_but_current_media_preserved(depth):
    retained = history(3)
    for i in (0, 2, 4):
        retained[i]["content"] = [
            {"type": "image_url", "image_url": {"url": "old-image"}},
            {"type": "image_pil", "image_pil": "old-pil"},
            {"type": "video_url", "video_url": {"url": "old-video"}},
            {"type": "input_audio", "input_audio": {"data": "old-audio", "format": "wav"}},
            {"type": "text", "text": f"question {i // 2}"},
        ]
    before = copy.deepcopy(retained)
    handler = QwenOmniStreamingVideoHandler(chat_service=object())
    messages, current = handler.build_engine_prompt(
        StreamingVideoSessionConfig(max_history_turns=depth),
        ["current-image"],
        bytearray(b"\x00\x00"),
        retained,
        "now",
        {},
        frame_indices=[0],
    )
    assert all(isinstance(msg["content"], str) for msg in messages[:-1])
    assert [part["type"] for part in current["content"]] == ["image_url", "input_audio", "text"]
    assert current["content"][0]["image_url"]["url"].endswith("current-image")
    assert retained == before


def test_window_does_not_delete_retained_state_or_cross_sessions():
    handler = QwenOmniStreamingVideoHandler(chat_service=object())
    a: list[dict[str, str]] = []
    b: list[dict[str, str]] = []
    for i in range(4):
        handler.on_turn_complete(a, {"role": "user", "content": f"question {i}"}, f"answer {i}")
        assert build(StreamingVideoSessionConfig(max_history_turns=1), a)[:-1] == history(i + 1)[-2:]
    assert a == history(4)
    assert build(StreamingVideoSessionConfig(max_history_turns=None), b)[:-1] == []
    assert build(StreamingVideoSessionConfig(max_history_turns=None), a)[:-1] == history(4)


@pytest.mark.parametrize("value", [0, -1, 1.5, "all", [], {}])
def test_invalid_depth_rejected(value):
    with pytest.raises(ValidationError):
        StreamingVideoSessionConfig(max_history_turns=value)


@pytest.mark.parametrize("value", [1, 2, None])
@pytest.mark.asyncio
async def test_wire_config_preserves_depth_and_round_trips(value, mocker):
    handler = QwenOmniStreamingVideoHandler(chat_service=object())
    ws = mocker.AsyncMock(spec=WebSocket)
    ws.receive_text.return_value = json.dumps({"type": "session.config", "max_history_turns": value})
    config = await handler._receive_config(ws)
    assert config is not None
    assert config.max_history_turns == value
    assert StreamingVideoSessionConfig.model_validate_json(config.model_dump_json()).max_history_turns == value


@pytest.mark.asyncio
async def test_wire_invalid_depth_returns_error(mocker):
    handler = QwenOmniStreamingVideoHandler(chat_service=object())
    ws = mocker.AsyncMock(spec=WebSocket)
    ws.receive_text.return_value = json.dumps({"type": "session.config", "max_history_turns": 0})
    assert await handler._receive_config(ws) is None
    ws.send_json.assert_awaited_once()
    error = ws.send_json.await_args.args[0]
    assert error["type"] == "error"
    assert "max_history_turns" in error["message"]
