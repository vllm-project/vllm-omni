# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
import base64
import json
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock

import aiohttp
import pytest
import torch
from comfyui_vllm_omni.node_lingbot import MODEL, VLLMOmniLingBotWorld
from comfyui_vllm_omni.utils import api_client
from comfyui_vllm_omni.utils.format import image_tensor_to_base64
from PIL import Image

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
ROOT = Path(__file__).resolve().parents[4]
APP = ROOT / "apps/ComfyUI-vLLM-Omni"


def _config(**changes):
    args = dict(
        image=torch.ones(1, 32, 32, 3),
        server_url="ws://localhost:8000/v1/realtime/video",
        model=MODEL,
        prompt="forest",
        width=832,
        height=480,
        num_frames=33,
        fps=16,
        seed=42,
    )
    args.update(changes)
    return VLLMOmniLingBotWorld().prepare(**args)["ui"]["omni_lingbot"][0]


def test_prepare_resizes_first_frame_and_builds_the_start_message():
    config = _config()
    payload = config["payload"]
    assert config["keyboard"]
    assert "camera_action_script" not in payload["extra_params"]
    image = Image.open(BytesIO(base64.b64decode(payload["image_reference"]["image_url"].split(",")[1])))
    assert image.size == (832, 480)
    assert image.getpixel((0, 0)) == (255, 255, 255)
    assert payload["num_inference_steps"] == 4
    assert payload["seed"] == 42
    # Existing image callers keep their original geometry without the new option.
    original = image_tensor_to_base64(torch.ones(1, 32, 32, 3))
    assert Image.open(BytesIO(base64.b64decode(original.split(",")[1]))).size == (32, 32)


def test_direct_connection_contract():
    local = "ws://127.0.0.1:8000/v1/realtime/video"
    assert _config(server_url=f" {local} ")["url"] == local
    assert _config(server_url="wss://model.example/v1/realtime/video")["url"] == "wss://model.example/v1/realtime/video"


@pytest.mark.parametrize(
    "changes",
    [
        {"num_frames": 32},
        {"width": 833},
        {"prompt": " "},
        {"server_url": "https://example.com"},
        {"server_url": "ws://user:secret@localhost"},
        {"image": torch.ones(2, 32, 32, 3)},
        {"image": torch.full((1, 32, 32, 3), float("nan"))},
    ],
)
def test_prepare_rejects_invalid_inputs(changes):
    with pytest.raises(ValueError):
        _config(**changes)


def test_workflow_uses_vllm_omni_prefix_and_widget_order():
    workflow = json.loads((APP / "example_workflows/vLLM-Omni LingBot World.json").read_text())
    load, node = workflow["nodes"]
    assert node["type"] == "VLLMOmniLingBotWorld"
    assert workflow["links"] == [[1, load["id"], 0, node["id"], 0, "IMAGE"]]
    names = [key for key in VLLMOmniLingBotWorld.INPUT_TYPES()["required"] if key != "image"]
    assert len(names) == len(node["widgets_values"])
    config = _config(**dict(zip(names, node["widgets_values"])))
    assert config["payload"]["num_frames"] == 129
    assert config["url"] == "ws://127.0.0.1:8000/v1/realtime/video"


@pytest.fixture
def video_connection(monkeypatch):
    """Mock transport and reuse the existing decoder seam, as in video reference tests."""
    socket = AsyncMock()
    session = MagicMock()
    session.ws_connect.return_value.__aenter__.return_value = socket
    client = MagicMock()
    client.return_value.__aenter__.return_value = session
    monkeypatch.setattr(api_client.aiohttp, "ClientSession", client)
    decode = Mock(return_value=b"decoded-video")
    monkeypatch.setattr(api_client, "bytes_to_video", decode)
    return socket, session.ws_connect.return_value, decode


async def test_stream_client_collects_all_fragments_before_decoding(video_connection):
    socket, _, decode = video_connection
    socket.receive.side_effect = [
        aiohttp.WSMessage(aiohttp.WSMsgType.TEXT, '{"type":"video.start"}', ""),
        aiohttp.WSMessage(aiohttp.WSMsgType.BINARY, b"init-fragment", ""),
        aiohttp.WSMessage(aiohttp.WSMsgType.BINARY, b"media-fragment", ""),
        aiohttp.WSMessage(aiohttp.WSMsgType.TEXT, '{"type":"session.done","stopped":false}', ""),
    ]
    result = await api_client.VLLMOmniClient("ws://omni.example/v1/realtime/video").generate_video_stream(
        {"type": "session.start"}
    )
    decode.assert_called_once_with(b"init-fragmentmedia-fragment")
    assert result == b"decoded-video"


async def test_stream_client_sends_each_prompt_one_chunk_before_its_boundary(video_connection):
    socket, _, _ = video_connection
    socket.receive.side_effect = [
        aiohttp.WSMessage(aiohttp.WSMsgType.TEXT, '{"type":"video.chunk_metadata","generation_chunk_index":3}', ""),
        aiohttp.WSMessage(aiohttp.WSMsgType.BINARY, b"fragment", ""),
        aiohttp.WSMessage(aiohttp.WSMsgType.TEXT, '{"type":"session.done","stopped":false}', ""),
    ]
    await api_client.VLLMOmniClient("ws://omni.example/v1/realtime/video").generate_video_stream(
        {"type": "session.start"}, {4: "second shot"}
    )
    interaction = socket.send_json.call_args_list[1].args[0]
    assert interaction == {
        "type": "session.interaction",
        "interaction": {"event_id": "shot-4", "event": {"prompt": "second shot"}},
    }


@pytest.mark.parametrize(
    "event,partial,match",
    [
        (None, False, "before session.done"),
        (None, True, "before session.done"),
        ({"type": "error", "message": "bad camera"}, False, "bad camera"),
        ({"type": "session.done", "stopped": True}, True, "stopped"),
        ({"type": "session.done", "stopped": False}, False, "without video data"),
    ],
)
async def test_stream_client_rejects_incomplete_or_failed_sessions(video_connection, event, partial, match):
    socket, _, decode = video_connection
    message = (
        aiohttp.WSMessage(aiohttp.WSMsgType.TEXT, json.dumps(event), "")
        if event is not None
        else aiohttp.WSMessage(aiohttp.WSMsgType.CLOSE, None, "")
    )
    socket.receive.side_effect = ([aiohttp.WSMessage(aiohttp.WSMsgType.BINARY, b"partial", "")] if partial else []) + [
        message
    ]
    with pytest.raises(RuntimeError, match=match):
        await api_client.VLLMOmniClient("ws://omni.example/v1/realtime/video").generate_video_stream(
            {"type": "session.start"}
        )
    decode.assert_not_called()


async def test_stream_keepalive_and_cancellation_exit_connection(video_connection):
    socket, connection, decode = video_connection
    socket.receive.side_effect = [asyncio.TimeoutError(), asyncio.CancelledError()]
    with pytest.raises(asyncio.CancelledError):
        await api_client.VLLMOmniClient("ws://omni.example/v1/realtime/video").generate_video_stream(
            {"type": "session.start"}
        )
    assert [call.args[0]["type"] for call in socket.send_json.call_args_list] == ["session.start", "session.ping"]
    connection.__aexit__.assert_awaited_once()
    decode.assert_not_called()
