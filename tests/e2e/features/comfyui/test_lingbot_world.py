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
from comfyui_vllm_omni.node_lingbot import MODEL, MODES, VLLMOmniLingBotOfflineJSON, VLLMOmniLingBotWorld
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
        mode="stationary",
        width=832,
        height=480,
        num_frames=33,
        fps=16,
        seed=42,
    )
    args.update(changes)
    return VLLMOmniLingBotWorld().prepare(**args)["ui"]["omni_lingbot"][0]


def test_prepare_resizes_first_frame_and_builds_script():
    config = _config()
    payload = config["payload"]
    assert payload["extra_params"]["camera_action_script"] == [[[], [], []]] * 3
    image = Image.open(BytesIO(base64.b64decode(payload["image_reference"]["image_url"].split(",")[1])))
    assert image.size == (832, 480)
    assert image.getpixel((0, 0)) == (255, 255, 255)
    assert payload["num_inference_steps"] == 4
    assert payload["seed"] == 42
    # Existing image callers keep their original geometry without the new option.
    original = image_tensor_to_base64(torch.ones(1, 32, 32, 3))
    assert Image.open(BytesIO(base64.b64decode(original.split(",")[1]))).size == (32, 32)


def test_keyboard_and_direct_connection_contract():
    config = _config(mode=MODES[0])
    assert config["keyboard"]
    assert "camera_action_script" not in config["payload"]["extra_params"]
    assert config["url"] == "ws://localhost:8000/v1/realtime/video"
    local = "ws://127.0.0.1:8000/v1/realtime/video"
    assert _config(server_url=f" {local} ")["url"] == local
    remote = "wss://model.example/v1/realtime/video"
    assert _config(server_url=remote)["url"] == remote
    assert _config(mode="forward")["payload"]["extra_params"]["camera_action_script"] == [[["w"], ["w"], ["w"]]] * 3


@pytest.mark.parametrize(
    "changes",
    [
        {"num_frames": 32},
        {"mode": "unknown"},
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


@pytest.mark.parametrize("mode", ["Realtime", "Offline"])
def test_json_workflow_schema_and_camera_script(mode):
    from comfyui_vllm_omni import node_lingbot

    from vllm_omni.diffusion.models.lingbot_world.actions import parse_lingbot_camera_action_script

    workflow = json.loads((APP / f"example_workflows/vLLM-Omni LingBot World {mode} JSON.json").read_text())
    node = workflow["nodes"][1]
    cls = getattr(node_lingbot, node["type"])
    names = [key for key in cls.INPUT_TYPES()["required"] if key != "image"]
    assert len(names) == len(node["widgets_values"])
    args = dict(zip(names, node["widgets_values"]))
    config = cls().prepare(image=torch.ones(1, 32, 32, 3), **args)["ui"]["omni_lingbot"][0]
    script = config["payload"]["extra_params"]["camera_action_script"]
    assert script == json.loads(args["camera_json"])
    assert not config["keyboard"]
    assert len(parse_lingbot_camera_action_script(script, frames_per_chunk=3)) == 3
    assert config["payload"]["num_frames"] == 33
    assert tuple(output["type"] for output in node["outputs"]) == cls.RETURN_TYPES
    if mode == "Offline":
        assert workflow["links"][-1] == [2, 2, 0, 3, 0, "VIDEO"]
        assert workflow["nodes"][-1]["type"] == "SaveVideo"
        assert not cls.OUTPUT_NODE


@pytest.mark.parametrize(
    "script",
    [
        "{",
        "{}",
        "[]",
        "[[[],[],[]]]",
        "[[[],[],[]]]" * 10000,
        "[[[],[],[]],[[],[],[]],[[],[]]]",
        '[[[],[],[]],[[],[],[]],[[],[],["x"]]]',
        '[[[],[],[]],[[],[],[]],[[],[],"w"]]',
    ],
)
def test_json_camera_rejects_invalid_scripts(script):
    with pytest.raises(ValueError):
        _config(camera_json=script)


def test_json_camera_accepts_case_insensitive_and_simultaneous_keys():
    script = [[["W", "j"], [], ["d"]], [[], [], []], [[], [], []]]
    assert _config(camera_json=json.dumps(script))["payload"]["extra_params"]["camera_action_script"] == script


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


async def test_offline_json_collects_all_fragments_before_decoding(video_connection):
    socket, _, decode = video_connection
    socket.receive.side_effect = [
        aiohttp.WSMessage(aiohttp.WSMsgType.TEXT, '{"type":"video.start"}', ""),
        aiohttp.WSMessage(aiohttp.WSMsgType.BINARY, b"init-fragment", ""),
        aiohttp.WSMessage(aiohttp.WSMsgType.BINARY, b"media-fragment", ""),
        aiohttp.WSMessage(aiohttp.WSMsgType.TEXT, '{"type":"session.done","stopped":false}', ""),
    ]
    script = [[["w"], [], []]]
    (result,) = await VLLMOmniLingBotOfflineJSON().generate(
        image=torch.ones(1, 16, 16, 3),
        server_url="ws://omni.example/v1/realtime/video",
        model=MODEL,
        prompt="forest",
        width=16,
        height=16,
        num_frames=9,
        fps=16,
        seed=42,
        camera_json=json.dumps(script),
    )
    payload = socket.send_json.call_args.args[0]
    assert payload["extra_params"]["camera_action_script"] == script
    assert payload["num_frames"] == 9
    decode.assert_called_once_with(b"init-fragmentmedia-fragment")
    assert result == b"decoded-video"


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
async def test_offline_json_rejects_incomplete_or_failed_sessions(video_connection, event, partial, match):
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


async def test_offline_keepalive_and_cancellation_exit_connection(video_connection):
    socket, connection, decode = video_connection
    socket.receive.side_effect = [asyncio.TimeoutError(), asyncio.CancelledError()]
    with pytest.raises(asyncio.CancelledError):
        await api_client.VLLMOmniClient("ws://omni.example/v1/realtime/video").generate_video_stream(
            {"type": "session.start"}
        )
    assert [call.args[0]["type"] for call in socket.send_json.call_args_list] == ["session.start", "session.ping"]
    connection.__aexit__.assert_awaited_once()
    decode.assert_not_called()
