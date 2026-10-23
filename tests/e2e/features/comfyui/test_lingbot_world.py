# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
import base64
import json
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest
import torch
from comfyui_vllm_omni import node_lingbot_scene
from comfyui_vllm_omni.lingbot_moves import CameraMove, LingBotSequence, Shot
from comfyui_vllm_omni.node_lingbot import MODEL, VLLMOmniLingBotWorld
from comfyui_vllm_omni.node_lingbot_scene import VLLMOmniLingBotRender
from comfyui_vllm_omni.utils import api_client
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


def _sequence() -> LingBotSequence:
    return LingBotSequence(
        (
            Shot("A", 45, (CameraMove("move", "forward", 0, 45),)),
            Shot("B", 84, (CameraMove("strafe", "left", 0, 36),)),
            Shot("C", 60, (CameraMove("look_v", "up", 24, 36),)),
        )
    )


def test_prepare_resizes_the_first_frame_and_drops_the_scripted_camera():
    config = _config()
    payload = config["payload"]
    assert config["keyboard"]
    assert "camera_action_script" not in payload["extra_params"]
    image = Image.open(BytesIO(base64.b64decode(payload["image_reference"]["image_url"].split(",")[1])))
    assert image.size == (832, 480)
    assert (payload["num_inference_steps"], payload["seed"]) == (4, 42)
    assert (
        _config(server_url=" ws://127.0.0.1:8000/v1/realtime/video ")["url"] == "ws://127.0.0.1:8000/v1/realtime/video"
    )


@pytest.mark.parametrize(
    "changes",
    [{"num_frames": 32}, {"prompt": " "}, {"server_url": "https://example.com"}],
)
def test_prepare_rejects_inputs_it_cannot_send(changes):
    with pytest.raises(ValueError):
        _config(**changes)


def test_workflow_matches_the_node_contracts():
    workflow = json.loads((APP / "example_workflows/vLLM-Omni LingBot World.json").read_text())
    load, node = workflow["nodes"]
    assert workflow["links"] == [[1, load["id"], 0, node["id"], 0, "IMAGE"]]
    names = [key for key in VLLMOmniLingBotWorld.INPUT_TYPES()["required"] if key != "image"]
    config = _config(**dict(zip(names, node["widgets_values"])))
    assert config["payload"]["num_frames"] == 129


def test_render_turns_a_shot_chain_into_one_session(monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, url):
            captured["url"] = url

        async def generate_video_stream(self, payload, prompt_switches=None):
            captured["payload"], captured["switches"] = payload, prompt_switches
            return "VIDEO"

    monkeypatch.setattr(node_lingbot_scene, "VLLMOmniClient", FakeClient)
    video = asyncio.run(
        VLLMOmniLingBotRender().generate(
            image=torch.ones(1, 32, 32, 3),
            sequence=_sequence(),
            server_url="ws://localhost:8000/v1/realtime/video",
            model=MODEL,
            width=832,
            height=480,
            fps=16,
            seed=7,
        )
    )
    assert video == ("VIDEO",)
    assert captured["payload"]["num_frames"] == 189
    assert captured["payload"]["prompt"] == "A"
    assert captured["payload"]["extra_params"]["camera_action_script"]
    assert captured["switches"] == {4: "B", 11: "C"}


async def test_each_prompt_switch_goes_out_one_chunk_before_its_boundary(monkeypatch):
    socket = AsyncMock()
    session = MagicMock()
    session.ws_connect.return_value.__aenter__.return_value = socket
    client = MagicMock()
    client.return_value.__aenter__.return_value = session
    monkeypatch.setattr(api_client.aiohttp, "ClientSession", client)
    monkeypatch.setattr(api_client, "bytes_to_video", lambda data: b"decoded-video")
    socket.receive.side_effect = [
        aiohttp.WSMessage(aiohttp.WSMsgType.TEXT, '{"type":"video.chunk_metadata","generation_chunk_index":3}', ""),
        aiohttp.WSMessage(aiohttp.WSMsgType.BINARY, b"fragment", ""),
        aiohttp.WSMessage(aiohttp.WSMsgType.TEXT, '{"type":"session.done","stopped":false}', ""),
    ]

    await api_client.VLLMOmniClient("ws://omni.example/v1/realtime/video").generate_video_stream(
        {"type": "session.start"}, {4: "second shot"}
    )

    assert socket.send_json.call_args_list[1].args[0] == {
        "type": "session.interaction",
        "interaction": {"event_id": "shot-4", "event": {"prompt": "second shot"}},
    }
