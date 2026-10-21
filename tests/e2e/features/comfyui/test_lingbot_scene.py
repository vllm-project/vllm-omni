# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
import json
from pathlib import Path

import pytest
import torch
from comfyui_vllm_omni.lingbot_geometry import LingBotGeometry
from comfyui_vllm_omni.lingbot_moves import CameraMove, LingBotSequence, Shot
from comfyui_vllm_omni.node_lingbot_scene import (
    VLLMOmniLingBotRender,
    VLLMOmniLingBotShot,
    VLLMOmniLingBotVisualizer,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
GEOMETRY = LingBotGeometry()
ROOT = Path(__file__).resolve().parents[4]
APP = ROOT / "apps/ComfyUI-vLLM-Omni"


def _three_shot_sequence() -> LingBotSequence:
    return LingBotSequence(
        (
            Shot("A", 45, (CameraMove("move", "forward", 0, 45), CameraMove("look_h", "right", 21, 24))),
            Shot("B", 84, (CameraMove("move", "forward", 0, 84), CameraMove("strafe", "left", 0, 36))),
            Shot("C", 60, (CameraMove("look_v", "up", 24, 36),)),
        )
    )


def test_shot_node_chains_and_assigns_whole_chunks():
    first = VLLMOmniLingBotShot().add("A", 4, {"moves": []})[0]
    second = VLLMOmniLingBotShot().add("B", 7, sequence=first)[0]
    assert [shot.frames for shot in second.shots] == [45, 84]


def test_shot_node_reads_the_timeline_editor():
    timeline = {
        "moves": [
            {"lane": "move", "value": "forward", "start_frame": 0, "frames": 45},
            {"lane": "look_h", "value": "right", "start_frame": 21, "frames": 24},
        ]
    }
    shot = VLLMOmniLingBotShot().add("A", 4, timeline)[0]
    assert shot.shots[0].moves == (
        CameraMove("move", "forward", 0, 45),
        CameraMove("look_h", "right", 21, 24),
    )


def test_shot_node_rejects_a_malformed_timeline_move():
    with pytest.raises(ValueError, match="needs lane, value, start_frame and frames"):
        VLLMOmniLingBotShot().add("A", 4, {"moves": [{"lane": "move"}]})


def test_shot_node_rejects_an_empty_prompt():
    with pytest.raises(ValueError, match="needs a prompt"):
        VLLMOmniLingBotShot().add("   ", 4)


def test_render_node_builds_the_session_payload(monkeypatch):
    from comfyui_vllm_omni import node_lingbot_scene

    captured = {}

    class FakeClient:
        def __init__(self, url):
            captured["url"] = url

        async def generate_video_stream(self, payload, prompt_switches=None):
            captured["payload"] = payload
            captured["switches"] = prompt_switches
            return "VIDEO"

    monkeypatch.setattr(node_lingbot_scene, "VLLMOmniClient", FakeClient)
    sequence = _three_shot_sequence()
    video = asyncio.run(
        VLLMOmniLingBotRender().generate(
            image=torch.ones(1, 32, 32, 3),
            sequence=sequence,
            server_url="ws://localhost:8000/v1/realtime/video",
            model="robbyant/lingbot-world-v2-14b-causal-fast-diffusers",
            width=832,
            height=480,
            fps=16,
            seed=7,
        )
    )
    assert video == ("VIDEO",)
    payload = captured["payload"]
    assert payload["num_frames"] == 189
    assert payload["prompt"] == "A"
    assert payload["extra_params"]["camera_action_script"] != []
    assert captured["switches"] == {4: "B", 11: "C"}


def test_visualizer_checks_the_sequence_and_draws_nothing_itself():
    assert VLLMOmniLingBotVisualizer().show(_three_shot_sequence(), 16) == {"result": ()}
    with pytest.raises(ValueError):
        VLLMOmniLingBotVisualizer().show(_three_shot_sequence(), 0)


def test_shots_workflow_matches_the_node_contracts():
    workflow = json.loads((APP / "example_workflows/vLLM-Omni LingBot World Shots.json").read_text())
    nodes = {node["id"]: node for node in workflow["nodes"]}
    classes = {
        "VLLMOmniLingBotShot": VLLMOmniLingBotShot,
        "VLLMOmniLingBotRender": VLLMOmniLingBotRender,
        "VLLMOmniLingBotVisualizer": VLLMOmniLingBotVisualizer,
    }
    linked = {
        "VLLMOmniLingBotShot": {"sequence"},
        "VLLMOmniLingBotRender": {"image", "sequence"},
        "VLLMOmniLingBotVisualizer": {"sequence"},
    }
    for node in workflow["nodes"]:
        node_class = classes.get(node["type"])
        if node_class is None:
            continue
        widgets = [name for name in node_class.INPUT_TYPES()["required"] if name not in linked[node["type"]]]
        assert len(node["widgets_values"]) == len(widgets), node["type"]

    for link_id, from_node, from_slot, to_node, to_slot, link_type in workflow["links"]:
        source, target = nodes[from_node], nodes[to_node]
        assert source["outputs"][from_slot]["type"] == link_type
        assert target["inputs"][to_slot]["type"] == link_type
        assert link_id in source["outputs"][from_slot]["links"]
