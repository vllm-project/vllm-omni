# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio

import pytest
import torch
from comfyui_vllm_omni.lingbot_geometry import LingBotGeometry
from comfyui_vllm_omni.lingbot_moves import (
    CameraMove,
    LingBotSequence,
    Shot,
    compile_sequence,
    parse_moves,
)
from comfyui_vllm_omni.node_lingbot_scene import (
    VLLMOmniLingBotMoves,
    VLLMOmniLingBotRender,
    VLLMOmniLingBotShot,
    VLLMOmniLingBotVisualizer,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
GEOMETRY = LingBotGeometry()


def _three_shot_sequence() -> LingBotSequence:
    return LingBotSequence(
        (
            Shot("A", 45, parse_moves("move forward 0 end\nlook_h right 21 end", shot_frames=45)),
            Shot("B", 84, parse_moves("move forward 0 end\nstrafe left 0 36", shot_frames=84)),
            Shot("C", 60, parse_moves("look_v up 24 end", shot_frames=60)),
        )
    )


def test_shot_node_chains_and_assigns_whole_chunks():
    moves = VLLMOmniLingBotMoves().build("move forward 0 end")[0]
    first = VLLMOmniLingBotShot().add("A", 4, moves=moves)[0]
    second = VLLMOmniLingBotShot().add("B", 7, sequence=first)[0]
    assert [shot.frames for shot in second.shots] == [45, 84]
    assert second.shots[0].moves == (CameraMove("move", "forward", 0, 45),)
    assert second.shots[1].moves == ()


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


def test_moves_node_rejects_an_empty_text():
    with pytest.raises(ValueError, match="at least one move"):
        VLLMOmniLingBotMoves().build("  \n ")


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
    assert payload["extra_params"]["camera_action_script"] == compile_sequence(sequence, GEOMETRY).camera_script()
    assert captured["switches"] == {4: "B", 11: "C"}


def test_visualizer_node_returns_draw_data_only():
    sequence = _three_shot_sequence()
    output = VLLMOmniLingBotVisualizer().show(sequence, 16)
    assert output["result"] == ()
    payload = output["ui"]["omni_lingbot_sequence"][0]
    assert payload["total_chunks"] == 16
    assert len(payload["shots"]) == 3


def test_shots_workflow_matches_the_node_contracts():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[4]
    workflow = json.loads(
        (root / "apps/ComfyUI-vLLM-Omni/example_workflows/vLLM-Omni LingBot World Shots.json").read_text()
    )
    nodes = {node["id"]: node for node in workflow["nodes"]}
    classes = {
        "VLLMOmniLingBotMoves": VLLMOmniLingBotMoves,
        "VLLMOmniLingBotShot": VLLMOmniLingBotShot,
        "VLLMOmniLingBotRender": VLLMOmniLingBotRender,
        "VLLMOmniLingBotVisualizer": VLLMOmniLingBotVisualizer,
    }
    linked_inputs = {
        "VLLMOmniLingBotMoves": set(),  # its moves text is a widget, not a socket
        "VLLMOmniLingBotShot": {"sequence", "moves"},
        "VLLMOmniLingBotRender": {"image", "sequence"},
        "VLLMOmniLingBotVisualizer": {"sequence"},
    }
    for node in workflow["nodes"]:
        node_class = classes.get(node["type"])
        if node_class is None:
            continue
        required = node_class.INPUT_TYPES()["required"]
        widgets = [name for name in required if name not in linked_inputs[node["type"]]]
        assert len(node["widgets_values"]) == len(widgets), node["type"]

    for link_id, from_node, from_slot, to_node, to_slot, link_type in workflow["links"]:
        source = nodes[from_node]
        target = nodes[to_node]
        assert source["outputs"][from_slot]["type"] == link_type
        assert target["inputs"][to_slot]["type"] == link_type
        assert link_id in source["outputs"][from_slot]["links"]
