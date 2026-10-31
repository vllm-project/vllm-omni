# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
import re
from graphlib import TopologicalSorter
from pathlib import Path

import pytest
from comfyui_vllm_omni import nodes as omni_nodes

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

WORKFLOW = (
    Path(__file__).resolve().parents[4]
    / "apps/ComfyUI-vLLM-Omni/example_workflows/vLLM-Omni MiniMax-H3 Music Video.json"
)
FPS = 24
SHOT_FRAMES = 124
SHOT_SECONDS = SHOT_FRAMES / FPS
# Trim widgets step by 0.01 s; one frame is ~0.042 s.
TIME_TOLERANCE = 0.005


@pytest.fixture
def workflow():
    return json.loads(WORKFLOW.read_text(encoding="utf-8"))


def _index(workflow):
    nodes = {node["id"]: node for node in workflow["nodes"]}
    links = {link[0]: link for link in workflow["links"]}
    return nodes, links


def _source(nodes, links, node, name):
    link_id = next(input_["link"] for input_ in node["inputs"] if input_["name"] == name)
    return None if link_id is None else nodes[links[link_id][1]]


def _shots(nodes, links):
    """Shot Generate Video nodes in the order Concatenate Video joins them."""
    concat = next(node for node in nodes.values() if node["type"] == "ConcatenateVideo")
    shots = []
    for input_ in concat["inputs"]:
        if input_["name"].startswith("videos.") and input_["link"] is not None:
            shots.append(nodes[links[input_["link"]][1]])
    return concat, shots


def test_music_video_workflow_connections(workflow):
    nodes, links = _index(workflow)
    assert len(nodes) == len(workflow["nodes"])
    assert len(links) == len(workflow["links"])
    assert workflow["last_node_id"] == max(nodes)
    assert workflow["last_link_id"] == max(links)
    graph: dict[int, set[int]] = {node_id: set() for node_id in nodes}
    for link_id, source, output_slot, target, input_slot, kind in links.values():
        output = nodes[source]["outputs"][output_slot]
        input_ = nodes[target]["inputs"][input_slot]
        assert output["type"] == input_["type"] == kind
        assert link_id in output["links"]
        assert input_["link"] == link_id
        graph[target].add(source)
    assert len(tuple(TopologicalSorter(graph).static_order())) == len(nodes)
    for node in nodes.values():
        for slot, input_ in enumerate(node.get("inputs", [])):
            if input_["link"] is not None:
                assert links[input_["link"]][3:5] == [node["id"], slot]
        for slot, output in enumerate(node.get("outputs", [])):
            for link_id in output.get("links") or []:
                assert links[link_id][1:3] == [node["id"], slot]


def test_music_video_workflow_matches_omni_node_interfaces(workflow):
    for node in workflow["nodes"]:
        if not node["type"].startswith("VLLMOmni"):
            continue
        cls = getattr(omni_nodes, node["type"])
        schema = cls.INPUT_TYPES()
        inputs = {**schema.get("required", {}), **schema.get("optional", {})}
        assert [input_["name"] for input_ in node["inputs"]] == [
            *schema.get("optional", {}),
            *schema.get("required", {}),
        ]
        for input_ in node["inputs"]:
            kind = inputs[input_["name"]][0]
            # Saved workflows record combo widgets as COMBO, not their option list.
            assert input_["type"] == ("COMBO" if isinstance(kind, list) else kind)
        assert tuple(output["type"] for output in node["outputs"]) == cls.RETURN_TYPES
        expected_widgets = len(schema.get("required", {}))
        if "seed" in schema.get("required", {}):
            expected_widgets += 1
        assert len(node["widgets_values"]) == expected_widgets


def test_music_video_shots_are_ref2va_driven_by_consecutive_track_slices(workflow):
    nodes, links = _index(workflow)
    _, shots = _shots(nodes, links)
    assert len(shots) == 3
    track = next(node for node in nodes.values() if node["type"] == "LoadAudio")
    image = next(node for node in nodes.values() if node["type"] == "LoadImage")
    starts = []
    for index, shot in enumerate(shots):
        assert shot["type"] == "VLLMOmniGenerateVideo"
        assert shot["title"].startswith(f"Shot {index + 1} ")
        for name in ("frame", "first_frame", "last_frame", "lora", "fast_h3", "latent_edit"):
            assert _source(nodes, links, shot, name) is None
        refs = _source(nodes, links, shot, "references")
        assert refs["type"] == "VLLMOmniVideoReferences"
        connected = {input_["name"] for input_ in refs["inputs"] if input_["link"] is not None}
        assert connected == {"image_1", "audio_1"}
        assert _source(nodes, links, refs, "image_1") is image
        driving = _source(nodes, links, refs, "audio_1")
        assert driving["type"] == "TrimAudioDuration"
        assert _source(nodes, links, driving, "audio") is track
        start, duration = driving["widgets_values"]
        starts.append(start)
        assert duration >= SHOT_SECONDS - TIME_TOLERANCE
        assert "<Picture 1>" in shot["widgets_values"][2]
    for index, start in enumerate(starts):
        assert start == pytest.approx(starts[0] + index * SHOT_SECONDS, abs=TIME_TOLERANCE)

    params = {_source(nodes, links, shot, "model_params")["id"] for shot in shots}
    assert len(params) == 1
    assert nodes[params.pop()]["widgets_values"] == [3.0, 12.0, "lock_source"]


def test_music_video_shares_settings_and_shot_length(workflow):
    nodes, links = _index(workflow)
    _, shots = _shots(nodes, links)
    for name, kind in (
        ("url", "PrimitiveString"),
        ("model", "PrimitiveString"),
        ("duration", "PrimitiveFloat"),
        ("sampling_params", "VLLMOmniDiffusionSampling"),
        ("model_params", "VLLMOmniMiniMaxH3Params"),
    ):
        sources = {_source(nodes, links, shot, name)["id"] for shot in shots}
        assert len(sources) == 1, name
        assert nodes[sources.pop()]["type"] == kind
    shot_length = _source(nodes, links, shots[0], "duration")
    duration = shot_length["widgets_values"][0]
    num_frames = round(duration * FPS)
    assert num_frames == SHOT_FRAMES
    assert (num_frames - 5) % 17 == 0
    for shot in shots:
        refs = _source(nodes, links, shot, "references")
        assert _source(nodes, links, _source(nodes, links, refs, "audio_1"), "duration") is shot_length
        assert shot["widgets_values"][:2] == ["http://localhost:8000/v1", "MiniMaxAI/MiniMax-H3"]
        assert shot["widgets_values"][4:8] == [1344, 768, FPS, duration]


def test_music_video_replaces_audio_with_the_supplied_track(workflow):
    nodes, links = _index(workflow)
    concat, shots = _shots(nodes, links)
    names = [input_["name"] for input_ in concat["inputs"]]
    assert names == [*(f"videos.video{i}" for i in range(len(shots) + 1)), "codec", "complete_audio"]
    assert concat["widgets_values"] == ["auto"]
    soundtrack = _source(nodes, links, concat, "complete_audio")
    assert soundtrack["type"] == "TrimAudioDuration"
    track = next(node for node in nodes.values() if node["type"] == "LoadAudio")
    assert _source(nodes, links, soundtrack, "audio") is track
    start, duration = soundtrack["widgets_values"]
    first_slice = _source(nodes, links, _source(nodes, links, shots[0], "references"), "audio_1")
    assert start == first_slice["widgets_values"][0]
    assert duration == pytest.approx(len(shots) * SHOT_SECONDS, abs=TIME_TOLERANCE)
    save = next(node for node in nodes.values() if node["type"] == "SaveVideo")
    assert _source(nodes, links, save, "video") is concat
    assert save["widgets_values"] == ["video/MiniMax-H3-MV", "mp4", "h264"]


def test_music_video_groups_contain_their_nodes(workflow):
    nodes, _ = _index(workflow)
    groups = {group["title"]: group["bounding"] for group in workflow["groups"]}

    def inside(node, bounding):
        x, y, w, h = bounding
        (nx, ny), (nw, nh) = node["pos"], node["size"]
        return x <= nx and y <= ny and nx + nw <= x + w and ny + nh <= y + h

    for node in nodes.values():
        containing = [title for title, bounding in groups.items() if inside(node, bounding)]
        assert len(containing) == 1, node["title"]
        if shot := re.match(r"Shot \d+ - ", node["title"]):
            assert containing[0].startswith(shot.group(0).removesuffix(" - ") + ":"), node["title"]


def test_music_video_defaults_and_portable_assets(workflow):
    nodes, _ = _index(workflow)
    by_type: dict[str, list[dict]] = {}
    for node in nodes.values():
        assert node["mode"] == 0, node["title"]
        by_type.setdefault(node["type"], []).append(node)
    assert [node["widgets_values"] for node in by_type["VLLMOmniDiffusionSampling"]] == [
        [1, 50, 1.0, 1.0, False, False, 42, "fixed"]
    ]
    for kind in ("LoadImage", "LoadAudio"):
        (loader,) = by_type[kind]
        filename = loader["widgets_values"][0]
        assert filename and Path(filename).name == filename
        assert ":" not in filename and "\\" not in filename
    assert set(by_type) == {
        "PrimitiveString",
        "PrimitiveFloat",
        "LoadImage",
        "LoadAudio",
        "TrimAudioDuration",
        "ConcatenateVideo",
        "SaveVideo",
        "MarkdownNote",
        "VLLMOmniVideoReferences",
        "VLLMOmniGenerateVideo",
        "VLLMOmniDiffusionSampling",
        "VLLMOmniMiniMaxH3Params",
    }
