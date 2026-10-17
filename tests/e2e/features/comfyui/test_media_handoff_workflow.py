# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
from graphlib import TopologicalSorter
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

WORKFLOW = Path(__file__).resolve().parents[4] / "apps/ComfyUI-vLLM-Omni/example_workflows/vLLM-Omni Media Handoff.json"


@pytest.fixture
def graph():
    workflow = json.loads(WORKFLOW.read_text())
    return {node["id"]: node for node in workflow["nodes"]}, {link[0]: link for link in workflow["links"]}


def source(nodes, links, node, input_name):
    input_ = next(value for value in node["inputs"] if value["name"] == input_name)
    _, origin, slot, _, _, _ = links[input_["link"]]
    return nodes[origin], slot


def test_graph_connections_are_consistent_and_acyclic(graph):
    nodes, links = graph
    dependencies: dict[int, set[int]] = {node_id: set() for node_id in nodes}
    for link_id, origin, output_slot, target, input_slot, kind in links.values():
        output = nodes[origin]["outputs"][output_slot]
        input_ = nodes[target]["inputs"][input_slot]
        assert output["type"] == input_["type"] == kind
        assert link_id in output["links"]
        assert input_["link"] == link_id
        dependencies[target].add(origin)
    assert len(tuple(TopologicalSorter(dependencies).static_order())) == len(nodes)


def test_soundtrack_override_receives_original_video(graph):
    nodes, links = graph
    overrides = [node for node in nodes.values() if node["type"] == "ConcatenateVideo"]
    assert len(overrides) == 3
    for node in overrides:
        origin, output_slot = source(nodes, links, node, "videos.video0")
        assert origin["type"] == "LoadVideo" and output_slot == 0
        assert node["widgets_values"] == ["auto"]
        assert [value["name"] for value in node["inputs"]] == ["videos.video0", "complete_audio"]
    assert not any(node["type"] == "CreateVideo" for node in nodes.values())
    assert all(node["widgets_values"][2] == "auto" for node in nodes.values() if node["type"] == "SaveVideo")


def test_audio_policies_and_original_output_are_distinct(graph):
    nodes, links = graph
    outputs = {node["widgets_values"][0]: node for node in nodes.values() if node["type"] == "SaveVideo"}
    assert set(outputs) == {"CUI03/preserve", "CUI03/replace", "CUI03/mix", "CUI03/loop"}
    original, _ = source(nodes, links, outputs["CUI03/preserve"], "video")
    assert original["type"] == "LoadVideo"
    for policy, expected in (("replace", "LoadAudio"), ("mix", "AudioMerge"), ("loop", "VLLMOmniAudioToVideoDuration")):
        video, _ = source(nodes, links, outputs[f"CUI03/{policy}"], "video")
        assert video["type"] == "ConcatenateVideo"
        audio, _ = source(nodes, links, video, "complete_audio")
        assert audio["type"] == expected
    mix = next(node for node in nodes.values() if node["type"] == "AudioMerge")
    assert mix["widgets_values"] == ["mean"]
    assert source(nodes, links, mix, "audio1")[0]["type"] == "GetVideoComponents"
    assert source(nodes, links, mix, "audio2")[0]["type"] == "LoadAudio"
    repeat = next(node for node in nodes.values() if node["type"] == "VLLMOmniAudioToVideoDuration")
    assert repeat["widgets_values"] == ["loop"]
    assert source(nodes, links, repeat, "video")[0]["type"] == "LoadVideo"
    assert source(nodes, links, repeat, "audio")[0]["type"] == "LoadAudio"


def test_template_uses_media_nodes_and_portable_assets(graph):
    nodes, _ = graph
    assert {node["type"] for node in nodes.values()} == {
        "LoadVideo",
        "LoadAudio",
        "GetVideoComponents",
        "ConcatenateVideo",
        "SaveVideo",
        "AudioMerge",
        "VLLMOmniAudioToVideoDuration",
        "MarkdownNote",
    }
    for node in nodes.values():
        if node["type"] in {"LoadVideo", "LoadAudio"}:
            filename = node["widgets_values"][0]
            assert filename == Path(filename).name
            assert ":" not in filename and "\\" not in filename
