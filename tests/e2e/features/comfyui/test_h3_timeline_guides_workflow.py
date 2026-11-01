# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
from graphlib import TopologicalSorter
from pathlib import Path

import pytest
from comfyui_vllm_omni import nodes as omni_nodes

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

WORKFLOW = (
    Path(__file__).resolve().parents[4]
    / "apps/ComfyUI-vLLM-Omni/example_workflows/vLLM-Omni MiniMax-H3 Timeline Guides.json"
)


@pytest.fixture
def workflow():
    return json.loads(WORKFLOW.read_text())


def _source(workflow, node, input_name):
    links = {link[0]: link for link in workflow["links"]}
    nodes = {node["id"]: node for node in workflow["nodes"]}
    link_id = next(input_["link"] for input_ in node["inputs"] if input_["name"] == input_name)
    assert link_id is not None, f"{node['type']}.{input_name} is not connected"
    return nodes[links[link_id][1]], links[link_id][2]


def test_timeline_workflow_connections(workflow):
    nodes = {node["id"]: node for node in workflow["nodes"]}
    links = {link[0]: link for link in workflow["links"]}
    assert len(nodes) == len(workflow["nodes"])
    assert len(links) == len(workflow["links"])
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


def test_timeline_workflow_matches_omni_node_interfaces(workflow):
    for node in workflow["nodes"]:
        if not node["type"].startswith("VLLMOmni"):
            continue
        cls = getattr(omni_nodes, node["type"])
        schema = cls.INPUT_TYPES()
        inputs = {**schema.get("required", {}), **schema.get("optional", {})}
        for input_ in node["inputs"]:
            kind = inputs[input_["name"]][0]
            assert input_["type"] == ("COMBO" if isinstance(kind, list) else kind)
        assert tuple(output["type"] for output in node["outputs"]) == cls.RETURN_TYPES
        expected_widgets = len(schema.get("required", {}))
        if "seed" in schema.get("required", {}):
            expected_widgets += 1
        assert len(node["widgets_values"]) == expected_widgets


def test_timeline_workflow_guide_chain(workflow):
    nodes = {node["id"]: node for node in workflow["nodes"]}
    generate = next(node for node in nodes.values() if node["type"] == "VLLMOmniGenerateVideo")
    chain = []
    guide, _ = _source(workflow, generate, "timeline_guides")
    while True:
        assert guide["type"] == "VLLMOmniMiniMaxH3Guide"
        chain.append(guide)
        if next(input_ for input_ in guide["inputs"] if input_["name"] == "timeline_guides")["link"] is None:
            break
        guide, _ = _source(workflow, guide, "timeline_guides")
    chain.reverse()
    assert [guide["widgets_values"] for guide in chain] == [[36], [72], [-22]]

    def media(guide):
        return {i["name"] for i in guide["inputs"] if i["name"] in {"image", "video", "audio"} and i["link"]}

    assert [media(guide) for guide in chain] == [{"image"}, {"image"}, {"video", "audio"}]
    clip, _ = _source(workflow, chain[2], "video")
    components, audio_slot = _source(workflow, chain[2], "audio")
    assert components["type"] == "GetVideoComponents"
    assert components["outputs"][audio_slot]["name"] == "audio"
    assert _source(workflow, components, "video")[0] is clip


def test_timeline_workflow_defaults_and_portable_assets(workflow):
    nodes = {node["id"]: node for node in workflow["nodes"]}
    by_type = {node["type"]: node for node in nodes.values()}
    generate = by_type["VLLMOmniGenerateVideo"]
    values = generate["widgets_values"]
    assert values[:2] == ["http://localhost:8000/v1", "MiniMaxAI/MiniMax-H3"]
    assert values[3] == ""
    # Guided requests must use 24 FPS; 5.167 s gives a 17k+5 frame count.
    assert values[4:8] == [1344, 768, 24, 5.167]
    assert (round(values[6] * values[7]) - 5) % 17 == 0
    assert _source(workflow, generate, "sampling_params")[0]["type"] == "VLLMOmniDiffusionSampling"
    assert _source(workflow, generate, "model_params")[0]["type"] == "VLLMOmniMiniMaxH3Params"
    assert {i["name"] for i in generate["inputs"] if i["link"] is not None} == {
        "sampling_params",
        "model_params",
        "timeline_guides",
    }
    assert by_type["VLLMOmniDiffusionSampling"]["widgets_values"] == [1, 50, 1.0, 1.0, False, False, 42, "fixed"]
    assert by_type["VLLMOmniMiniMaxH3Params"]["widgets_values"] == [3.0, 12.0, "native"]
    save = by_type["SaveVideo"]
    assert _source(workflow, save, "video")[0] is generate
    assert save["widgets_values"] == ["video/MiniMax-H3-Timeline-Guides", "mp4", "h264"]
    for node in nodes.values():
        if node["type"] in {"LoadImage", "LoadVideo"}:
            filename = node["widgets_values"][0]
            assert filename and Path(filename).name == filename
            assert ":" not in filename and "\\" not in filename
