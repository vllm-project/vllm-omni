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
    / "apps/ComfyUI-vLLM-Omni/example_workflows/vLLM-Omni MiniMax-H3 Reference to Video.json"
)


@pytest.fixture
def workflow():
    return json.loads(WORKFLOW.read_text())


def test_reference_workflow_connections(workflow):
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
        for slot, output in enumerate(node.get("outputs", [])):
            for link_id in output.get("links") or []:
                assert links[link_id][1:3] == [node["id"], slot]


def test_reference_workflow_matches_omni_node_interfaces(workflow):
    for node in workflow["nodes"]:
        if not node["type"].startswith("VLLMOmni"):
            continue
        cls = getattr(omni_nodes, node["type"])
        schema = cls.INPUT_TYPES()
        inputs = {**schema.get("required", {}), **schema.get("optional", {})}
        for input_ in node["inputs"]:
            expected_type = inputs[input_["name"]][0]
            # ComfyUI serializes a list-backed dropdown as the generic COMBO
            # type rather than embedding all choices in the workflow input.
            if isinstance(expected_type, list):
                expected_type = "COMBO"
            assert input_["type"] == expected_type
        assert tuple(output["type"] for output in node["outputs"]) == cls.RETURN_TYPES
        expected_widgets = len(schema.get("required", {}))
        if "seed" in schema.get("required", {}):
            expected_widgets += 1
        assert len(node["widgets_values"]) == expected_widgets
    refs = next(node for node in workflow["nodes"] if node["type"] == "VLLMOmniVideoReferences")
    assert [input_["name"] for input_ in refs["inputs"]] == list(
        omni_nodes.VLLMOmniVideoReferences.INPUT_TYPES()["optional"]
    )


def test_reference_workflow_default_route_and_output(workflow):
    nodes = {node["id"]: node for node in workflow["nodes"]}
    links = {link[0]: link for link in workflow["links"]}
    generate = next(node for node in nodes.values() if node["type"] == "VLLMOmniGenerateVideo")
    inputs = {input_["name"]: input_["link"] for input_ in generate["inputs"]}
    assert inputs["frame"] is None
    assert inputs["lora"] is None
    for name, kind in (
        ("references", "VLLMOmniVideoReferences"),
        ("sampling_params", "VLLMOmniDiffusionSampling"),
        ("model_params", "VLLMOmniMiniMaxH3Params"),
    ):
        assert nodes[links[inputs[name]][1]]["type"] == kind
    save = next(node for node in nodes.values() if node["type"] == "SaveVideo")
    assert links[save["inputs"][0]["link"]][1] == generate["id"]
    assert save["widgets_values"] == ["video/MiniMax-H3-Ref2VA", "mp4", "h264"]


def test_reference_workflow_defaults_and_portable_assets(workflow):
    by_type = {node["type"]: node for node in workflow["nodes"]}
    values = by_type["VLLMOmniGenerateVideo"]["widgets_values"]
    assert values[0] == "http://localhost:8000/v1"
    assert values[1] == "MiniMaxAI/MiniMax-H3"
    assert "<Picture 1>" in values[2]
    assert values[4:8] == [1344, 768, 24, 5.167]
    num_frames = round(values[7] * values[6])
    assert num_frames == 124
    assert values[3] == ""
    assert (num_frames - 5) % 17 == 0
    assert 4 <= values[7] <= 15
    assert by_type["VLLMOmniDiffusionSampling"]["widgets_values"] == [1, 50, 1.0, 1.0, False, False, 42, "fixed"]
    assert by_type["VLLMOmniMiniMaxH3Params"]["widgets_values"] == [3.0, 12.0, "auto"]
    assert by_type["VLLMOmniRemoteLoRA"]["widgets_values"][0] == ""
    for kind in ("LoadImage", "LoadVideo", "LoadAudio"):
        filename = by_type[kind]["widgets_values"][0]
        assert filename and Path(filename).name == filename
        assert ":" not in filename and "\\" not in filename
    assert set(by_type) == {
        "LoadImage",
        "LoadVideo",
        "LoadAudio",
        "SaveVideo",
        "MarkdownNote",
        "VLLMOmniVideoReferences",
        "VLLMOmniGenerateVideo",
        "VLLMOmniDiffusionSampling",
        "VLLMOmniMiniMaxH3Params",
        "VLLMOmniRemoteLoRA",
    }
