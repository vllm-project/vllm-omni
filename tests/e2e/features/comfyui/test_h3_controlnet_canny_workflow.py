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
    / "apps/ComfyUI-vLLM-Omni/example_workflows/vLLM-Omni MiniMax-H3 ControlNet Canny.json"
)


@pytest.fixture
def workflow():
    return json.loads(WORKFLOW.read_text())


def _graph(workflow):
    return {node["id"]: node for node in workflow["nodes"]}, {link[0]: link for link in workflow["links"]}


def _source(nodes, links, node, name):
    link = next(input_["link"] for input_ in node["inputs"] if input_["name"] == name)
    return None if link is None else nodes[links[link][1]]


def test_canny_workflow_connections(workflow):
    nodes, links = _graph(workflow)
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


def test_canny_workflow_matches_omni_node_interfaces(workflow):
    for node in workflow["nodes"]:
        if not node["type"].startswith("VLLMOmni"):
            continue
        cls = getattr(omni_nodes, node["type"])
        schema = cls.INPUT_TYPES()
        inputs = {**schema.get("required", {}), **schema.get("optional", {})}
        for input_ in node["inputs"]:
            kind = inputs[input_["name"]][0]
            assert input_["type"] == ("COMBO" if isinstance(kind, list) else kind)
        assert tuple(output["type"] for output in node["outputs"]) == cls.RETURN_TYPES[: len(node["outputs"])]
        expected_widgets = len(schema.get("required", {}))
        if "seed" in schema.get("required", {}):
            expected_widgets += 1
        optional_widgets = int("enabled" in schema.get("optional", {}))
        assert expected_widgets <= len(node["widgets_values"]) <= expected_widgets + optional_widgets


def test_canny_workflow_builds_control_video_in_graph(workflow):
    nodes, links = _graph(workflow)
    by_type = {node["type"]: node for node in nodes.values()}
    control = by_type["VLLMOmniMiniMaxH3Control"]
    assert control["widgets_values"] == ["canny", 1.0]
    for name in ("source_video", "mask", "mask_video"):
        assert _source(nodes, links, control, name) is None

    # Load Video -> components -> fit -> trim -> Canny -> Create Video -> control_video
    chain = ["CreateVideo", "Canny", "ImageFromBatch", "ImageScale", "GetVideoComponents", "LoadVideo"]
    node = control
    for kind, name in zip(chain, ("control_video", "images", "image", "image", "image", "video")):
        node = _source(nodes, links, node, name)
        assert node["type"] == kind
    assert by_type["ImageScale"]["widgets_values"][1:] == [1344, 768, "center"]
    assert by_type["ImageFromBatch"]["widgets_values"] == [0, 124]
    assert by_type["CreateVideo"]["widgets_values"][0] == 24

    generate = by_type["VLLMOmniGenerateVideo"]
    for name, kind in (
        ("control", "VLLMOmniMiniMaxH3Control"),
        ("sampling_params", "VLLMOmniDiffusionSampling"),
        ("model_params", "VLLMOmniMiniMaxH3Params"),
    ):
        assert _source(nodes, links, generate, name)["type"] == kind
    saves = {
        _source(nodes, links, node, "video")["type"]: node for node in nodes.values() if node["type"] == "SaveVideo"
    }
    assert set(saves) == {"VLLMOmniGenerateVideo", "CreateVideo"}
    assert saves["VLLMOmniGenerateVideo"]["widgets_values"] == ["video/MiniMax_H3_Control_canny", "mp4", "h264"]


def test_canny_workflow_defaults_and_portable_assets(workflow):
    by_type = {node["type"]: node for node in workflow["nodes"]}
    values = by_type["VLLMOmniGenerateVideo"]["widgets_values"]
    assert values[0] == "http://localhost:8000/v1"
    assert values[1] == "MiniMaxAI/MiniMax-H3"
    assert values[2] and values[3] == ""
    assert values[4:8] == [1344, 768, 24, 5.167]
    num_frames = round(values[7] * values[6])
    assert num_frames == by_type["ImageFromBatch"]["widgets_values"][1]
    assert (num_frames - 5) % 17 == 0
    assert by_type["VLLMOmniDiffusionSampling"]["widgets_values"] == [1, 40, 1.0, 1.0, False, False, 43, "fixed"]
    assert by_type["VLLMOmniMiniMaxH3Params"]["widgets_values"] == [3.0, 12.0, "native"]
    filename = by_type["LoadVideo"]["widgets_values"][0]
    assert filename and Path(filename).name == filename
    assert ":" not in filename and "\\" not in filename
    assert "--controlnet-model-path" in by_type["MarkdownNote"]["widgets_values"][0]
    assert set(by_type) == {
        "LoadVideo",
        "GetVideoComponents",
        "ImageScale",
        "ImageFromBatch",
        "Canny",
        "CreateVideo",
        "SaveVideo",
        "MarkdownNote",
        "VLLMOmniMiniMaxH3Control",
        "VLLMOmniGenerateVideo",
        "VLLMOmniDiffusionSampling",
        "VLLMOmniMiniMaxH3Params",
    }
