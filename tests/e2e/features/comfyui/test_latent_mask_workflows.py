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
    / "apps/ComfyUI-vLLM-Omni/example_workflows/vLLM-Omni MiniMax-H3 Latent Mask Editing.json"
)


@pytest.fixture
def workflow():
    return json.loads(WORKFLOW.read_text())


def _source_of(workflow: dict, node: dict, input_name: str) -> dict:
    nodes = {n["id"]: n for n in workflow["nodes"]}
    links = {link[0]: link for link in workflow["links"]}
    link_id = next(i["link"] for i in node["inputs"] if i["name"] == input_name)
    assert link_id is not None, f"{node['type']}.{input_name} is not connected"
    return nodes[links[link_id][1]]


def _cases(workflow: dict) -> dict[str, dict]:
    return {
        node["title"].removeprefix("Generate Video(").removesuffix(")"): node
        for node in workflow["nodes"]
        if node["type"] == "VLLMOmniGenerateVideo"
    }


def test_latent_mask_workflow_connections(workflow):
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


def test_latent_mask_workflow_matches_omni_node_interfaces(workflow):
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


def test_latent_mask_workflow_defaults(workflow):
    cases = _cases(workflow)
    assert set(cases) == {"Object Removal", "Inpainting", "Continuation", "Extension"}
    for generate in cases.values():
        assert generate["widgets_values"][0] == "http://127.0.0.1:8000/v1"
        assert generate["widgets_values"][4:6] == [1344, 768]


@pytest.mark.parametrize("case", ["Object Removal", "Inpainting"])
def test_spatial_cases_use_a_composited_mask(workflow, case):
    edit = _source_of(workflow, _cases(workflow)[case], "latent_edit")
    assert _source_of(workflow, edit, "source_video")["type"] == "LoadVideo"
    assert _source_of(workflow, edit, "video_mask")["type"] == "MaskComposite"


@pytest.mark.parametrize("case", ["Continuation", "Extension"])
def test_temporal_cases_build_a_per_frame_mask(workflow, case):
    # The server pads a short [T, H, W] mask with its last slice, so the mask must
    # come from the Temporal Mask node (one slice per output frame), not a
    # hand-batched stack of a few slices.
    generate = _cases(workflow)[case]
    edit = _source_of(workflow, generate, "latent_edit")
    temporal = _source_of(workflow, edit, "video_mask")
    assert temporal["type"] == "VLLMOmniMiniMaxH3TemporalMask"
    assert _source_of(workflow, temporal, "images")["type"] == "GetVideoComponents"
    assert _source_of(workflow, edit, "source_video")["type"] == "LoadVideo"
    _, duration, mode, _ = temporal["widgets_values"]
    assert mode == case.lower()
    assert duration == generate["widgets_values"][7]
