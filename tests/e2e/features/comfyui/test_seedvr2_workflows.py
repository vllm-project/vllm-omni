# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
from graphlib import TopologicalSorter
from pathlib import Path

import pytest
from comfyui_vllm_omni import nodes as omni_nodes

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
WORKFLOWS = Path(__file__).resolve().parents[4] / "apps/ComfyUI-vLLM-Omni/example_workflows"


@pytest.mark.parametrize(
    "filename",
    [
        "vLLM-Omni SeedVR2 Video Restoration.json",
        "vLLM-Omni MiniMax H3 Remote SeedVR2.json",
    ],
)
def test_restoration_workflow_interfaces_and_handoffs(filename):
    workflow = json.loads((WORKFLOWS / filename).read_text())
    nodes = {node["id"]: node for node in workflow["nodes"]}
    links = {link[0]: link for link in workflow["links"]}
    dependencies: dict[int, set[int]] = {key: set() for key in nodes}
    for link_id, source, output_slot, target, input_slot, kind in links.values():
        output = nodes[source]["outputs"][output_slot]
        input_ = nodes[target]["inputs"][input_slot]
        assert output["type"] == input_["type"] == kind
        assert link_id in output["links"]
        assert input_["link"] == link_id
        dependencies[target].add(source)
    assert len(tuple(TopologicalSorter(dependencies).static_order())) == len(nodes)
    for node in nodes.values():
        if node["type"].startswith("VLLMOmni"):
            cls = getattr(omni_nodes, node["type"])
            schema = cls.INPUT_TYPES()
            inputs = schema.get("required", {}) | schema.get("optional", {})
            assert all(port["type"] == inputs[port["name"]][0] for port in node["inputs"])
            assert tuple(port["type"] for port in node["outputs"]) == cls.RETURN_TYPES

    restore = next(node for node in nodes.values() if node["type"] == "VLLMOmniRestoreVideo")
    source = nodes[links[restore["inputs"][0]["link"]][1]]
    assert source["type"] == ("LoadVideo" if "Restoration" in filename else "VLLMOmniGenerateVideo")
    restore_link = links[restore["outputs"][0]["links"][0]]
    assert nodes[restore_link[3]]["type"] == "SaveVideo"
    assert restore["widgets_values"][0:2] == ["http://localhost:8098/v1", "seedvr2"]
    assert not any("SeedVR2" in node["type"] for node in nodes.values())  # no local SeedVR2 loaders
    if source["type"] == "LoadVideo":
        assert Path(source["widgets_values"][0]).name == source["widgets_values"][0]
    else:
        assert source["widgets_values"][0] != restore["widgets_values"][0]
        assert any(nodes[links[link_id][3]]["type"] == "SaveVideo" for link_id in source["outputs"][0]["links"])
