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
    / "apps/ComfyUI-vLLM-Omni/example_workflows/vLLM-Omni MiniMax-H3 Multiframe Reference.json"
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


def _guides(workflow):
    nodes = {node["id"]: node for node in workflow["nodes"]}
    generate = next(node for node in nodes.values() if node["type"] == "VLLMOmniGenerateVideo")
    chain = []
    guide, _ = _source(workflow, generate, "timeline_guides")
    while True:
        assert guide["type"] == "VLLMOmniMiniMaxH3Guide"
        chain.append(guide)
        upstream = next(input_ for input_ in guide["inputs"] if input_["name"] == "timeline_guides")
        if upstream["link"] is None:
            break
        guide, _ = _source(workflow, guide, "timeline_guides")
    chain.reverse()
    return chain


def test_multiframe_workflow_connections(workflow):
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


def test_multiframe_workflow_matches_omni_node_interfaces(workflow):
    for node in workflow["nodes"]:
        if not node["type"].startswith("VLLMOmni"):
            continue
        cls = getattr(omni_nodes, node["type"])
        schema = cls.INPUT_TYPES()
        inputs = {**schema.get("required", {}), **schema.get("optional", {})}
        for input_ in node["inputs"]:
            kind = inputs[input_["name"]][0]
            # Saved workflows record combo widgets as COMBO, not their option list.
            assert input_["type"] == ("COMBO" if isinstance(kind, list) else kind)
        assert tuple(output["type"] for output in node["outputs"]) == cls.RETURN_TYPES
        expected_widgets = len(schema.get("required", {}))
        if "seed" in schema.get("required", {}):
            expected_widgets += 1
        assert len(node["widgets_values"]) == expected_widgets
    refs = next(node for node in workflow["nodes"] if node["type"] == "VLLMOmniVideoReferences")
    assert [input_["name"] for input_ in refs["inputs"]] == list(
        omni_nodes.VLLMOmniVideoReferences.INPUT_TYPES()["optional"]
    )


def test_multiframe_workflow_anchor_layout(workflow):
    nodes = {node["id"]: node for node in workflow["nodes"]}
    generate = next(node for node in nodes.values() if node["type"] == "VLLMOmniGenerateVideo")

    # One semantic reference: frame 1 stays a <Picture N> reference for the prompt.
    references, _ = _source(workflow, generate, "references")
    assert references["type"] == "VLLMOmniVideoReferences"
    connected = {input_["name"] for input_ in references["inputs"] if input_["link"] is not None}
    assert connected == {"image_1"}

    # Frames 2-4 are timeline anchors at 1.5 s, 3.0 s, and 5.0 s on the 24 FPS timeline.
    guides = _guides(workflow)
    assert [guide["widgets_values"] for guide in guides] == [[36], [72], [120]]

    def media(guide):
        return {i["name"] for i in guide["inputs"] if i["name"] in {"image", "video", "audio"} and i["link"]}

    assert [media(guide) for guide in guides] == [{"image"}, {"image"}, {"image"}]
    for guide in guides:
        image, slot = _source(workflow, guide, "image")
        assert (image["type"], image["outputs"][slot]["type"]) == ("LoadImage", "IMAGE")

    # Every anchor is a distinct loader, so swapping one frame cannot move another.
    loaders = [next(input_["link"] for input_ in guide["inputs"] if input_["name"] == "image") for guide in guides]
    assert len(set(loaders)) == 3


def test_multiframe_workflow_defaults_and_portable_assets(workflow):
    nodes = {node["id"]: node for node in workflow["nodes"]}
    generate = next(node for node in nodes.values() if node["type"] == "VLLMOmniGenerateVideo")
    values = generate["widgets_values"]
    assert values[:2] == ["http://localhost:8000/v1", "MiniMaxAI/MiniMax-H3"]
    assert values[3] == ""
    assert values[4:8] == [1344, 768, 24, 5.167]

    # Guided requests are Ref2VA here: references, then the three guides.
    connected = {input_["name"] for input_ in generate["inputs"] if input_["link"] is not None}
    assert connected == {"references", "sampling_params", "model_params", "timeline_guides"}

    # Base H3 only: FastH3, request LoRA, and latent-mask editing reject guides.
    for node_type in ("VLLMOmniRemoteLoRA", "VLLMOmniFastH3Deployment", "VLLMOmniLatentMaskEditing"):
        assert all(node["type"] != node_type for node in nodes.values())

    # 124 frames is the 17k+5 lattice H3 accepts; the anchors are its 1.5/3.0/5.0 s frames.
    num_frames = round(values[6] * values[7])
    assert (num_frames - 5) % 17 == 0
    assert num_frames == 124
    assert [guide["widgets_values"][0] for guide in _guides(workflow)] == [
        round(1.5 * 24),
        round(3.0 * 24),
        round(5.0 * 24),
    ]

    # Guide images are compositions, not references: only <Picture 1> is tagged.
    assert "<Picture 1>" in values[2]
    assert "<Picture 2>" not in values[2]

    sampling = next(node for node in nodes.values() if node["type"] == "VLLMOmniDiffusionSampling")
    assert sampling["widgets_values"] == [1, 50, 1.0, 1.0, False, False, 42, "fixed"]
    params = next(node for node in nodes.values() if node["type"] == "VLLMOmniMiniMaxH3Params")
    assert params["widgets_values"] == [3.0, 12.0, "native"]

    save = next(node for node in nodes.values() if node["type"] == "SaveVideo")
    assert _source(workflow, save, "video")[0] is generate
    assert save["widgets_values"] == ["video/MiniMax-H3-Multiframe-Reference", "mp4", "h264"]

    assert set(node["type"] for node in nodes.values()) == {
        "LoadImage",
        "SaveVideo",
        "MarkdownNote",
        "VLLMOmniVideoReferences",
        "VLLMOmniMiniMaxH3Guide",
        "VLLMOmniGenerateVideo",
        "VLLMOmniDiffusionSampling",
        "VLLMOmniMiniMaxH3Params",
    }
    for node in nodes.values():
        if node["type"] != "LoadImage":
            continue
        filename = node["widgets_values"][0]
        assert filename and Path(filename).name == filename
        assert ":" not in filename and "\\" not in filename
    notes = [node for node in nodes.values() if node["type"] == "MarkdownNote"]
    assert any("Four anchors" in note["widgets_values"][0] for note in notes)
    assert any("Server requirements" in note["widgets_values"][0] for note in notes)
