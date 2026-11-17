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
    / "apps/ComfyUI-vLLM-Omni/example_workflows/vLLM-Omni MiniMax-H3 All-in-One.json"
)
ACTIVE, MUTED = 0, 2
FRAME_INPUTS = ("frame", "first_frame", "last_frame", "references")
# Mode group title prefix -> (saved video prefix, frame/reference inputs that select the server task)
MODES = {
    "T2VA": ("video/MiniMax-H3-AIO-T2VA", set()),
    "FL2VA": ("video/MiniMax-H3-AIO-FL2VA", {"first_frame", "last_frame"}),
    "Ref2VA": ("video/MiniMax-H3-AIO-Ref2VA", {"references"}),
}


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


def _generate_nodes(nodes):
    return {node["title"].split(" ")[0]: node for node in nodes.values() if node["type"] == "VLLMOmniGenerateVideo"}


def test_all_in_one_workflow_connections(workflow):
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


def test_all_in_one_workflow_matches_omni_node_interfaces(workflow):
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


def test_all_in_one_workflow_routes_each_mode(workflow):
    nodes, links = _index(workflow)
    generates = _generate_nodes(nodes)
    assert set(generates) == set(MODES)
    for mode, (prefix, frame_inputs) in MODES.items():
        generate = generates[mode]
        connected = {name for name in FRAME_INPUTS if _source(nodes, links, generate, name) is not None}
        assert connected == frame_inputs, mode
        for name in ("lora", "fast_h3", "latent_edit"):
            assert _source(nodes, links, generate, name) is None
        saves = [
            node
            for node in nodes.values()
            if node["type"] == "SaveVideo" and _source(nodes, links, node, "video") is generate
        ]
        assert [save["widgets_values"] for save in saves] == [[prefix, "mp4", "h264"]]

    for name in ("first_frame", "last_frame"):
        assert _source(nodes, links, generates["FL2VA"], name)["type"] == "LoadImage"
    refs = _source(nodes, links, generates["Ref2VA"], "references")
    assert refs["type"] == "VLLMOmniVideoReferences"
    assert _source(nodes, links, refs, "image_1")["type"] == "LoadImage"


def test_all_in_one_workflow_shares_settings_across_modes(workflow):
    nodes, links = _index(workflow)
    for name, kind in (
        ("url", "PrimitiveString"),
        ("model", "PrimitiveString"),
        ("duration", "PrimitiveFloat"),
        ("sampling_params", "VLLMOmniDiffusionSampling"),
        ("model_params", "VLLMOmniMiniMaxH3Params"),
    ):
        sources = {_source(nodes, links, generate, name)["id"] for generate in _generate_nodes(nodes).values()}
        assert len(sources) == 1, name
        assert nodes[sources.pop()]["type"] == kind


def test_all_in_one_workflow_enables_only_t2va_by_default(workflow):
    for node in workflow["nodes"]:
        mode = node["title"].split(" ")[0]
        if node["type"] in ("LoadImage", "VLLMOmniVideoReferences", "VLLMOmniGenerateVideo", "SaveVideo"):
            assert mode in MODES, node["title"]
        expected = MUTED if mode in MODES and mode != "T2VA" else ACTIVE
        assert node["mode"] == expected, node["title"]


def test_all_in_one_workflow_groups_contain_their_nodes(workflow):
    nodes, _ = _index(workflow)
    groups = {group["title"]: group["bounding"] for group in workflow["groups"]}

    def inside(node, bounding):
        x, y, w, h = bounding
        (nx, ny), (nw, nh) = node["pos"], node["size"]
        return x <= nx and y <= ny and nx + nw <= x + w and ny + nh <= y + h

    for node in nodes.values():
        containing = [title for title, bounding in groups.items() if inside(node, bounding)]
        assert len(containing) == 1, node["title"]
        mode = node["title"].split(" ")[0]
        if mode in MODES:
            assert mode in containing[0], node["title"]


def test_all_in_one_workflow_defaults_and_portable_assets(workflow):
    nodes, _ = _index(workflow)
    by_title = {node["title"]: node for node in nodes.values()}
    assert by_title["Server URL (all modes)"]["widgets_values"] == ["http://localhost:8000/v1"]
    assert by_title["Served model name (all modes)"]["widgets_values"] == ["MiniMaxAI/MiniMax-H3"]
    duration = by_title["Duration seconds (5.167 = 124 frames)"]["widgets_values"][0]
    for generate in _generate_nodes(nodes).values():
        values = generate["widgets_values"]
        assert values[:2] == ["http://localhost:8000/v1", "MiniMaxAI/MiniMax-H3"]
        assert values[3] == ""
        assert values[4:8] == [1344, 768, 24, duration]
        num_frames = round(duration * values[6])
        assert num_frames == 124
        assert (num_frames - 5) % 17 == 0
        assert 4 <= duration <= 15
    assert "<Picture 1>" in _generate_nodes(nodes)["Ref2VA"]["widgets_values"][2]

    by_type: dict[str, list[dict]] = {}
    for node in nodes.values():
        by_type.setdefault(node["type"], []).append(node)
    assert [node["widgets_values"] for node in by_type["VLLMOmniDiffusionSampling"]] == [
        [1, 50, 1.0, 1.0, False, False, 42, "fixed"]
    ]
    assert [node["widgets_values"] for node in by_type["VLLMOmniMiniMaxH3Params"]] == [[3.0, 12.0, "native"]]
    for node in by_type["LoadImage"]:
        filename = node["widgets_values"][0]
        assert filename and Path(filename).name == filename
        assert ":" not in filename and "\\" not in filename
    assert set(by_type) == {
        "PrimitiveString",
        "PrimitiveFloat",
        "LoadImage",
        "SaveVideo",
        "MarkdownNote",
        "VLLMOmniVideoReferences",
        "VLLMOmniGenerateVideo",
        "VLLMOmniDiffusionSampling",
        "VLLMOmniMiniMaxH3Params",
    }
    guide = next(node for node in by_type["MarkdownNote"] if node["title"] == "How to use this graph")
    assert "no prompt-assistance stage" in guide["widgets_values"][0]


def test_generate_video_accepts_url_and_model_from_shared_primitives():
    # ComfyUI passes linked inputs to VALIDATE_INPUTS as None; the template links url and model.
    assert omni_nodes.VLLMOmniGenerateVideo.VALIDATE_INPUTS(url=None, model=None) is True
    assert omni_nodes.VLLMOmniGenerateVideo.VALIDATE_INPUTS(url="", model="m") == "URL must not be empty"
    assert omni_nodes.VLLMOmniGenerateVideo.VALIDATE_INPUTS(url="u", model="") == "Model must not be empty"
