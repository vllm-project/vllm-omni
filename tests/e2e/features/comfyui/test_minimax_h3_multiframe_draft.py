# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU checks of the non-executable WF-04 development layout.

Run with --noconftest to avoid importing the inference/ComfyUI test environment.
These are structure checks; they do not establish API or model support.
"""

import ast
import json
from pathlib import Path, PurePosixPath, PureWindowsPath

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

ROOT = Path(__file__).resolve().parents[4]
APP = ROOT / "apps" / "ComfyUI-vLLM-Omni"
DRAFT = APP / "docs" / "drafts" / "MiniMax_H3_Multiframe_Reference.draft.json"
GRAPH = json.loads(DRAFT.read_text())
CONTRACT = GRAPH["extra"]["wf04"]
NODES = {node["id"]: node for node in GRAPH["nodes"]}


def _node_schema(kind):
    """Read literal INPUT_TYPES without importing Torch or mocking the nodes."""
    tree = ast.parse((APP / "comfyui_vllm_omni" / "nodes.py").read_text())
    cls = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == kind)
    method = next(item for item in cls.body if isinstance(item, ast.FunctionDef) and item.name == "INPUT_TYPES")
    result = next(item for item in method.body if isinstance(item, ast.Return))
    assert result.value is not None
    return ast.literal_eval(result.value)


def test_incomplete_workflow_cannot_silently_generate_plain_ref2va():
    assert CONTRACT["status"] == "BLOCKED_DEPENDENCIES"
    outputs = [n for n in NODES.values() if n["type"] in {"VLLMOmniGenerateVideo", "SaveVideo"}]
    assert {n["type"] for n in outputs} == {"VLLMOmniGenerateVideo", "SaveVideo"}
    assert all(n["mode"] == 2 for n in outputs)
    assert not (APP / "example_workflows" / DRAFT.name).exists()
    assert CONTRACT["guide_node_type"] is None
    assert CONTRACT["guide_api_fields"] is None
    assert set(CONTRACT["validation"].values()) == {"NOT_RUN"}


def test_four_images_are_not_misrepresented_as_four_timeline_guides():
    anchors = CONTRACT["anchors"]
    assert [a["image"] for a in anchors] == [f"h3_frame_ref_{i}.png" for i in range(1, 5)]
    assert anchors[0]["role"] == "semantic_reference"
    assert anchors[0]["frame_index"] is None
    assert anchors[0]["reference_order"] == 1
    guides = [a for a in anchors if a["role"] == "timeline_guide"]
    assert [a["frame_index"] for a in guides] == [36, 72, 120]
    assert [a["guide_length"] for a in guides] == [1, 1, 1]
    assert [a["frame_index"] / 24 for a in guides] == [1.5, 3, 5]
    assert all(0 <= a["frame_index"] < a["frame_index"] + a["guide_length"] <= 124 for a in guides)


def test_native_output_contract_and_fixed_seed():
    output = CONTRACT["output"]
    assert output == {
        "width": 1344,
        "height": 768,
        "aspect_ratio": "16:9",
        "fps": 24,
        "num_frames": 124,
        "audio_required": True,
    }
    assert (output["num_frames"] - 5) % 17 == 0
    generate = next(n for n in NODES.values() if n["type"] == "VLLMOmniGenerateVideo")
    assert generate["widgets_values"][-4:] == [1344, 768, 24, 5.167]
    assert round(generate["widgets_values"][-1] * output["fps"]) == output["num_frames"]
    sampling = next(n for n in NODES.values() if n["type"] == "VLLMOmniDiffusionSampling")
    assert sampling["widgets_values"] == [1, 50, 1, 1, False, False, 738004, "fixed"]


def test_default_reference_and_prompt_tags_match_official_topology():
    reference = next(n for n in NODES.values() if n["type"] == "VLLMOmniVideoReferences")
    connected = [p for p in reference["inputs"] if p["link"] is not None]
    assert [p["name"] for p in connected] == ["image_1"]
    link = next(link for link in GRAPH["links"] if link[0] == connected[0]["link"])
    assert NODES[link[1]]["widgets_values"][0] == "h3_frame_ref_1.png"
    generate = next(n for n in NODES.values() if n["type"] == "VLLMOmniGenerateVideo")
    prompt = generate["widgets_values"][2]
    assert "<Picture 1>" in prompt
    assert all(f"<Picture {i}>" not in prompt for i in range(2, 5))
    assert next(p for p in generate["inputs"] if p["name"] == "frame")["link"] is None
    assert next(p for p in generate["inputs"] if p["name"] == "fast_h3")["link"] is None


def test_links_have_consistent_source_destination_and_types():
    assert len(NODES) == len(GRAPH["nodes"])
    links = {link[0]: link for link in GRAPH["links"]}
    assert len(links) == len(GRAPH["links"])
    assert GRAPH["last_node_id"] == max(NODES)
    assert GRAPH["last_link_id"] == max(links)
    for link_id, source_id, source_slot, target_id, target_slot, kind in links.values():
        source = NODES[source_id]["outputs"][source_slot]
        target = NODES[target_id]["inputs"][target_slot]
        assert source["type"] == target["type"] == kind
        assert link_id in source["links"]
        assert target["link"] == link_id
    for node in NODES.values():
        for slot, port in enumerate(node["inputs"]):
            if port["link"] is not None:
                assert links[port["link"]][3:5] == [node["id"], slot]
        for slot, port in enumerate(node["outputs"]):
            for link_id in port["links"] or []:
                assert links[link_id][1:3] == [node["id"], slot]


@pytest.mark.parametrize(
    "kind",
    ["VLLMOmniVideoReferences", "VLLMOmniDiffusionSampling", "VLLMOmniMiniMaxH3Params", "VLLMOmniGenerateVideo"],
)
def test_draft_uses_existing_node_inputs_and_widget_order(kind):
    schema = _node_schema(kind)
    node = next(n for n in NODES.values() if n["type"] == kind)
    fields = schema.get("required", {}) | schema.get("optional", {})
    assert {p["name"] for p in node["inputs"]} == set(fields)
    assert all(
        p["type"] == ("COMBO" if isinstance(fields[p["name"]][0], list) else fields[p["name"]][0])
        for p in node["inputs"]
    )
    widget_names = [p["widget"]["name"] for p in node["inputs"] if "widget" in p]
    expected_widgets = [
        name
        for name, definition in fields.items()
        if isinstance(definition[0], list) or definition[0] in {"STRING", "INT", "FLOAT", "BOOLEAN"}
    ]
    assert widget_names == expected_widgets
    assert len(node["widgets_values"]) == len(widget_names) + ("seed" in widget_names)


def test_no_local_model_loaders_or_unimplemented_guide_node_types():
    allowed = {
        "LoadImage",
        "SaveVideo",
        "MarkdownNote",
        "VLLMOmniVideoReferences",
        "VLLMOmniGenerateVideo",
        "VLLMOmniDiffusionSampling",
        "VLLMOmniMiniMaxH3Params",
    }
    assert {n["type"] for n in NODES.values()} <= allowed
    guide_notes = [n for n in NODES.values() if n["title"].startswith("GUIDE-02 pending:")]
    assert len(guide_notes) == 3
    assert all(n["type"] == "MarkdownNote" and not n["outputs"] for n in guide_notes)


def test_media_paths_are_portable_and_model_is_a_served_name():
    images = [n for n in NODES.values() if n["type"] == "LoadImage"]
    assert len(images) == 4
    for node in images:
        name = node["widgets_values"][0]
        assert PurePosixPath(name).name == name
        assert not PureWindowsPath(name).is_absolute()
    generate = next(n for n in NODES.values() if n["type"] == "VLLMOmniGenerateVideo")
    assert generate["widgets_values"][0:2] == ["http://localhost:8000/v1", "MiniMaxAI/MiniMax-H3"]
    save = next(n for n in NODES.values() if n["type"] == "SaveVideo")
    assert save["widgets_values"] == ["video/MiniMax_H3_Multiframe", "mp4", "auto"]
