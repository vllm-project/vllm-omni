# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""WF-04: the MiniMax-H3 Multiframe Reference template.

Checks the saved graph against the node schemas, then executes it on CPU with
the real vLLM-Omni nodes and asserts the ``/v1/videos`` request it produces:
one Ref2VA image reference plus three ordered still-image timeline guides.
"""

import inspect
import json
import re
from graphlib import TopologicalSorter
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock

import numpy as np
import pytest
import torch
from comfyui_vllm_omni import nodes as omni_nodes
from comfyui_vllm_omni.utils import api_client
from PIL import Image

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

WORKFLOW = (
    Path(__file__).resolve().parents[4]
    / "apps/ComfyUI-vLLM-Omni/example_workflows/vLLM-Omni MiniMax-H3 Multiframe Reference.json"
)
REFERENCE = "h3_frame_ref_1.png"
GUIDES = [(36, "h3_frame_ref_2.png"), (72, "h3_frame_ref_3.png"), (120, "h3_frame_ref_4.png")]


@pytest.fixture
def workflow():
    return json.loads(WORKFLOW.read_text())


def _nodes(workflow):
    return {node["id"]: node for node in workflow["nodes"]}


def _source(workflow, node, input_name):
    links = {link[0]: link for link in workflow["links"]}
    link_id = next(input_["link"] for input_ in node["inputs"] if input_["name"] == input_name)
    assert link_id is not None, f"{node['type']}.{input_name} is not connected"
    return _nodes(workflow)[links[link_id][1]], links[link_id][2]


def _linked(node):
    return {input_["name"] for input_ in node["inputs"] if input_["link"] is not None}


def _only(workflow, node_type):
    (node,) = [node for node in workflow["nodes"] if node["type"] == node_type]
    return node


def _guide_chain(workflow):
    chain = []
    guide, _ = _source(workflow, _only(workflow, "VLLMOmniGenerateVideo"), "timeline_guides")
    while True:
        assert guide["type"] == "VLLMOmniMiniMaxH3Guide"
        chain.append(guide)
        if "timeline_guides" not in _linked(guide):
            return chain[::-1]
        guide, _ = _source(workflow, guide, "timeline_guides")


def test_multiframe_workflow_connections(workflow):
    nodes = _nodes(workflow)
    links = {link[0]: link for link in workflow["links"]}
    assert len(nodes) == len(workflow["nodes"]) and len(links) == len(workflow["links"])
    assert workflow["last_node_id"] == max(nodes) and workflow["last_link_id"] == max(links)
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
        assert node["mode"] == 0, f"{node['type']} {node['id']} is muted or bypassed"
        for slot, input_ in enumerate(node.get("inputs", [])):
            if input_["link"] is not None:
                assert links[input_["link"]][3:5] == [node["id"], slot]
        for output in node.get("outputs", []):
            for link_id in output.get("links") or []:
                assert links[link_id][1] == node["id"]


def test_multiframe_workflow_matches_omni_node_interfaces(workflow):
    for node in workflow["nodes"]:
        if not node["type"].startswith("VLLMOmni"):
            continue
        cls = getattr(omni_nodes, node["type"])
        schema = cls.INPUT_TYPES()
        inputs = {**schema.get("required", {}), **schema.get("optional", {})}
        # Every port is saved, so links keep their slots when the graph is loaded.
        assert sorted(input_["name"] for input_ in node["inputs"]) == sorted(inputs)
        for input_ in node["inputs"]:
            kind = inputs[input_["name"]][0]
            assert input_["type"] == ("COMBO" if isinstance(kind, list) else kind)
        assert tuple(output["type"] for output in node["outputs"]) == cls.RETURN_TYPES
        expected_widgets = len(schema.get("required", {})) + ("seed" in schema.get("required", {}))
        assert len(node["widgets_values"]) == expected_widgets


def test_multiframe_workflow_separates_reference_from_guides(workflow):
    generate = _only(workflow, "VLLMOmniGenerateVideo")
    references, _ = _source(workflow, generate, "references")
    assert references["type"] == "VLLMOmniVideoReferences"
    assert _linked(references) == {"image_1"}
    loader, slot = _source(workflow, references, "image_1")
    assert (loader["type"], loader["outputs"][slot]["name"], loader["widgets_values"][0]) == (
        "LoadImage",
        "IMAGE",
        REFERENCE,
    )

    chain = _guide_chain(workflow)
    assert [(guide["widgets_values"][0], _linked(guide) - {"timeline_guides"}) for guide in chain] == [
        (frame, {"image"}) for frame, _ in GUIDES
    ]
    for guide, (_, filename) in zip(chain, GUIDES, strict=True):
        loader, _ = _source(workflow, guide, "image")
        assert (loader["type"], loader["widgets_values"][0]) == ("LoadImage", filename)
    # Every anchor image feeds exactly one role.
    assert sorted(node["widgets_values"][0] for node in workflow["nodes"] if node["type"] == "LoadImage") == sorted(
        [REFERENCE, *(filename for _, filename in GUIDES)]
    )


def test_multiframe_workflow_defaults_and_portable_assets(workflow):
    generate = _only(workflow, "VLLMOmniGenerateVideo")
    values = generate["widgets_values"]
    assert values[:2] == ["http://localhost:8000/v1", "MiniMaxAI/MiniMax-H3"]
    assert values[3] == ""
    assert values[4:8] == [1344, 768, 24, 5.167]
    num_frames = round(values[6] * values[7])
    assert num_frames == 124 and (num_frames - 5) % 17 == 0
    assert all(0 < frame < num_frames for frame, _ in GUIDES)
    assert _linked(generate) == {"references", "sampling_params", "model_params", "timeline_guides"}
    # Guided requests need base weights: no LoRA, FastH3 or latent-edit nodes.
    types = {node["type"] for node in workflow["nodes"]}
    assert not types & {"VLLMOmniRemoteLoRA", "VLLMOmniFastH3Deployment", "VLLMOmniLatentMaskEditing"}

    # Guides never enter the Qwen presentation, so only the reference has a tag.
    assert set(re.findall(r"<Picture (\d+)>", values[2])) == {"1"}
    assert not re.search(r"<(Video|Audio) \d+>", values[2])

    assert _only(workflow, "VLLMOmniDiffusionSampling")["widgets_values"] == [
        1,
        50,
        1,
        1,
        False,
        False,
        738004,
        "fixed",
    ]
    assert _only(workflow, "VLLMOmniMiniMaxH3Params")["widgets_values"] == [3.0, 12.0, "native"]
    save = _only(workflow, "SaveVideo")
    assert _source(workflow, save, "video")[0] is generate
    assert save["widgets_values"] == ["video/MiniMax-H3-Multiframe-Reference", "mp4", "h264"]
    for node in workflow["nodes"]:
        if node["type"] == "LoadImage":
            filename = node["widgets_values"][0]
            assert filename and Path(filename).name == filename
            assert ":" not in filename and "\\" not in filename


def _loaded_image(filename: str) -> torch.Tensor:
    """A distinct 1x32x64x3 image per anchor, keyed by the loader filename."""
    match = re.search(r"(\d+)\.png$", filename)
    assert match, filename
    value = int(match.group(1)) * 50
    return torch.full((1, 32, 64, 3), value / 255.0)


async def _execute(workflow) -> None:
    """Run the saved UI graph with the real vLLM-Omni nodes, the way ComfyUI resolves its API prompt."""
    nodes = _nodes(workflow)
    links = {link[0]: link for link in workflow["links"]}
    results: dict[int, tuple] = {}

    async def run(node_id: int) -> tuple:
        if node_id in results:
            return results[node_id]
        node = nodes[node_id]
        if node["type"] == "LoadImage":
            results[node_id] = (_loaded_image(node["widgets_values"][0]), None)
            return results[node_id]
        widgets = [input_["name"] for input_ in node["inputs"] if "widget" in input_]
        values = node["widgets_values"][: len(widgets)]
        kwargs = dict(zip(widgets, values, strict=True))
        for input_ in node["inputs"]:
            if input_["link"] is not None:
                _, source, slot, *_ = links[input_["link"]]
                kwargs[input_["name"]] = (await run(source))[slot]
        cls = getattr(omni_nodes, node["type"])
        output = getattr(cls(), cls.FUNCTION)(**kwargs)
        results[node_id] = await output if inspect.isawaitable(output) else output
        return results[node_id]

    await run(_only(workflow, "VLLMOmniGenerateVideo")["id"])


async def test_multiframe_workflow_sends_ref2va_with_ordered_guides(workflow, monkeypatch):
    calls = AsyncMock(return_value={"id": "test-video", "status": "completed"})
    monkeypatch.setattr(api_client, "url_json", calls)
    monkeypatch.setattr(api_client, "url_bytes", AsyncMock(return_value=b"generated-video"))
    monkeypatch.setattr(api_client, "bytes_to_video", lambda data: data)

    await _execute(workflow)

    _, url, verb = calls.call_args_list[0].args
    assert (url, verb) == ("http://localhost:8000/v1/videos", "post")
    fields = [
        (options["name"], options.get("filename"), value)
        for options, _, value in calls.call_args_list[0].kwargs["data"]._fields
    ]
    scalars = {name: value for name, filename, value in fields if filename is None}
    assert {key: scalars[key] for key in ("model", "width", "height", "num_frames", "fps")} == {
        "model": "MiniMaxAI/MiniMax-H3",
        "width": "1344",
        "height": "768",
        "num_frames": "124",
        "fps": "24",
    }
    assert {key: scalars[key] for key in ("num_inference_steps", "seed", "quality", "flow_shift")} == {
        "num_inference_steps": "50",
        "seed": "738004",
        "quality": "lossless",
        "flow_shift": "12.0",
    }
    assert json.loads(scalars["extra_params"]) == {"task": "ref2va", "audio_flow_shift": 3.0}
    assert "lora" not in scalars and "negative_prompt" not in scalars

    def pixels(value: BytesIO) -> int:
        return int(np.asarray(Image.open(BytesIO(value.getvalue())))[0, 0, 0])

    references = [(filename, pixels(value)) for name, filename, value in fields if name == "input_references"]
    assert references == [("image_1.png", 50)]
    assert json.loads(scalars["timeline_guides"]) == [
        {"frame_index": frame, "image": {"upload_index": index}} for index, (frame, _) in enumerate(GUIDES)
    ]
    guides = [(filename, pixels(value)) for name, filename, value in fields if name == "guide_files"]
    assert guides == [(f"guide_{index}_image.png", 50 * (index + 2)) for index in range(len(GUIDES))]
