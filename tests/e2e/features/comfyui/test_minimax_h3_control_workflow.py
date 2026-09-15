# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import json
from pathlib import Path

import pytest
from comfyui_vllm_omni.nodes import VLLMOmniGenerateVideo, VLLMOmniMiniMaxH3Params
from comfyui_vllm_omni.utils.models import _minimaxh3_params_builder

from vllm_omni.model_executor.models.minimax_h3.preprocessing import resolve_minimax_h3_aspect_ratio

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

WORKFLOW_PATH = (
    Path(__file__).resolve().parents[4]
    / "apps/ComfyUI-vLLM-Omni/example_workflows/vLLM-Omni MiniMax-H3 Fun ControlNet Union.json"
)


@pytest.fixture(scope="module")
def workflow():
    return json.loads(WORKFLOW_PATH.read_text())


def _node(workflow, node_type):
    return next(node for node in workflow["nodes"] if node["type"] == node_type)


def _source(workflow, node, input_name):
    port = next(port for port in node["inputs"] if port["name"] == input_name)
    link = next(link for link in workflow["links"] if link[0] == port["link"])
    assert link[3:5] == [node["id"], node["inputs"].index(port)]
    source = next(node for node in workflow["nodes"] if node["id"] == link[1])
    assert link[0] in source["outputs"][link[2]]["links"]
    return source, link[2]


def test_default_canny_generation_path(workflow):
    generate = _node(workflow, "VLLMOmniGenerateVideo")
    control = _node(workflow, "VLLMOmniMiniMaxH3Control")
    assert control["widgets_values"] == ["canny", 1.0]
    assert _source(workflow, generate, "control") == (control, 0)
    assert _source(workflow, _node(workflow, "SaveVideo"), "video") == (generate, 0)

    video, slot = _source(workflow, control, "control_video")
    assert (video["type"], slot, video["widgets_values"][0]) == ("CreateVideo", 0, 24)
    canny = _node(workflow, "Canny")
    assert _source(workflow, video, "images") == (canny, 0)
    frames, slot = _source(workflow, canny, "image")
    assert (frames["type"], slot) == ("GetVideoComponents", 0)
    source, slot = _source(workflow, frames, "video")
    assert (source["type"], slot) == ("LoadVideo", 0)


def test_shared_h3_defaults_and_portable_inputs(workflow):
    generate = _node(workflow, "VLLMOmniGenerateVideo")
    fields = VLLMOmniGenerateVideo.INPUT_TYPES()["required"]
    assert len(generate["widgets_values"]) == len(fields)
    values = dict(zip(fields, generate["widgets_values"]))
    assert values["model"] == "MiniMaxAI/MiniMax-H3"
    assert (values["width"], values["height"], values["fps"], values["duration"]) == (1344, 768, 24, 5.167)
    duration = next(port for port in generate["inputs"] if port["name"] == "duration")
    assert duration["widget"]["name"] == "duration"
    assert not any(port["name"] == "num_frames" for port in generate["inputs"])

    sampling, _ = _source(workflow, generate, "sampling_params")
    assert sampling["type"] == "VLLMOmniDiffusionSampling"
    assert sampling["widgets_values"][1:4] == [40, 1.0, 1.0]
    assert sampling["widgets_values"][-2:] == [43, "fixed"]
    params, _ = _source(workflow, generate, "model_params")
    assert params["type"] == "VLLMOmniMiniMaxH3Params"
    assert params["widgets_values"] == [3.0, 12.0]
    assert len(params["widgets_values"]) == len(VLLMOmniMiniMaxH3Params.INPUT_TYPES()["required"])
    assert all(
        port["link"] is None for port in generate["inputs"] if port["name"] in {"frame", "references", "fast_h3"}
    )

    for node in workflow["nodes"]:
        if node["type"] in {"LoadVideo", "LoadImageMask"}:
            filename = Path(node["widgets_values"][0])
            assert not filename.is_absolute()
            assert ".." not in filename.parts
            field = "image" if node["type"] == "LoadImageMask" else "file"
            assert node["widgets_values_named"][field] == str(filename)


def test_pose_subgraph_keeps_model_vae_and_detection_connections(workflow):
    (subgraph,) = workflow["definitions"]["subgraphs"]
    pose = _node(workflow, subgraph["id"])
    assert _source(workflow, pose, "video")[0]["type"] == "LoadVideo"
    pose_video = next(
        node
        for node in workflow["nodes"]
        if node["type"] == "CreateVideo" and _source(workflow, node, "images")[0] == pose
    )
    assert pose_video["widgets_values"][0] == 24
    assert pose["widgets_values"][0] == pose["widgets_values_named"]["resize_type.longer_size"] == 1344

    model = _node(subgraph, "CheckpointLoaderSimple")
    detector = _node(subgraph, "UNETLoader")
    extract = _node(subgraph, "SDPoseKeypointExtractor")
    detect = _node(subgraph, "RTDETR_detect")
    draw = _node(subgraph, "SDPoseDrawKeypoints")
    resize = _node(subgraph, "ResizeImageMaskNode")
    edges = {
        (link["origin_id"], link["origin_slot"], link["target_id"], link["target_slot"]) for link in subgraph["links"]
    }
    assert {
        (model["id"], 0, extract["id"], 0),
        (model["id"], 2, extract["id"], 1),
        (resize["id"], 0, extract["id"], 2),
        (detector["id"], 0, detect["id"], 0),
        (resize["id"], 0, detect["id"], 1),
        (detect["id"], 0, extract["id"], 3),
        (extract["id"], 0, draw["id"], 0),
    } <= edges
    assert model["widgets_values"][0] == "sdpose_wholebody_fp16.safetensors"
    assert detector["widgets_values"][0] == "rt_detr_v4-x-hgnet_fp16.safetensors"


def test_default_path_leaves_alternative_inputs_idle(workflow):
    control = _node(workflow, "VLLMOmniMiniMaxH3Control")
    assert all(
        port["link"] is None for port in control["inputs"] if port["name"] in {"source_video", "mask", "mask_video"}
    )
    mask = _node(workflow, "LoadImageMask")
    assert mask["widgets_values"][1] == "red"
    assert not mask["outputs"][0]["links"]
    for video in (node for node in workflow["nodes"] if node["type"] == "CreateVideo"):
        assert all(port["link"] is None for port in video["inputs"] if port["name"] == "audio")
    assert len([node for node in workflow["nodes"] if node["type"] == "SaveVideo"]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("controlled", [False, True])
async def test_workflow_duration_and_dimensions_reach_the_h3_request(workflow, controlled, mocker):
    node = _node(workflow, "VLLMOmniMiniMaxH3Params")
    fields = VLLMOmniMiniMaxH3Params.INPUT_TYPES()["required"]
    (params,) = VLLMOmniMiniMaxH3Params().get_params(**dict(zip(fields, node["widgets_values"])))
    generate = _node(workflow, "VLLMOmniGenerateVideo")
    required = VLLMOmniGenerateVideo.INPUT_TYPES()["required"]
    values = dict(zip(required, generate["widgets_values"]))
    control = {"control_type": "canny", "control_context_scale": 1.0} if controlled else None
    send = mocker.patch("comfyui_vllm_omni.nodes.VLLMOmniClient.generate_video", new_callable=mocker.AsyncMock)

    await VLLMOmniGenerateVideo().generate(**values, model_params=params, control=control)

    send.assert_awaited_once()
    request = send.await_args.kwargs
    assert (request["width"], request["height"], request["fps"], request["num_frames"]) == (1344, 768, 24, 124)
    assert (request["num_frames"] - 5) % 17 == 0
    assert request["control"] == control
    extra: dict[str, object] = {"task": "t2va"}
    if controlled:
        extra["canny"] = {"control_context_scale": 1.0}
    wire = _minimaxh3_params_builder(
        request["model_params"], extra_params=extra, width=request["width"], height=request["height"]
    )
    received = json.loads(wire["extra_params"])
    assert received["aspect_ratio"] == "16:9"
    assert received["task"] == "t2va"
    assert resolve_minimax_h3_aspect_ratio("t2va", received["aspect_ratio"], None) == pytest.approx(16 / 9)
