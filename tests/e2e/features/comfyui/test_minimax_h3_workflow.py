# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Exercise the shipped WF-01 graph through the real nodes and request serializer."""

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from comfyui_vllm_omni import nodes
from comfyui_vllm_omni.utils import api_client

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

WORKFLOW = (
    Path(__file__).resolve().parents[4] / "apps/ComfyUI-vLLM-Omni/example_workflows/MiniMax_H3_Text_to_Video.json"
)


def load_nodes():
    return {node["id"]: node for node in json.loads(WORKFLOW.read_text())["nodes"]}


def widget_inputs(node):
    """Read widgets in the registered node's order, including optional widgets."""
    schema = getattr(nodes, node["type"]).INPUT_TYPES()
    values = iter(node["widgets_values"])
    result = {}
    for group in ("required", "optional"):
        for name, definition in schema.get(group, {}).items():
            kind = definition[0]
            if isinstance(kind, list) or kind in ("STRING", "INT", "FLOAT", "BOOLEAN"):
                result[name] = next(values)
                if name == "seed":
                    assert next(values) == "fixed"
    assert next(values, None) is None, "Unmapped widget values in exported template"
    return result


def test_workflow_links_and_native_defaults():
    workflow = json.loads(WORKFLOW.read_text())
    graph = load_nodes()
    for link_id, source, source_slot, target, target_slot, kind in workflow["links"]:
        output = graph[source]["outputs"][source_slot]
        input_ = graph[target]["inputs"][target_slot]
        assert link_id in output["links"]
        assert input_["link"] == link_id
        assert output["type"] == input_["type"] == kind
    generate = widget_inputs(graph[3])
    assert (generate["width"], generate["height"], generate["fps"]) == (1344, 768, 24)
    assert generate["duration"] == 5.167
    num_frames = round(generate["duration"] * generate["fps"])
    assert num_frames == 124
    assert (num_frames - 5) % 17 == 0
    assert 4 <= generate["duration"] <= 15
    assert all(input_["name"] != "num_frames" for input_ in graph[3]["inputs"])
    assert all(
        input_["link"] is None
        for input_ in graph[3]["inputs"]
        if input_["name"] in ("frame", "references", "lora", "fast_h3")
    )
    assert graph[4]["type"] == "SaveVideo"
    assert graph[4]["widgets_values"][1] == "mp4"
    assert widget_inputs(graph[5]) == {"local_path": "", "name": "h3-turbo-v1.0-768p", "scale": 1.0, "int_id": 0}
    for node in graph.values():
        if node["type"].startswith("VLLMOmni"):
            widget_inputs(node)


@pytest.mark.asyncio
@pytest.mark.parametrize("turbo", [False, True], ids=["base", "turbo-v1.0-768p"])
async def test_workflow_serializes_t2va_request(monkeypatch, turbo):
    workflow = json.loads(WORKFLOW.read_text())
    graph = load_nodes()
    links = {link[0]: link for link in workflow["links"]}
    connected = {
        input_["name"]: links[input_["link"]][1] for input_ in graph[3]["inputs"] if input_["link"] is not None
    }
    # Execute the base connections actually shipped in the graph. Turbo uses
    # the optional presets that users reconnect as documented.
    sampling_id, params_id = (6, 7) if turbo else (connected["sampling_params"], connected["model_params"])
    sampling = nodes.VLLMOmniDiffusionSampling().get_params(**widget_inputs(graph[sampling_id]))[0]
    params = nodes.VLLMOmniMiniMaxH3Params().get_params(**widget_inputs(graph[params_id]))[0]
    kwargs = widget_inputs(graph[3])
    kwargs.update(sampling_params=sampling, model_params=params)
    if turbo:
        lora = widget_inputs(graph[5])
        assert nodes.VLLMOmniRemoteLoRA.VALIDATE_INPUTS(lora["local_path"], lora["name"]) is True
        kwargs["lora"] = nodes.VLLMOmniRemoteLoRA().get_lora(**lora)[0]

    captured = {}

    async def request(session, url, verb="get", **request_kwargs):
        if verb == "post":
            assert url == "http://localhost:8000/v1/videos"
            # Inspect the FormData emitted by the real client before HTTP encoding.
            captured.update({options["name"]: value for options, _, value in request_kwargs["data"]._fields})
            return {"id": "wf01-test", "status": "completed"}
        assert verb == "delete"
        return {}

    monkeypatch.setattr(api_client, "url_json", request)
    monkeypatch.setattr(api_client, "url_bytes", AsyncMock(return_value=b"video-content"))
    decoded = object()
    monkeypatch.setattr(api_client, "bytes_to_video", lambda data: decoded)
    assert await nodes.VLLMOmniGenerateVideo().generate(**kwargs) == (decoded,)
    assert captured["model"] == "MiniMaxAI/MiniMax-H3"
    assert "aspect_ratio" not in captured
    assert captured["num_frames"] == "124"
    assert captured["fps"] == "24"
    assert captured["num_inference_steps"] == ("5" if turbo else "50")
    assert captured["flow_shift"] == ("6.0" if turbo else "12.0")
    assert captured["seed"] == "1101"
    assert json.loads(captured["extra_params"]) == {"task": "t2va", "audio_flow_shift": 3.0, "aspect_ratio": "16:9"}
    assert "input_reference" not in captured
    if turbo:
        assert json.loads(captured["lora"]) == {"name": "h3-turbo-v1.0-768p", "scale": 1.0, "int_id": None}
    else:
        assert "lora" not in captured


@pytest.mark.parametrize("aspect_ratio", ["16:9", "21:9", "4:3", "1:1", "3:4", "9:16"])
def test_legacy_explicit_h3_aspect_ratio_overrides_dimension_inference(aspect_ratio):
    from comfyui_vllm_omni.utils.models import _minimaxh3_params_builder

    params = nodes.VLLMOmniMiniMaxH3Params().get_params(
        flow_shift=12.0, audio_flow_shift=3.0, aspect_ratio=aspect_ratio
    )[0]
    fields = _minimaxh3_params_builder(params, extra_params={"task": "t2va"}, width=1344, height=768)
    assert "aspect_ratio" not in fields
    assert json.loads(fields["extra_params"])["aspect_ratio"] == aspect_ratio
    assert params["aspect_ratio"] == aspect_ratio


@pytest.mark.parametrize(
    ("width", "height", "aspect_ratio"),
    [(1344, 768, "16:9"), (21, 9, "21:9"), (4, 3, "4:3"), (1, 1, "1:1"), (3, 4, "3:4"), (768, 1344, "9:16")],
)
def test_h3_params_infer_supported_aspect_ratio_from_dimensions(width, height, aspect_ratio):
    from comfyui_vllm_omni.utils.models import _minimaxh3_params_builder

    params = nodes.VLLMOmniMiniMaxH3Params().get_params(flow_shift=12.0, audio_flow_shift=3.0)[0]
    assert "aspect_ratio" not in params
    fields = _minimaxh3_params_builder(params, extra_params={"task": "t2va"}, width=width, height=height)
    assert "aspect_ratio" not in fields
    assert json.loads(fields["extra_params"])["aspect_ratio"] == aspect_ratio


@pytest.mark.parametrize("int_id, expected_id", [(0, None), (10, 10)])
def test_remote_lora_preserves_legacy_widget_order_and_explicit_path(int_id, expected_id):
    # ComfyUI restores saved widget values positionally, including the path first.
    legacy_node = {
        "type": "VLLMOmniRemoteLoRA",
        "widgets_values": [" /server/models/legacy.safetensors ", " legacy ", 0.7, int_id],
    }
    values = widget_inputs(legacy_node)
    assert list(values) == ["local_path", "name", "scale", "int_id"]
    assert nodes.VLLMOmniRemoteLoRA.VALIDATE_INPUTS(values["local_path"], values["name"]) is True
    assert nodes.VLLMOmniRemoteLoRA().get_lora(**values)[0] == {
        "local_path": "/server/models/legacy.safetensors",
        "name": "legacy",
        "scale": 0.7,
        "int_id": expected_id,
    }


def test_remote_lora_omits_blank_path_for_registered_name():
    assert nodes.VLLMOmniRemoteLoRA.VALIDATE_INPUTS("  ", " registered ") is True
    assert nodes.VLLMOmniRemoteLoRA().get_lora("  ", " registered ", 0.5, 0)[0] == {
        "name": "registered",
        "scale": 0.5,
        "int_id": None,
    }


@pytest.mark.parametrize("local_path", ["", "/server/models/legacy.safetensors"])
@pytest.mark.parametrize("name", ["", "  "])
def test_remote_lora_requires_name_with_or_without_path(local_path, name):
    assert nodes.VLLMOmniRemoteLoRA.VALIDATE_INPUTS(local_path, name) == "LoRA name must be provided."
