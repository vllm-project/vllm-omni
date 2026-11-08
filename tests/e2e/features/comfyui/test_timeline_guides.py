# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""MiniMax-H3 timeline guides in the ComfyUI integration.

Covers the **MiniMax H3 Add Guide** node, the ``timeline_guides`` +
``guide_files`` multipart contract that ``VLLMOmniClient.generate_video`` sends
to ``/v1/videos``, error propagation from a real HTTP server, and decoding of
really encoded PNG/MP4/FLAC payloads. Runs on CPU with the ``comfy_api`` mocks
from this directory's ``conftest.py``.
"""

import importlib.util
import json
import math
import sys
from fractions import Fraction
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import av
import numpy as np
import pytest
import torch
from aiohttp import test_utils, web
from comfy_api.input import VideoInput
from comfyui_vllm_omni import nodes as omni_nodes
from comfyui_vllm_omni.nodes import (
    VLLMOmniGenerateVideo,
    VLLMOmniLatentMaskEditing,
    VLLMOmniMiniMaxH3Guide,
    VLLMOmniVideoReferences,
)
from comfyui_vllm_omni.utils import api_client
from comfyui_vllm_omni.utils import format as omni_format
from comfyui_vllm_omni.utils.types import DiffusionSamplingParams, H3TimelineGuides
from PIL import Image

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

PLUGIN_DIR = Path(__file__).resolve().parents[4] / "apps" / "ComfyUI-vLLM-Omni"
GUIDE_WORKFLOW = PLUGIN_DIR / "example_workflows" / "vLLM-Omni MiniMax-H3 Timeline Guides.json"
SCHEMA_COMPLETE_WORKFLOWS = {
    GUIDE_WORKFLOW,
    PLUGIN_DIR / "example_workflows" / "vLLM-Omni MiniMax-H3 All-in-One.json",
    PLUGIN_DIR / "example_workflows" / "vLLM-Omni MiniMax-H3 Music Video.json",
}
H3 = "MiniMaxAI/MiniMax-H3"
WIDTH, HEIGHT = 64, 32


# ---------------------------------------------------------------- media helpers


def _image(value: float = 0.5, batch: int = 1) -> torch.Tensor:
    return torch.full((batch, HEIGHT, WIDTH, 3), value)


def _audio(channels: int = 2, samples: int = 4800, sample_rate: int = 24000, batch: int = 1) -> dict:
    t = torch.arange(samples) / sample_rate
    tones = torch.stack([0.5 * torch.sin(2 * math.pi * 440 * t), 0.25 * torch.sin(2 * math.pi * 660 * t)])
    if channels > 2:
        tones = tones.repeat(math.ceil(channels / 2), 1)
    return {"waveform": tones[:channels].unsqueeze(0).repeat(batch, 1, 1), "sample_rate": sample_rate}


def _mp4(frames: int = 22, fps: int = 24, audio_rate: int | None = None) -> bytes:
    """Encode a real H.264 MP4 whose frame ``i`` is filled with ``10 * i``."""
    buffer = BytesIO()
    with av.open(buffer, mode="w", format="mp4") as container:
        video = container.add_stream("libx264", rate=fps)
        video.width, video.height, video.pix_fmt = WIDTH, HEIGHT, "yuv420p"
        video.options = {"crf": "0", "preset": "ultrafast"}
        audio = container.add_stream("aac", rate=audio_rate, layout="stereo") if audio_rate else None
        for index in range(frames):
            frame = av.VideoFrame.from_ndarray(np.full((HEIGHT, WIDTH, 3), 10 * index % 256, np.uint8), format="rgb24")
            frame.pts = index
            frame.time_base = Fraction(1, fps)
            container.mux(video.encode(frame))
        container.mux(video.encode())
        if audio is not None and audio_rate:
            t = np.arange(round(frames / fps * audio_rate)) / audio_rate
            tones = np.stack([0.5 * np.sin(2 * np.pi * 440 * t), 0.25 * np.sin(2 * np.pi * 660 * t)])
            samples = av.AudioFrame.from_ndarray(tones.astype(np.float32), format="fltp", layout="stereo")
            samples.sample_rate = audio_rate
            samples.pts = 0
            container.mux(audio.encode(samples))
            container.mux(audio.encode())
    return buffer.getvalue()


def _decode_video(data: bytes) -> tuple[list[np.ndarray], Fraction, str]:
    with av.open(BytesIO(data)) as container:
        stream = container.streams.video[0]
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]
        return frames, Fraction(stream.average_rate), stream.codec_context.name


def _decode_audio(data: bytes) -> tuple[np.ndarray, int, str]:
    with av.open(BytesIO(data)) as container:
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="fltp")
        chunks = [
            out.to_ndarray()
            for frame in [*container.decode(stream), None]
            for out in resampler.resample(frame)
            if out is not None
        ]
        return np.concatenate(chunks, axis=1), stream.sample_rate, stream.codec_context.name


# ------------------------------------------------------------------ node helpers


def _chain(*guides: tuple[int, dict], upstream=None) -> H3TimelineGuides:
    node = VLLMOmniMiniMaxH3Guide()
    chain = upstream
    for frame_index, media in guides:
        (chain,) = node.add_guide(frame_index=frame_index, timeline_guides=chain, **media)
    return chain


# --------------------------------------------------------------- client helpers


@pytest.fixture
def api_calls(monkeypatch):
    calls = AsyncMock(return_value={"id": "test-video", "status": "completed"})
    monkeypatch.setattr(api_client, "url_json", calls)
    monkeypatch.setattr(api_client, "url_bytes", AsyncMock(return_value=b"generated-video"))
    monkeypatch.setattr(api_client, "bytes_to_video", lambda data: data)
    return calls


@pytest.fixture
def encoders(monkeypatch):
    """Record every media encoding the client performs."""
    calls: list[str] = []
    for name in ("image_tensor_to_png_bytes", "video_to_bytes", "audio_to_bytes"):

        def spy(*args, _original=getattr(api_client, name), _name=name, **kwargs):
            calls.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(api_client, name, spy)
    return calls


async def _generate(timeline_guides, **kwargs):
    params = {
        "model": H3,
        "prompt": "Follow the guides.",
        "width": WIDTH,
        "height": HEIGHT,
        "num_frames": 124,
        "fps": 24,
        **kwargs,
    }
    return await api_client.VLLMOmniClient("http://localhost/v1").generate_video(
        timeline_guides=timeline_guides, **params
    )


def _fields(calls) -> list[tuple[str, str | None, str | None, Any]]:
    session, url, verb = calls.call_args_list[0].args
    assert (url, verb) == ("http://localhost/v1/videos", "post")
    return [
        (options["name"], options.get("filename"), headers.get("Content-Type"), value)
        for options, headers, value in calls.call_args_list[0].kwargs["data"]._fields
    ]


def _value(calls, name: str):
    (value,) = [value for field, _, _, value in _fields(calls) if field == name]
    return value


def _uploads(calls, name: str) -> list[tuple[str | None, str | None, bytes]]:
    return [
        (filename, content_type, value.getvalue())
        for field, filename, content_type, value in _fields(calls)
        if field == name
    ]


# ======================================================================== node


def test_guide_node_interface():
    schema = VLLMOmniMiniMaxH3Guide.INPUT_TYPES()
    assert list(schema["required"]) == ["frame_index"]
    kind, options = schema["required"]["frame_index"]
    assert kind == "INT"
    assert options["default"] == 0 and options["step"] == 1 and options["min"] < -1 < 0 < options["max"]
    assert {name: spec[0] for name, spec in schema["optional"].items()} == {
        "image": "IMAGE",
        "video": "VIDEO",
        "audio": "AUDIO",
        "timeline_guides": "H3_TIMELINE_GUIDES",
    }
    assert VLLMOmniMiniMaxH3Guide.RETURN_TYPES == ("H3_TIMELINE_GUIDES",)
    assert VLLMOmniMiniMaxH3Guide.RETURN_NAMES == ("timeline_guides",)
    assert VLLMOmniMiniMaxH3Guide.FUNCTION == "add_guide"
    assert hasattr(VLLMOmniMiniMaxH3Guide, "add_guide")


def test_plugin_registers_guide_node(monkeypatch):
    name = "comfyui_vllm_omni_plugin_under_test"
    spec = importlib.util.spec_from_file_location(
        name, PLUGIN_DIR / "__init__.py", submodule_search_locations=[str(PLUGIN_DIR)]
    )
    assert spec is not None and spec.loader is not None
    plugin = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, plugin)
    try:
        spec.loader.exec_module(plugin)
        cls = plugin.NODE_CLASS_MAPPINGS["VLLMOmniMiniMaxH3Guide"]
        assert cls.__name__ == "VLLMOmniMiniMaxH3Guide"
        assert cls.RETURN_TYPES == ("H3_TIMELINE_GUIDES",)
        assert plugin.NODE_DISPLAY_NAME_MAPPINGS["VLLMOmniMiniMaxH3Guide"] == "MiniMax H3 Add Guide"
        assert plugin.NODE_CLASS_MAPPINGS.keys() == plugin.NODE_DISPLAY_NAME_MAPPINGS.keys()
    finally:
        for module in [module for module in sys.modules if module.startswith(f"{name}.")]:
            del sys.modules[module]


def test_generate_video_appends_timeline_guides_port_last():
    schema = VLLMOmniGenerateVideo.INPUT_TYPES()
    assert list(schema["required"]) == [
        "url",
        "model",
        "prompt",
        "negative_prompt",
        "width",
        "height",
        "fps",
        "duration",
    ]
    assert list(schema["optional"]) == [
        "frame",
        "first_frame",
        "last_frame",
        "references",
        "sampling_params",
        "lora",
        "model_params",
        "fast_h3",
        "latent_edit",
        "timeline_guides",
    ]
    assert schema["optional"]["timeline_guides"] == ("H3_TIMELINE_GUIDES",)
    assert VLLMOmniGenerateVideo.RETURN_TYPES == ("VIDEO",)


@pytest.mark.parametrize("path", sorted((PLUGIN_DIR / "example_workflows").glob("*.json")), ids=lambda p: p.stem)
def test_saved_workflows_keep_generate_video_slots(path):
    """Legacy graphs may omit the port; schema-complete graphs save it in place."""
    workflow = json.loads(path.read_text())
    schema = VLLMOmniGenerateVideo.INPUT_TYPES()
    current = {**schema["required"], **schema["optional"]}
    order = list(schema["optional"])
    for node in workflow.get("nodes", []):
        if node["type"] != "VLLMOmniGenerateVideo":
            continue
        names = [input_["name"] for input_ in node.get("inputs", [])]
        sockets = [name for name in names if name in schema["optional"]]
        assert sockets == sorted(sockets, key=order.index)
        assert ("timeline_guides" in names) == (path in SCHEMA_COMPLETE_WORKFLOWS)
        for input_ in node.get("inputs", []):
            assert input_["type"] == current[input_["name"]][0]
        assert len(node["widgets_values"]) == len(schema["required"])


@pytest.mark.parametrize(
    "media",
    [
        pytest.param({"image": "image"}, id="image"),
        pytest.param({"video": "video"}, id="clip"),
        pytest.param({"audio": "audio"}, id="audio-only"),
        pytest.param({"image": "image", "audio": "audio"}, id="image+audio"),
        pytest.param({"video": "video", "audio": "audio"}, id="video+audio"),
    ],
)
@pytest.mark.parametrize("frame_index", [0, 36, -1, -22])
def test_guide_node_accepts_supported_media(media, frame_index):
    values = {"image": _image(), "video": VideoInput(b"clip"), "audio": _audio()}
    inputs = {kind: values[kind] for kind in media}
    chain = _chain((frame_index, inputs))
    assert isinstance(chain, H3TimelineGuides)
    assert chain == [{"frame_index": frame_index, **inputs}]
    assert all(chain[0][kind] is value for kind, value in inputs.items())


def test_guide_chain_copies_upstream_and_keeps_media_objects():
    image, clip, audio = _image(), VideoInput(b"clip"), _audio()
    first = _chain((36, {"image": image}))
    first_snapshot = [dict(entry) for entry in first]
    second = _chain((-22, {"video": clip, "audio": audio}), upstream=first)

    assert first == first_snapshot and len(first) == 1
    assert second is not first and second[0] is not first[0]
    assert [entry["frame_index"] for entry in second] == [36, -22]
    assert second[0]["image"] is image and second[1]["video"] is clip and second[1]["audio"] is audio
    assert audio["waveform"].shape == (1, 2, 4800)  # upstream media untouched


def test_forked_guide_chains_do_not_share_entries():
    root = _chain((0, {"image": _image(0.1)}))
    left = _chain((36, {"image": _image(0.2)}), upstream=root)
    right = _chain((72, {"audio": _audio()}), upstream=root)

    assert [entry["frame_index"] for entry in root] == [0]
    assert [entry["frame_index"] for entry in left] == [0, 36]
    assert [entry["frame_index"] for entry in right] == [0, 72]
    left[0]["frame_index"] = 99
    assert root[0]["frame_index"] == 0 and right[0]["frame_index"] == 0


def test_duplicate_overlapping_and_unordered_guides_keep_chain_order():
    image = _image()
    chain = _chain(
        (72, {"image": image}),
        (36, {"image": image}),
        (36, {"image": image}),
        (-1, {"audio": _audio()}),
        (0, {"image": image}),
    )
    assert [entry["frame_index"] for entry in chain] == [72, 36, 36, -1, 0]
    assert len({id(entry) for entry in chain}) == 5
    assert all(entry["image"] is image for entry in chain if "image" in entry)


def test_api_prompt_round_trip_keeps_chain_order_and_widget_values():
    """Exported API prompts resolve links into the same chain after a JSON round trip."""
    media = {"img2": _image(0.2), "img3": _image(0.3), "img4": _image(0.4), "clip": VideoInput(b"c"), "aud": _audio()}
    prompt = {
        "21": {"class_type": "VLLMOmniMiniMaxH3Guide", "inputs": {"frame_index": 36, "image": ["img2", 0]}},
        "22": {
            "class_type": "VLLMOmniMiniMaxH3Guide",
            "inputs": {"frame_index": 72, "image": ["img3", 0], "timeline_guides": ["21", 0]},
        },
        "23": {
            "class_type": "VLLMOmniMiniMaxH3Guide",
            "inputs": {"frame_index": 120, "image": ["img4", 0], "timeline_guides": ["22", 0]},
        },
        "24": {
            "class_type": "VLLMOmniMiniMaxH3Guide",
            "inputs": {"frame_index": -22, "video": ["clip", 0], "audio": ["aud", 0], "timeline_guides": ["23", 0]},
        },
    }
    restored = json.loads(json.dumps(prompt))
    assert restored == prompt
    results: dict[str, tuple] = {}

    def run(node_id: str) -> tuple:
        if node_id in media:
            return (media[node_id],)
        if node_id not in results:
            node = restored[node_id]
            kwargs = {
                name: run(value[0])[value[1]] if isinstance(value, list) else value
                for name, value in node["inputs"].items()
            }
            results[node_id] = getattr(omni_nodes, node["class_type"])().add_guide(**kwargs)
        return results[node_id]

    (chain,) = run("24")
    assert [entry["frame_index"] for entry in chain] == [36, 72, 120, -22]
    assert all(chain[i]["image"] is media[key] for i, key in enumerate(("img2", "img3", "img4")))
    assert chain[3]["video"] is media["clip"] and chain[3]["audio"] is media["aud"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({}, "needs an image, video, or audio"),
        ({"image": _image(), "video": VideoInput(b"clip")}, "cannot combine image and video"),
        ({"frame_index": 1.0, "image": _image()}, "frame_index must be an integer"),
        ({"frame_index": "36", "image": _image()}, "frame_index must be an integer"),
        ({"frame_index": True, "image": _image()}, "frame_index must be an integer"),
        ({"frame_index": None, "image": _image()}, "frame_index must be an integer"),
        ({"image": _image(batch=2)}, "batch of 2 images.*Create Video at 24 FPS"),
        ({"image": torch.zeros(HEIGHT, WIDTH, 3)}, "IMAGE shaped"),
        ({"video": b"not-a-video"}, "must be a ComfyUI VIDEO"),
        ({"audio": _audio(batch=2)}, "batch of 2 clips"),
        ({"audio": _audio(channels=6)}, "6 channels"),
        ({"audio": _audio(samples=0)}, "audio is empty"),
        ({"audio": {**_audio(), "sample_rate": 0}}, "positive integer sample_rate"),
        ({"audio": {"waveform": torch.full((1, 1, 8), float("nan")), "sample_rate": 8}}, "NaN or infinite"),
        ({"image": _image(), "timeline_guides": {"frame_index": 0}}, "must come from another"),
        ({"image": _image(), "timeline_guides": ["guide"]}, "must come from another"),
    ],
)
def test_guide_node_rejects_invalid_guides(kwargs, message):
    kwargs = {"frame_index": 0, **kwargs}
    with pytest.raises(ValueError, match=message):
        VLLMOmniMiniMaxH3Guide().add_guide(**kwargs)


def test_guide_node_rejects_a_ninth_guide():
    chain = _chain(*[(index, {"audio": _audio()}) for index in range(8)])
    with pytest.raises(ValueError, match="at most 8 guides"):
        _chain((8, {"audio": _audio()}), upstream=chain)
    assert len(chain) == 8


# ====================================================================== client


async def test_guide_multipart_contract(api_calls):
    clip_bytes = _mp4(frames=22)
    audio = _audio(channels=2, samples=4800, sample_rate=24000)
    mono = _audio(channels=1, samples=480, sample_rate=16000)
    chain = _chain(
        (36, {"image": _image(0.2)}),
        (-22, {"video": VideoInput(clip_bytes), "audio": audio}),
        (0, {"audio": mono}),
        (-1, {"image": _image(0.8), "audio": audio}),
    )

    assert await _generate(chain) == b"generated-video"

    manifest = json.loads(_value(api_calls, "timeline_guides"))
    assert manifest == [
        {"frame_index": 36, "image": {"upload_index": 0}},
        {"frame_index": -22, "video": {"upload_index": 1}, "audio": {"upload_index": 2}},
        {"frame_index": 0, "audio": {"upload_index": 3}},
        {"frame_index": -1, "image": {"upload_index": 4}, "audio": {"upload_index": 5}},
    ]
    uploads = _uploads(api_calls, "guide_files")
    assert [(filename, content_type) for filename, content_type, _ in uploads] == [
        ("guide_0_image.png", "image/png"),
        ("guide_1_video.mp4", "video/mp4"),
        ("guide_2_audio.flac", "audio/flac"),
        ("guide_3_audio.flac", "audio/flac"),
        ("guide_4_image.png", "image/png"),
        ("guide_5_audio.flac", "audio/flac"),
    ]
    assert not any(name.startswith("guide_files[") for name, *_ in _fields(api_calls))

    for index, value in ((0, 0.2), (4, 0.8)):
        decoded = Image.open(BytesIO(uploads[index][2]))
        assert decoded.format == "PNG" and decoded.size == (WIDTH, HEIGHT)
        assert decoded.convert("RGB").getpixel((5, 5)) == (int(value * 255),) * 3

    frames, fps, codec = _decode_video(uploads[1][2])
    assert uploads[1][2] == clip_bytes
    assert (len(frames), fps, codec) == (22, 24, "h264")
    # YUV 4:2:0 round trip: allow a couple of code values of drift.
    assert all(abs(int(frame[0, 0, 0]) - 10 * index) <= 2 for index, frame in enumerate(frames))

    for index, source in ((2, audio), (3, mono), (5, audio)):
        waveform, sample_rate, codec = _decode_audio(uploads[index][2])
        expected = source["waveform"][0].numpy()
        assert (codec, sample_rate) == ("flac", source["sample_rate"])
        assert waveform.shape == expected.shape
        assert np.abs(waveform - expected).max() < 1e-3
        assert np.abs(waveform).max() > 0.1

    # Guides ride beside, not inside, extra_params; nothing local reaches the wire.
    extra_params = json.loads(_value(api_calls, "extra_params"))
    assert extra_params == {"task": "t2va", "aspect_ratio": "16:9"}
    raw_manifest = _value(api_calls, "timeline_guides")
    assert "tensor" not in raw_manifest and "/" not in raw_manifest and "\\" not in raw_manifest
    assert _value(api_calls, "quality") == "lossless"
    assert _value(api_calls, "fps") == "24"
    assert _value(api_calls, "num_frames") == "124"


async def test_guides_leave_reference_uploads_and_labels_alone(api_calls):
    (references,) = VLLMOmniVideoReferences().get_references(image_1=_image(0.1), audio_1=_audio())
    chain = _chain((36, {"image": _image(0.6)}), (72, {"audio": _audio()}))

    await _generate(chain, references=references)

    assert [(name, content_type) for name, content_type, _ in _uploads(api_calls, "input_references")] == [
        ("image_1.png", "image/png"),
        ("audio_1.mp3", "audio/mpeg"),
    ]
    assert json.loads(_value(api_calls, "timeline_guides")) == [
        {"frame_index": 36, "image": {"upload_index": 0}},
        {"frame_index": 72, "audio": {"upload_index": 1}},
    ]
    assert json.loads(_value(api_calls, "extra_params"))["task"] == "ref2va"


@pytest.mark.parametrize(
    ("inputs", "task", "frame_indices"),
    [
        pytest.param({}, "t2va", None, id="guides-only"),
        pytest.param({"first_frame": _image(0.0)}, "fl2va", [0], id="first-frame"),
        pytest.param({"last_frame": _image(1.0)}, "fl2va", [-1], id="last-frame"),
        pytest.param({"first_frame": _image(0.0), "last_frame": _image(1.0)}, "fl2va", [0, -1], id="both"),
        pytest.param({"references": {"image_1": _image(0.3)}}, "ref2va", None, id="ref2va"),
    ],
)
async def test_guides_do_not_change_task_routing(api_calls, inputs, task, frame_indices):
    await _generate(_chain((36, {"image": _image()})), **inputs)
    extra_params = json.loads(_value(api_calls, "extra_params"))
    assert extra_params["task"] == task
    assert extra_params.get("frame_indices") == frame_indices
    assert len(_uploads(api_calls, "guide_files")) == 1


@pytest.mark.parametrize(
    "inputs",
    [
        pytest.param({}, id="t2va"),
        pytest.param({"lora": {"local_path": "/l.safetensors", "name": "turbo", "scale": 1.0}}, id="lora"),
        pytest.param(
            {"latent_edit": {"source_video": VideoInput(b"src"), "video_mask": torch.zeros(HEIGHT, WIDTH)}},
            id="latent-edit",
        ),
    ],
)
async def test_requests_without_guides_are_unchanged(api_calls, inputs):
    await _generate(None, **inputs)
    names = [name for name, *_ in _fields(api_calls)]
    expected = ["model", "prompt", "width", "height", "num_frames", "fps"]
    expected += ["lora"] if "lora" in inputs else []
    expected += ["source_video", "video_noise_mask"] if "latent_edit" in inputs else []
    assert names == [*expected, "extra_params"]
    assert "quality" not in names and "timeline_guides" not in names and "guide_files" not in names


@pytest.mark.parametrize("steps", [4, 5, 50])
async def test_guides_keep_sampling_steps(api_calls, steps):
    sampling = DiffusionSamplingParams({"num_inference_steps": steps, "seed": 7})
    await _generate(_chain((0, {"image": _image()})), sampling_params=sampling)
    assert _value(api_calls, "num_inference_steps") == str(steps)
    assert _value(api_calls, "quality") == "lossless"


async def test_explicit_lossless_quality_is_not_duplicated(api_calls):
    await _generate(_chain((0, {"image": _image()})), sampling_params={"quality": "lossless"})
    assert _value(api_calls, "quality") == "lossless"


async def test_zero_scale_lora_is_inactive_and_still_forwarded(api_calls):
    lora = {"local_path": "/l.safetensors", "name": "turbo", "scale": 0.0}
    await _generate(_chain((0, {"image": _image()})), lora=lora)
    assert json.loads(_value(api_calls, "lora")) == lora


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"model": "Wan-AI/Wan2.2-T2V-A14B-Diffusers"}, "supported only for MiniMax-H3"),
        ({"fps": 16}, "24 FPS timeline; set fps to 24 \\(got 16\\)"),
        ({"lora": {"local_path": "/l", "name": "turbo", "scale": 0.7}}, "disconnect the request-level LoRA"),
        ({"lora": {"local_path": "/l", "name": "turbo"}}, "disconnect the request-level LoRA"),
        ({"sampling_params": {"quality": "high"}}, "require quality=lossless"),
        ({"latent_edit": {"source_video": VideoInput(b"src"), "video_mask": torch.zeros(1, 4, 4)}}, "latent_edit"),
        ({"latent_edit": {"audio_mask": 0.0}}, "latent_edit"),
    ],
)
async def test_unsupported_guide_requests_fail_before_encoding(api_calls, encoders, kwargs, message):
    with pytest.raises(ValueError, match=message):
        await _generate(
            _chain((36, {"image": _image()}), (-22, {"video": VideoInput(b"c"), "audio": _audio()})), **kwargs
        )
    api_calls.assert_not_called()
    assert encoders == []


@pytest.mark.parametrize(
    ("guides", "message"),
    [
        (H3TimelineGuides(), "timeline_guides is empty"),
        ({"frame_index": 0}, "must be the output of a MiniMax H3 Add Guide"),
        ("[]", "must be the output of a MiniMax H3 Add Guide"),
        ([{"frame_index": index, "audio": _audio()} for index in range(9)], "holds 9 guides"),
        ([{"frame_index": 0, "image": _image()}, "guide"], "timeline_guides\\[1\\] must be a guide"),
        ([{"frame_index": 0, "image": _image()}, {"frame_index": 2.0, "image": _image()}], "\\[1\\].frame_index"),
        ([{"frame_index": 0, "image": _image(batch=3)}], "Create Video"),
        ([{"frame_index": 0, "audio": _audio(batch=2)}], "batch of 2 clips"),
        ([{"frame_index": 0, "audio": _audio(channels=3)}], "3 channels"),
        ([{"frame_index": 0}], "needs an image, video, or audio"),
    ],
)
async def test_invalid_guide_chains_fail_before_encoding(api_calls, encoders, guides, message):
    with pytest.raises(ValueError, match=message):
        await _generate(guides)
    api_calls.assert_not_called()
    assert encoders == []


@pytest.mark.parametrize(
    ("kind", "media"),
    [("video", {"video": VideoInput(b"x" * 17)}), ("image", {"image": _image()}), ("audio", {"audio": _audio()})],
)
async def test_guide_byte_limits_use_encoded_size(api_calls, monkeypatch, kind, media):
    chain = _chain((0, media))
    monkeypatch.setitem(api_client.MAX_TIMELINE_GUIDE_BYTES, kind, 16)
    with pytest.raises(ValueError, match=f"timeline_guides\\[0\\].{kind} is .* MiB .* above the server's 0 MiB limit"):
        await _generate(chain)
    api_calls.assert_not_called()


async def test_guide_byte_limit_is_inclusive(api_calls, monkeypatch):
    monkeypatch.setitem(api_client.MAX_TIMELINE_GUIDE_BYTES, "video", 16)
    await _generate(_chain((0, {"video": VideoInput(b"x" * 16)})))
    assert _uploads(api_calls, "guide_files")[0][2] == b"x" * 16


def test_guide_byte_limits_match_server_contract():
    assert api_client.MAX_TIMELINE_GUIDE_BYTES == {"image": 30 * 2**20, "video": 50 * 2**20, "audio": 15 * 2**20}
    assert api_client.MAX_TIMELINE_GUIDES == 8


async def test_eight_guides_with_audio_use_sixteen_uploads(api_calls):
    chain = _chain(*[(index * 17, {"image": _image(index / 10), "audio": _audio(samples=240)}) for index in range(8)])
    await _generate(chain)
    manifest = json.loads(_value(api_calls, "timeline_guides"))
    assert [entry["frame_index"] for entry in manifest] == [index * 17 for index in range(8)]
    assert [entry[kind]["upload_index"] for entry in manifest for kind in ("image", "audio")] == list(range(16))
    assert len(_uploads(api_calls, "guide_files")) == 16


async def test_guide_clip_is_written_as_mp4(api_calls):
    """ComfyUI keeps a loaded MKV/WebM container unless MP4 is requested; the server takes MP4/MOV only."""
    calls = []

    class RecordingVideo(VideoInput):
        def save_to(self, file, **kwargs):
            calls.append(kwargs)
            super().save_to(file)

    await _generate(_chain((0, {"video": RecordingVideo(b"clip")})))
    assert calls == [{"format": "mp4"}]
    assert _uploads(api_calls, "guide_files") == [("guide_0_video.mp4", "video/mp4", b"clip")]


async def test_reference_video_encoding_is_unchanged(api_calls):
    calls = []

    class RecordingVideo(VideoInput):
        def save_to(self, file, **kwargs):
            calls.append(kwargs)
            super().save_to(file)

    (references,) = VLLMOmniVideoReferences().get_references(video_1=RecordingVideo(b"ref"))
    await _generate(None, references=references)
    assert calls == [{}]


async def test_guide_audio_is_encoded_from_a_detached_float32_copy(api_calls):
    waveform = _audio()["waveform"].to(torch.float64).requires_grad_(True)
    audio = {"waveform": waveform, "sample_rate": 24000}
    await _generate(_chain((0, {"audio": audio})))
    assert audio["waveform"] is waveform and waveform.dtype == torch.float64 and waveform.requires_grad
    decoded, _, _ = _decode_audio(_uploads(api_calls, "guide_files")[0][2])
    assert np.abs(decoded - waveform.detach()[0].numpy()).max() < 1e-3


# ===================================================================== node -> client


def _node_inputs(**overrides) -> dict:
    return {
        "url": "http://localhost/v1",
        "model": H3,
        "prompt": "Follow the guides.",
        "width": WIDTH,
        "height": HEIGHT,
        "fps": 24,
        "duration": 5.167,
        **overrides,
    }


async def test_generate_video_node_forwards_guides(api_calls):
    chain = _chain((36, {"image": _image()}))
    (result,) = await VLLMOmniGenerateVideo().generate(**_node_inputs(timeline_guides=chain))
    assert result == b"generated-video"
    assert json.loads(_value(api_calls, "timeline_guides")) == [{"frame_index": 36, "image": {"upload_index": 0}}]
    assert _value(api_calls, "num_frames") == "124"


async def test_generate_video_node_rejects_fast_h3_guides_before_upload(api_calls, encoders):
    fast_h3 = {"url": "http://localhost/v1", "model": H3}
    chain = _chain((36, {"image": _image()}))
    validation = VLLMOmniGenerateVideo.VALIDATE_INPUTS(url="u", model="m", fast_h3=fast_h3, timeline_guides=chain)
    assert isinstance(validation, str) and "not FastH3" in validation
    with pytest.raises(ValueError, match="not FastH3"):
        await VLLMOmniGenerateVideo().generate(
            **_node_inputs(url="u", model="m", fast_h3=fast_h3, timeline_guides=chain)
        )
    api_calls.assert_not_called()
    assert encoders == []


async def test_generate_video_node_rejects_noop_latent_edit_with_guides(api_calls, encoders):
    (edit,) = VLLMOmniLatentMaskEditing().get_latent_edit(audio_mask=0.0)
    with pytest.raises(ValueError, match="cannot be combined with latent_edit"):
        await VLLMOmniGenerateVideo().generate(
            **_node_inputs(latent_edit=edit, timeline_guides=_chain((0, {"image": _image()})))
        )
    api_calls.assert_not_called()
    assert encoders == []


async def test_generate_video_node_keeps_duration_and_fps_with_guides(api_calls):
    with pytest.raises(ValueError, match="got 16"):
        await VLLMOmniGenerateVideo().generate(
            **_node_inputs(fps=16, duration=5.0, timeline_guides=_chain((0, {"image": _image()})))
        )
    api_calls.assert_not_called()


# ============================================================ real HTTP error paths


class _FakeVideosServer:
    """Minimal /v1/videos job server that records the multipart parts it parses."""

    def __init__(self, *, post=(200, {"id": "job", "status": "queued"}), polls=(), content=b""):
        self.post_response = post
        self.polls = list(polls)
        self.content = content
        self.posts: list[list[tuple[str, str | None, str | None, bytes]]] = []
        self.deleted = False
        app = web.Application()
        app.router.add_post("/v1/videos", self._create)
        app.router.add_get("/v1/videos/{id}", self._status)
        app.router.add_get("/v1/videos/{id}/content", self._content)
        app.router.add_delete("/v1/videos/{id}", self._delete)
        self.server = test_utils.TestServer(app)

    async def _create(self, request):
        parts = []
        async for part in await request.multipart():
            parts.append((part.name, part.filename, part.headers.get("Content-Type"), await part.read()))
        self.posts.append(parts)
        status, body = self.post_response
        return (
            web.json_response(body, status=status) if isinstance(body, dict) else web.Response(status=status, text=body)
        )

    async def _status(self, request):
        status, body = self.polls.pop(0) if len(self.polls) > 1 else self.polls[0]
        return (
            web.json_response(body, status=status) if isinstance(body, dict) else web.Response(status=status, text=body)
        )

    async def _content(self, request):
        return web.Response(body=self.content, content_type="video/mp4")

    async def _delete(self, request):
        self.deleted = True
        return web.json_response({"id": request.match_info["id"], "deleted": True})


@pytest.fixture
async def fake_server():
    servers = []

    async def start(**kwargs):
        fake = _FakeVideosServer(**kwargs)
        await fake.server.start_server()
        servers.append(fake)
        return fake, str(fake.server.make_url("/v1"))

    yield start
    for fake in servers:
        await fake.server.close()


async def _generate_against(url: str, **kwargs):
    client = api_client.VLLMOmniClient(url, poll_interval=0.01, max_poll_duration=kwargs.pop("max_poll", 5.0))
    return await client.generate_video(
        model=H3,
        prompt="Follow the guides.",
        width=WIDTH,
        height=HEIGHT,
        num_frames=124,
        fps=24,
        timeline_guides=_chain((36, {"image": _image()}), (-22, {"video": VideoInput(_mp4()), "audio": _audio()})),
        **kwargs,
    )


async def test_guides_reach_a_real_multipart_parser_and_output_audio_is_decoded(fake_server, monkeypatch):
    # Capture the real VideoComponents instead of conftest's discarding mock.
    monkeypatch.setattr(omni_format, "InputImpl", SimpleNamespace(VideoFromComponents=lambda components: components))
    output = _mp4(frames=29, fps=24, audio_rate=48000)
    fake, url = await fake_server(polls=[(200, {"id": "job", "status": "completed"})], content=output)

    components = await _generate_against(url)

    (parts,) = fake.posts
    names = [name for name, *_ in parts]
    assert names.count("timeline_guides") == 1 and names.count("guide_files") == 3
    guide_files = [(filename, content_type) for name, filename, content_type, _ in parts if name == "guide_files"]
    assert guide_files == [
        ("guide_0_image.png", "image/png"),
        ("guide_1_video.mp4", "video/mp4"),
        ("guide_2_audio.flac", "audio/flac"),
    ]
    manifest = json.loads(next(data for name, _, _, data in parts if name == "timeline_guides"))
    assert [entry["frame_index"] for entry in manifest] == [36, -22]

    images, audio = components["images"], components["audio"]
    assert tuple(images.shape) == (29, HEIGHT, WIDTH, 3)
    assert components["frame_rate"] == 24
    assert abs(float(images[10, 0, 0, 0]) - 100 / 255) < 2 / 255
    assert audio["sample_rate"] == 48000 and tuple(audio["waveform"].shape[:2]) == (1, 2)
    left, right = audio["waveform"][0]
    assert float(left.abs().max()) > 0.3 and float(right.abs().max()) > 0.15
    assert not torch.allclose(left, right)
    audio_seconds = audio["waveform"].shape[-1] / audio["sample_rate"]
    assert abs(audio_seconds - images.shape[0] / 24) <= 1 / 24
    assert fake.deleted


@pytest.mark.parametrize(
    ("server", "error", "message"),
    [
        pytest.param(
            {"post": (400, {"detail": "Timeline guides are not supported by this model."})},
            ValueError,
            "status 400: .*Timeline guides are not supported by this model",
            id="post-400",
        ),
        pytest.param(
            {"polls": [(404, "Video job not found")]}, ValueError, "status 404: Video job not found", id="poll-404"
        ),
        pytest.param(
            {"polls": [(502, "upstream worker unavailable")]},
            RuntimeError,
            "status 502: upstream worker unavailable",
            id="poll-502",
        ),
        pytest.param(
            {
                "polls": [
                    (200, {"id": "job", "status": "in_progress"}),
                    (
                        200,
                        {
                            "id": "job",
                            "status": "failed",
                            "error": {"code": 400, "message": "timeline guide at frame_index=-22 does not fit"},
                        },
                    ),
                ]
            },
            RuntimeError,
            "Video job failed: .*timeline guide at frame_index=-22 does not fit",
            id="failed-job-400",
        ),
        pytest.param(
            {"polls": [(200, {"id": "job", "status": "failed", "error": {"code": 500, "message": "worker crashed"}})]},
            RuntimeError,
            "Video job failed: .*500.*worker crashed",
            id="failed-job-500",
        ),
    ],
)
async def test_guide_request_errors_keep_status_and_body(fake_server, server, error, message):
    fake, url = await fake_server(**server)
    with pytest.raises(error, match=message):
        await _generate_against(url)
    assert len(fake.posts) == 1  # neither retried nor resent without guides
    assert any(name == "timeline_guides" for name, *_ in fake.posts[0])


async def test_guide_request_timeout_is_reported_without_retry(fake_server):
    fake, url = await fake_server(polls=[(200, {"id": "job", "status": "in_progress"})])
    with pytest.raises(RuntimeError, match="Timed out waiting for video job job"):
        await _generate_against(url, max_poll=0.05)
    assert len(fake.posts) == 1
