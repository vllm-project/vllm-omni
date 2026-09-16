# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import AsyncIterator
from io import BytesIO
from pathlib import Path
from typing import Any

import av
import numpy as np
import pytest
import pytest_asyncio
import torch
from aiohttp import web
from comfy_api.input import VideoInput
from comfyui_vllm_omni import nodes as nodes_module
from comfyui_vllm_omni.nodes import VLLMOmniGenerateVideo, VLLMOmniMiniMaxH3Control
from comfyui_vllm_omni.utils.api_client import VLLMOmniClient
from comfyui_vllm_omni.utils.format import mask_tensor_to_png_bytes
from comfyui_vllm_omni.utils.types import MiniMaxH3ModelSpecificParams
from PIL import Image

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _encoded_video_bytes(frame_count: int = 2, width: int = 16, height: int = 16) -> bytes:
    output = BytesIO()
    with av.open(output, mode="w", format="mp4") as container:
        stream = container.add_stream("h264", rate=8)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        for _ in range(frame_count):
            frame = av.VideoFrame.from_ndarray(np.zeros((height, width, 3), dtype=np.uint8), format="rgb24")
            container.mux(stream.encode(frame))
        container.mux(stream.encode(None))
    return output.getvalue()


@pytest_asyncio.fixture
async def multipart_video_server(unused_tcp_port: int) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    captured: dict[str, Any] = {"post_count": 0, "poll_count": 0}
    response_video = _encoded_video_bytes()

    async def create_video(request: web.Request) -> web.Response:
        captured["post_count"] += 1
        captured["fields"] = {}
        captured["files"] = []
        if request.content_type.startswith("multipart/"):
            reader = await request.multipart()
            while part := await reader.next():
                payload = await part.read()
                if part.filename is None:
                    captured["fields"][part.name] = payload.decode()
                else:
                    captured["files"].append(
                        {
                            "name": part.name,
                            "filename": part.filename,
                            "content_type": part.headers.get("Content-Type"),
                            "payload": payload,
                        }
                    )
        else:
            captured["fields"].update(await request.post())
        if response_status := captured.get("response_status"):
            return web.Response(status=response_status, text="controlled fake server failure")
        return web.json_response({"id": "job-1", "status": captured.get("initial_status", "completed")})

    async def video_status(_request: web.Request) -> web.Response:
        responses = captured.get("status_responses", [{"id": "job-1", "status": "completed"}])
        index = captured["poll_count"]
        captured["poll_count"] += 1
        response = responses[min(index, len(responses) - 1)]
        return web.json_response(response)

    async def video_content(_request: web.Request) -> web.Response:
        return web.Response(body=response_video, content_type="video/mp4")

    async def delete_video(_request: web.Request) -> web.Response:
        return web.json_response({"deleted": True})

    app = web.Application()
    app.router.add_post("/v1/videos", create_video)
    app.router.add_get("/v1/videos/job-1", video_status)
    app.router.add_get("/v1/videos/job-1/content", video_content)
    app.router.add_delete("/v1/videos/job-1", delete_video)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", unused_tcp_port)
    await site.start()
    try:
        yield f"http://127.0.0.1:{unused_tcp_port}/v1", captured
    finally:
        await runner.cleanup()


def _h3_model_params(*, include_type: bool = True) -> MiniMaxH3ModelSpecificParams:
    params = MiniMaxH3ModelSpecificParams({"audio_flow_shift": 3.0, "flow_shift": 12.0})
    if include_type:
        params["type"] = "minimax_h3"
    return params


async def _generate_with_control(
    api_server: str,
    control: dict[str, Any],
    *,
    model: str = "company/custom-served-h3-name",
) -> VideoInput:
    (result,) = await VLLMOmniGenerateVideo().generate(
        url=api_server,
        model=model,
        prompt="Control test",
        width=32,
        height=32,
        fps=8,
        duration=5 / 8,
        model_params=_h3_model_params(),
        control=control,
    )
    return result


class EncodingVideo:
    """Small VIDEO-compatible object that encodes its components when save_to is called."""

    def __init__(self, images: torch.Tensor, fps: int = 8):
        self.images = images
        self.fps = fps
        self.save_calls: list[tuple[str, str, float | None]] = []

    def save_to(
        self,
        output: BytesIO,
        format: str = "auto",
        codec: str = "auto",
        crf: float | None = None,
    ) -> None:
        self.save_calls.append((format, codec, crf))
        output_format = "mp4" if format == "auto" else format
        output_codec = "h264" if codec == "auto" else codec
        with av.open(output, mode="w", format=output_format) as container:
            stream = container.add_stream(output_codec, rate=self.fps)
            stream.width = self.images.shape[2]
            stream.height = self.images.shape[1]
            stream.pix_fmt = "yuv420p"
            if crf is not None:
                stream.options = {"crf": str(crf)}
            for image in self.images:
                rgb = (image.numpy() * 255).astype(np.uint8)
                container.mux(stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")))
            container.mux(stream.encode(None))


def test_minimax_h3_control_node_contract_and_registration() -> None:
    input_types = VLLMOmniMiniMaxH3Control.INPUT_TYPES()
    assert set(input_types["required"]) == {"control_type", "strength"}
    assert input_types["required"]["control_type"][0] == ["canny", "depth", "hed", "mlsd", "pose", "inpaint"]
    assert input_types["required"]["strength"][1]["default"] == 1.0
    assert {name: socket[:1] for name, socket in input_types["optional"].items()} == {
        "control_video": ("VIDEO",),
        "source_video": ("VIDEO",),
        "mask": ("MASK",),
        "mask_video": ("VIDEO",),
    }
    assert VLLMOmniMiniMaxH3Control.RETURN_TYPES == ("MINIMAX_H3_CONTROL",)
    assert VLLMOmniGenerateVideo.INPUT_TYPES()["optional"]["control"] == ("MINIMAX_H3_CONTROL",)

    plugin_root = Path(__file__).parents[4] / "apps" / "ComfyUI-vLLM-Omni"
    spec = importlib.util.spec_from_file_location(
        "comfyui_vllm_omni_plugin",
        plugin_root / "__init__.py",
        submodule_search_locations=[str(plugin_root)],
    )
    assert spec is not None and spec.loader is not None
    plugin = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = plugin
    spec.loader.exec_module(plugin)
    assert plugin.NODE_CLASS_MAPPINGS["VLLMOmniMiniMaxH3Control"].__name__ == "VLLMOmniMiniMaxH3Control"
    assert plugin.NODE_DISPLAY_NAME_MAPPINGS["VLLMOmniMiniMaxH3Control"] == "MiniMax-H3 Control"


@pytest.mark.asyncio
async def test_pose_control_uses_t2va_multipart_and_type_routing(
    multipart_video_server: tuple[str, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api_server, captured = multipart_video_server
    captured["initial_status"] = "queued"
    captured["status_responses"] = [
        {"id": "job-1", "status": "in_progress"},
        {"id": "job-1", "status": "completed"},
    ]
    monkeypatch.setattr(
        nodes_module,
        "VLLMOmniClient",
        lambda base_url: VLLMOmniClient(base_url, poll_interval=0),
    )
    (control,) = VLLMOmniMiniMaxH3Control().get_control(
        control_type="pose",
        strength=1.25,
        control_video=VideoInput(b"pose-control-video"),  # type: ignore[reportAbstractUsage]
    )

    result = await VLLMOmniGenerateVideo().generate(
        url=api_server,
        model="company/custom-served-h3-name",
        prompt="Animate the subject using the pose sequence.",
        width=32,
        height=32,
        fps=8,
        duration=5 / 8,
        model_params=_h3_model_params(),
        control=control,
    )

    assert isinstance(result[0], VideoInput)
    assert captured["post_count"] == 1
    assert captured["poll_count"] == 2
    assert captured["fields"]["control_type"] == "pose"
    assert captured["fields"]["flow_shift"] == "12.0"
    assert "type" not in captured["fields"]
    assert json.loads(captured["fields"]["extra_params"]) == {
        "pose": {"control_context_scale": 1.25},
        "audio_flow_shift": 3.0,
        "task": "t2va",
        "aspect_ratio": "1:1",
    }
    assert captured["files"] == [
        {
            "name": "control_reference",
            "filename": "control.mp4",
            "content_type": "video/mp4",
            "payload": b"pose-control-video",
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("control_type", ["canny", "depth", "hed", "mlsd", "pose"])
async def test_structure_modes_share_wire_path_without_preprocessing(
    multipart_video_server: tuple[str, dict[str, Any]],
    control_type: str,
) -> None:
    api_server, captured = multipart_video_server
    payload = f"{control_type}-unchanged".encode()
    (control,) = VLLMOmniMiniMaxH3Control().get_control(
        control_type=control_type,
        strength=0.75,
        control_video=VideoInput(payload),  # type: ignore[reportAbstractUsage]
    )

    await _generate_with_control(api_server, control)

    assert captured["fields"]["control_type"] == control_type
    assert json.loads(captured["fields"]["extra_params"])[control_type] == {"control_context_scale": 0.75}
    assert captured["files"][0]["name"] == "control_reference"
    assert captured["files"][0]["payload"] == payload


@pytest.mark.asyncio
async def test_unsupported_control_mode_is_rejected_before_post(
    multipart_video_server: tuple[str, dict[str, Any]],
) -> None:
    api_server, captured = multipart_video_server
    with pytest.raises(ValueError, match="Unsupported MiniMax-H3 control_type"):
        await _generate_with_control(
            api_server,
            {
                "control_type": "scribble",
                "control_context_scale": 1.0,
                "control_video": VideoInput(b"unchanged"),  # type: ignore[reportAbstractUsage]
            },
        )
    assert captured["post_count"] == 0


@pytest.mark.asyncio
async def test_unknown_control_field_is_rejected_before_post(
    multipart_video_server: tuple[str, dict[str, Any]],
) -> None:
    api_server, captured = multipart_video_server
    with pytest.raises(ValueError, match="Unknown MiniMax-H3 control fields: typo_field"):
        await _generate_with_control(
            api_server,
            {
                "control_type": "pose",
                "control_context_scale": 1.0,
                "control_video": VideoInput(b"unchanged"),  # type: ignore[reportAbstractUsage]
                "typo_field": "silently ignored before F1",
            },
        )
    assert captured["post_count"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("extra_mode", ["pose", "canny"])
async def test_control_rejects_extra_params_mode_namespace_before_post(
    multipart_video_server: tuple[str, dict[str, Any]],
    extra_mode: str,
) -> None:
    api_server, captured = multipart_video_server
    with pytest.raises(ValueError, match=f"Conflicting MiniMax-H3 control namespaces: {extra_mode}"):
        await VLLMOmniClient(api_server).generate_video(
            model="custom-h3",
            prompt="namespace conflict",
            width=32,
            height=32,
            num_frames=5,
            fps=8,
            model_params=_h3_model_params(),
            control={
                "control_type": "pose",
                "control_context_scale": 1.0,
                "control_video": VideoInput(b"control"),  # type: ignore[reportAbstractUsage]
            },
            **{extra_mode: {"control_context_scale": 2.0}},
        )
    assert captured["post_count"] == 0


def test_structure_mode_requires_control_video() -> None:
    with pytest.raises(ValueError, match="pose requires control_video"):
        VLLMOmniMiniMaxH3Control().get_control(
            control_type="pose",
            strength=1.0,
        )


@pytest.mark.parametrize("shape", [(4, 5), (1, 4, 5)])
def test_static_mask_png_is_grayscale_thresholded_and_rewound(shape: tuple[int, ...]) -> None:
    mask = torch.tensor(
        [
            [0.0, 0.49, 0.5, 0.5001, 1.0],
            [1.0, 0.0, 0.9, 0.1, 0.8],
            [0.2, 0.7, 0.3, 0.6, 0.4],
            [1.0, 1.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    ).reshape(shape)

    encoded = mask_tensor_to_png_bytes(mask)

    assert encoded.tell() == 0
    with Image.open(encoded) as image:
        assert image.mode == "L"
        assert image.size == (5, 4)
        pixels = np.asarray(image)
    assert set(np.unique(pixels)) == {0, 255}
    assert np.array_equal(pixels > 0, mask.reshape(4, 5).numpy() > 0.5)


@pytest.mark.parametrize(
    "mask",
    [
        torch.zeros((2, 4, 5)),
        torch.zeros((1, 1, 4, 5)),
        torch.tensor([[float("nan")]]),
        torch.tensor([[float("inf")]]),
    ],
)
def test_static_mask_rejects_invalid_shape_or_values(mask: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        mask_tensor_to_png_bytes(mask)


@pytest.mark.asyncio
async def test_inpaint_static_mask_only_multipart(
    multipart_video_server: tuple[str, dict[str, Any]],
) -> None:
    api_server, captured = multipart_video_server
    mask = torch.tensor([[0.0, 1.0], [0.25, 0.75]])
    (control,) = VLLMOmniMiniMaxH3Control().get_control(
        control_type="inpaint",
        strength=1.0,
        mask=mask,
    )

    await _generate_with_control(api_server, control)

    assert captured["fields"]["control_type"] == "inpaint"
    assert [item["name"] for item in captured["files"]] == ["mask_reference"]
    uploaded = captured["files"][0]
    assert uploaded["filename"] == "mask.png"
    assert uploaded["content_type"] == "image/png"
    with Image.open(BytesIO(uploaded["payload"])) as image:
        assert image.mode == "L"
        assert image.size == (2, 2)


@pytest.mark.parametrize(
    "control,expected",
    [
        ({"control_type": "inpaint", "strength": 1.0}, "inpaint requires mask or mask_video"),
        (
            {
                "control_type": "inpaint",
                "strength": 1.0,
                "source_video": object(),
            },
            "source_video requires mask or mask_video",
        ),
    ],
)
def test_inpaint_rejects_missing_mask(control: dict[str, Any], expected: str) -> None:
    with pytest.raises(ValueError, match=expected):
        VLLMOmniMiniMaxH3Control().get_control(**control)


@pytest.mark.asyncio
async def test_source_video_and_static_mask_use_distinct_fields(
    multipart_video_server: tuple[str, dict[str, Any]],
) -> None:
    api_server, captured = multipart_video_server
    (control,) = VLLMOmniMiniMaxH3Control().get_control(
        control_type="inpaint",
        strength=1.0,
        source_video=VideoInput(b"source-video"),  # type: ignore[reportAbstractUsage]
        mask=torch.ones((2, 2)),
    )

    await _generate_with_control(api_server, control)

    files = {item["name"]: item for item in captured["files"]}
    assert set(files) == {"source_reference", "mask_reference"}
    assert files["source_reference"]["filename"] == "source.mp4"
    assert files["source_reference"]["content_type"] == "video/mp4"
    assert files["source_reference"]["payload"] == b"source-video"
    assert files["mask_reference"]["filename"] == "mask.png"
    assert files["mask_reference"]["content_type"] == "image/png"
    assert "input_reference" not in files


@pytest.mark.asyncio
async def test_control_uploads_pin_lossless_mp4_h264_for_every_video_role(
    multipart_video_server: tuple[str, dict[str, Any]],
) -> None:
    api_server, captured = multipart_video_server
    control_video = VideoInput(b"control-video")  # type: ignore[reportAbstractUsage]
    source_video = VideoInput(b"source-video")  # type: ignore[reportAbstractUsage]
    mask_video = EncodingVideo(torch.zeros((2, 16, 16, 3), dtype=torch.float32))
    (control,) = VLLMOmniMiniMaxH3Control().get_control(
        control_type="canny",
        strength=1.0,
        control_video=control_video,
        source_video=source_video,
        mask_video=mask_video,
    )

    await _generate_with_control(api_server, control)

    assert control_video.save_calls == [("mp4", "h264", 0)]
    assert source_video.save_calls == [("mp4", "h264", 0)]
    assert mask_video.save_calls == [("mp4", "h264", 0)]
    files = {item["name"]: item for item in captured["files"]}
    assert {name: files[name]["content_type"] for name in files} == {
        "control_reference": "video/mp4",
        "source_reference": "video/mp4",
        "mask_reference": "video/mp4",
    }


@pytest.mark.asyncio
async def test_temporal_mask_is_explicit_mp4_h264_and_preserves_binary_classes(
    multipart_video_server: tuple[str, dict[str, Any]],
) -> None:
    api_server, captured = multipart_video_server
    images = torch.zeros((3, 32, 32, 3), dtype=torch.float32)
    images[0, :, 16:, :] = 1.0
    images[1, 16:, :, :] = 1.0
    images[2, 8:24, 8:24, :] = 1.0
    mask_video = EncodingVideo(images)
    (control,) = VLLMOmniMiniMaxH3Control().get_control(
        control_type="inpaint",
        strength=1.0,
        mask_video=mask_video,
    )

    await _generate_with_control(api_server, control)

    assert mask_video.save_calls == [("mp4", "h264", 0)]
    uploaded = next(item for item in captured["files"] if item["name"] == "mask_reference")
    assert uploaded["filename"] == "mask.mp4"
    assert uploaded["content_type"] == "video/mp4"
    with av.open(BytesIO(uploaded["payload"])) as container:
        stream = container.streams.video[0]
        decoded = np.stack([frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]) / 255.0
        assert container.format.name.startswith("mov,mp4")
        assert stream.codec_context.name == "h264"
    assert decoded.shape == tuple(images.shape)
    assert np.array_equal(decoded[..., 0] > 0.5, images[..., 0].numpy() > 0.5)


def test_static_and_temporal_masks_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="Provide only one of mask or mask_video"):
        VLLMOmniMiniMaxH3Control().get_control(
            control_type="inpaint",
            strength=1.0,
            mask=torch.ones((2, 2)),
            mask_video=VideoInput(b"mask-video"),  # type: ignore[reportAbstractUsage]
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("control_type", ["canny", "depth", "hed", "mlsd", "pose"])
@pytest.mark.parametrize(
    "has_source,mask_kind",
    [(False, None), (False, "static"), (False, "temporal"), (True, "static"), (True, "temporal")],
)
async def test_structure_control_legal_matrix(
    multipart_video_server: tuple[str, dict[str, Any]],
    control_type: str,
    has_source: bool,
    mask_kind: str | None,
) -> None:
    api_server, captured = multipart_video_server
    control: dict[str, Any] = {
        "control_type": control_type,
        "control_context_scale": 0.5,
        "control_video": VideoInput(b"control"),  # type: ignore[reportAbstractUsage]
    }
    if has_source:
        control["source_video"] = VideoInput(b"source")  # type: ignore[reportAbstractUsage]
    if mask_kind == "static":
        control["mask"] = torch.ones((2, 2))
    elif mask_kind == "temporal":
        control["mask_video"] = EncodingVideo(torch.ones((2, 16, 16, 3)))

    await _generate_with_control(api_server, control)

    names = [item["name"] for item in captured["files"]]
    assert ("control_reference" in names) is True
    assert ("source_reference" in names) is has_source
    assert ("mask_reference" in names) is (mask_kind is not None)
    assert captured["fields"]["control_type"] == control_type
    assert json.loads(captured["fields"]["extra_params"])[control_type] == {"control_context_scale": 0.5}


@pytest.mark.asyncio
@pytest.mark.parametrize("has_control_hint", [False, True])
@pytest.mark.parametrize("has_source", [False, True])
@pytest.mark.parametrize("mask_kind", ["static", "temporal"])
async def test_inpaint_legal_matrix(
    multipart_video_server: tuple[str, dict[str, Any]],
    has_control_hint: bool,
    has_source: bool,
    mask_kind: str,
) -> None:
    api_server, captured = multipart_video_server
    control: dict[str, Any] = {"control_type": "inpaint", "control_context_scale": 2.0}
    if has_control_hint:
        control["control_video"] = VideoInput(b"hint")  # type: ignore[reportAbstractUsage]
    if has_source:
        control["source_video"] = VideoInput(b"source")  # type: ignore[reportAbstractUsage]
    if mask_kind == "static":
        control["mask"] = torch.ones((2, 2))
    else:
        control["mask_video"] = EncodingVideo(torch.ones((2, 16, 16, 3)))

    await _generate_with_control(api_server, control)

    names = [item["name"] for item in captured["files"]]
    assert ("control_reference" in names) is has_control_hint
    assert ("source_reference" in names) is has_source
    assert "mask_reference" in names
    assert captured["fields"]["control_type"] == "inpaint"
    assert "inpaint" not in captured["fields"]
    assert json.loads(captured["fields"]["extra_params"])["inpaint"] == {"control_context_scale": 2.0}


@pytest.mark.parametrize("strength", [-1, float("nan"), float("inf"), float("-inf"), True, "1.0", 10**1000])
def test_invalid_strength_is_rejected(strength: Any) -> None:
    with pytest.raises(ValueError, match="finite, non-negative number"):
        VLLMOmniMiniMaxH3Control().get_control(
            control_type="pose",
            strength=strength,
            control_video=VideoInput(b"control"),  # type: ignore[reportAbstractUsage]
        )


@pytest.mark.parametrize("strength", [0, 0.0, 1, 2.5])
def test_finite_non_negative_strength_is_allowed(strength: float) -> None:
    (control,) = VLLMOmniMiniMaxH3Control().get_control(
        control_type="pose",
        strength=strength,
        control_video=VideoInput(b"control"),  # type: ignore[reportAbstractUsage]
    )
    assert control["control_context_scale"] == strength


@pytest.mark.asyncio
@pytest.mark.parametrize("status,error_type", [(400, ValueError), (500, RuntimeError)])
async def test_video_post_errors_keep_exception_type_and_are_not_retried(
    multipart_video_server: tuple[str, dict[str, Any]],
    status: int,
    error_type: type[Exception],
) -> None:
    api_server, captured = multipart_video_server
    captured["response_status"] = status
    control = {
        "control_type": "pose",
        "control_context_scale": 1.0,
        "control_video": VideoInput(b"control"),  # type: ignore[reportAbstractUsage]
    }
    with pytest.raises(error_type, match=f"status {status}.*controlled fake server failure"):
        await _generate_with_control(api_server, control)
    assert captured["post_count"] == 1


@pytest.mark.asyncio
async def test_failed_video_job_keeps_runtime_error_context(
    multipart_video_server: tuple[str, dict[str, Any]],
) -> None:
    api_server, captured = multipart_video_server
    captured["initial_status"] = "queued"
    captured["status_responses"] = [
        {"id": "job-1", "status": "failed", "error": "control decode failed"},
    ]
    with pytest.raises(RuntimeError, match="Video job failed:.*control decode failed"):
        await VLLMOmniClient(api_server, poll_interval=0).generate_video(
            model="custom-h3",
            prompt="failed job",
            width=32,
            height=32,
            num_frames=5,
            fps=8,
            model_params=_h3_model_params(),
            control={
                "control_type": "pose",
                "control_context_scale": 1.0,
                "control_video": VideoInput(b"control"),  # type: ignore[reportAbstractUsage]
            },
        )
    assert captured["post_count"] == 1
    assert captured["poll_count"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy_mode", ["t2va", "fl2va", "ref2va"])
async def test_legacy_h3_requests_keep_exact_non_control_payload(
    multipart_video_server: tuple[str, dict[str, Any]],
    legacy_mode: str,
) -> None:
    api_server, captured = multipart_video_server
    kwargs: dict[str, Any] = {}
    if legacy_mode == "fl2va":
        kwargs["frame"] = torch.zeros((1, 4, 4, 3))
    elif legacy_mode == "ref2va":
        kwargs["references"] = {
            "video_1": VideoInput(b"reference-one"),  # type: ignore[reportAbstractUsage]
            "video_2": VideoInput(b"reference-two"),  # type: ignore[reportAbstractUsage]
        }

    await VLLMOmniClient(api_server).generate_video(
        model="MiniMaxAI/MiniMax-H3",
        prompt="Legacy request",
        width=32,
        height=32,
        num_frames=5,
        fps=8,
        model_params=_h3_model_params(include_type=legacy_mode != "t2va"),
        **kwargs,
    )

    expected_extra_params: dict[str, Any] = {"task": legacy_mode, "audio_flow_shift": 3.0}
    if legacy_mode == "t2va":
        expected_extra_params["aspect_ratio"] = "1:1"
    assert captured["fields"] == {
        "model": "MiniMaxAI/MiniMax-H3",
        "prompt": "Legacy request",
        "width": "32",
        "height": "32",
        "num_frames": "5",
        "fps": "8",
        "flow_shift": "12.0",
        "extra_params": json.dumps(expected_extra_params, ensure_ascii=False),
    }
    expected_file_names = {"t2va": [], "fl2va": ["input_reference"], "ref2va": ["input_references", "input_references"]}
    assert [item["name"] for item in captured["files"]] == expected_file_names[legacy_mode]
    assert all(
        item["name"] not in {"control_reference", "source_reference", "mask_reference"} for item in captured["files"]
    )
    assert "control_type" not in captured["fields"]


@pytest.mark.parametrize("conflicting_input", ["frame", "references"])
def test_generate_video_rejects_control_with_existing_conditioning(conflicting_input: str) -> None:
    kwargs: dict[str, Any] = {conflicting_input: object()}
    result = VLLMOmniGenerateVideo.VALIDATE_INPUTS(
        "http://localhost:8000/v1",
        "custom-model",
        control={"control_type": "pose"},
        **kwargs,
    )
    assert result == "MiniMax-H3 control cannot be combined with frame or references."


@pytest.mark.asyncio
@pytest.mark.parametrize("conflicting_input", ["frame", "references"])
async def test_client_rejects_control_conflicts_before_post(
    multipart_video_server: tuple[str, dict[str, Any]],
    conflicting_input: str,
) -> None:
    api_server, captured = multipart_video_server
    kwargs: dict[str, Any] = {conflicting_input: object()}
    with pytest.raises(ValueError, match="cannot be combined with frame or references"):
        await VLLMOmniClient(api_server).generate_video(
            model="custom-h3",
            prompt="conflict",
            width=32,
            height=32,
            num_frames=5,
            fps=8,
            control={
                "control_type": "pose",
                "control_context_scale": 1.0,
                "control_video": VideoInput(b"control"),  # type: ignore[reportAbstractUsage]
            },
            **kwargs,
        )
    assert captured["post_count"] == 0
