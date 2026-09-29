# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from io import BytesIO
from unittest.mock import AsyncMock

import av
import pytest
import torch
from comfy_api.input import VideoInput
from comfyui_vllm_omni.nodes import VLLMOmniVideoReferences
from comfyui_vllm_omni.utils import api_client
from PIL import Image

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def api_calls(monkeypatch):
    calls = AsyncMock(return_value={"id": "test-video", "status": "completed"})
    monkeypatch.setattr(api_client, "url_json", calls)
    monkeypatch.setattr(api_client, "url_bytes", AsyncMock(return_value=b"generated-video"))
    monkeypatch.setattr(api_client, "bytes_to_video", lambda data: data)
    return calls


async def _generate(references, **kwargs):
    return await api_client.VLLMOmniClient("http://localhost/v1").generate_video(
        model="MiniMaxAI/MiniMax-H3",
        prompt="Use the reference subjects and sounds.",
        width=896,
        height=512,
        num_frames=90,
        fps=24,
        references=references,
        **kwargs,
    )


def _references(image_count, video_count, audio_count):
    refs = {f"image_{i}": torch.full((1, 16, 16, 3), i / 10) for i in range(1, image_count + 1)}
    refs.update({f"video_{i}": VideoInput(f"video-{i}".encode()) for i in range(1, video_count + 1)})
    refs.update(
        {f"audio_{i}": {"waveform": torch.zeros(1, 1, 24000), "sample_rate": 24000} for i in range(1, audio_count + 1)}
    )
    return refs


def test_reference_node_preserves_existing_ports_and_exposes_h3_limits():
    ports = VLLMOmniVideoReferences.INPUT_TYPES()["optional"]
    assert list(ports)[:6] == ["image_1", "image_2", "audio_1", "audio_2", "video_1", "video_2"]
    assert {name for name, kind in ports.items() if kind == ("IMAGE",)} == {f"image_{i}" for i in range(1, 10)}
    assert {name for name, kind in ports.items() if kind == ("VIDEO",)} == {"video_1", "video_2", "video_3"}
    assert {name for name, kind in ports.items() if kind == ("AUDIO",)} == {"audio_1", "audio_2", "audio_3"}
    inputs = _references(6, 3, 3)
    (refs,) = VLLMOmniVideoReferences().get_references(**inputs)
    assert refs.keys() == inputs.keys()
    assert all(refs[name] is value for name, value in inputs.items())


@pytest.mark.parametrize(
    "counts", [(1, 0, 0), (9, 0, 0), (0, 3, 0), (1, 0, 3), (0, 1, 3), (6, 3, 3), (9, 3, 0), (9, 0, 3)]
)
async def test_reference_uploads_keep_numeric_order_and_media_types(api_calls, counts):
    refs = _references(*counts)
    result = await _generate(dict(reversed(list(refs.items()))))
    assert result == b"generated-video"
    fields = api_calls.call_args_list[0].kwargs["data"]._fields
    uploads = [
        (options["filename"], headers["Content-Type"], value)
        for options, headers, value in fields
        if options["name"] == "input_references"
    ]
    expected = [
        (f"{kind}_{i}.{extension}", content_type)
        for kind, count, extension, content_type in (
            ("image", counts[0], "png", "image/png"),
            ("video", counts[1], "mp4", "video/mp4"),
            ("audio", counts[2], "mp3", "audio/mpeg"),
        )
        for i in range(1, count + 1)
    ]
    assert [(name, content_type) for name, content_type, _ in uploads] == expected
    for name, content_type, payload in uploads:
        data = payload.getvalue()
        if content_type == "image/png":
            index = int(name.removeprefix("image_").removesuffix(".png"))
            assert Image.open(BytesIO(data)).getpixel((0, 0)) == (int(index / 10 * 255),) * 3
        elif content_type == "video/mp4":
            assert data == name.removesuffix(".mp4").replace("_", "-").encode()
        else:
            with av.open(BytesIO(data)) as container:
                assert container.streams.audio[0].sample_rate == 24000
                assert sum(frame.samples for frame in container.decode(audio=0)) > 0


async def test_sparse_reference_slots_are_not_dropped(api_calls):
    inputs = _references(9, 3, 3)
    inputs = {name: inputs[name] for name in ("audio_3", "image_9", "video_3", "image_2")}
    (refs,) = VLLMOmniVideoReferences().get_references(image_1=None, **inputs)
    await _generate(refs)
    fields = api_calls.call_args_list[0].kwargs["data"]._fields
    assert [options["filename"] for options, _, _ in fields if options["name"] == "input_references"] == [
        "image_2.png",
        "image_9.png",
        "video_3.mp4",
        "audio_3.mp3",
    ]


@pytest.mark.parametrize(
    "counts, message",
    [
        ((0, 0, 0), "at least one image or video"),
        ((0, 0, 3), "at least one image or video"),
        ((9, 3, 1), "at most 12"),
        ((9, 3, 3), "at most 12"),
    ],
)
async def test_invalid_reference_combinations_fail_before_upload(api_calls, counts, message):
    with pytest.raises(ValueError, match=message):
        await _generate(_references(*counts))
    api_calls.assert_not_called()


@pytest.mark.parametrize("name", ["image_10", "video_4", "audio_4", "image_0"])
async def test_unsupported_reference_slots_are_not_silently_ignored(api_calls, name):
    refs = _references(1, 0, 0)
    refs[name] = object()
    with pytest.raises(ValueError, match="Unsupported reference input"):
        await _generate(refs)
    api_calls.assert_not_called()


async def test_frame_and_references_are_mutually_exclusive(api_calls):
    with pytest.raises(ValueError, match="only one of frame or references"):
        await _generate(_references(1, 0, 0), frame=torch.zeros(1, 16, 16, 3))
    api_calls.assert_not_called()
