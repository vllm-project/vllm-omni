# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""End-to-end test for latent-mask editing serialization.

Exercises ``VLLMOmniClient.generate_video`` with a ``latent_edit`` payload and
asserts the exact multipart fields the client produces — in particular that the
masks are uploaded as JSON *file* parts (the server's ``_parse_video_form``
declares ``video_noise_mask``/``audio_noise_mask`` as ``UploadFile``, so a plain
string field would be rejected with a 422). Runs on CPU: ``comfy_api`` /
``comfy_extras`` are mocked by this directory's ``conftest.py``.
"""

import json
from unittest.mock import AsyncMock

import pytest
import torch
from comfy_api.input import VideoInput
from comfyui_vllm_omni.utils import api_client

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def api_calls(monkeypatch):
    calls = AsyncMock(return_value={"id": "test-video", "status": "completed"})
    monkeypatch.setattr(api_client, "url_json", calls)
    monkeypatch.setattr(api_client, "url_bytes", AsyncMock(return_value=b"generated-video"))
    monkeypatch.setattr(api_client, "bytes_to_video", lambda data: data)
    return calls


def _fields_by_name(calls):
    fields = calls.call_args_list[0].kwargs["data"]._fields
    return {options["name"]: (options, headers, value) for options, headers, value in fields}


async def _generate(**kwargs):
    return await api_client.VLLMOmniClient("http://localhost/v1").generate_video(
        model="MiniMaxAI/MiniMax-H3",
        prompt="restyle the clip",
        width=160,
        height=120,
        num_frames=22,
        fps=24,
        **kwargs,
    )


async def test_latent_edit_serialization(api_calls):
    latent_edit = {
        "source_video": VideoInput(b"mock_source_video"),
        "video_mask": torch.zeros(1, 120, 160),
        "audio_mask": 0.5,
    }

    out = await _generate(latent_edit=latent_edit)
    assert out == b"generated-video"

    fields = _fields_by_name(api_calls)

    source = fields["source_video"]
    assert source[0]["filename"] == "source.mp4"
    assert source[1]["Content-Type"] == "video/mp4"

    # video_noise_mask must be a JSON file part, not a string field. The raw
    # mask is sent as-is (the server resolves the latent grid).
    video = fields["video_noise_mask"]
    assert video[0]["filename"] == "video-mask.json"
    assert video[1]["Content-Type"] == "application/json"
    grid = json.loads(video[2].decode())
    assert len(grid) == 1  # one temporal slice
    assert len(grid[0]) == 120  # raw height
    assert len(grid[0][0]) == 160  # raw width

    # A scalar audio_mask is sent as a JSON file part carrying the scalar value.
    audio = fields["audio_noise_mask"]
    assert audio[0]["filename"] == "audio-mask.json"
    assert audio[1]["Content-Type"] == "application/json"
    assert audio[2].decode() == "0.5"


async def test_latent_edit_serialization_temporal_audio(api_calls):
    latent_edit = {
        "source_video": VideoInput(b"mock_source_video"),
        "video_mask": torch.zeros(1, 120, 160),
        "audio_temporal_mask": torch.full((178,), 0.5),
    }

    out = await _generate(latent_edit=latent_edit)
    assert out == b"generated-video"

    fields = _fields_by_name(api_calls)

    # A temporal audio mask is sent as a raw JSON file part.
    audio = fields["audio_noise_mask"]
    assert audio[0]["filename"] == "audio-mask.json"
    assert audio[1]["Content-Type"] == "application/json"
    audio_grid = json.loads(audio[2].decode())
    assert len(audio_grid) == 178  # time steps
