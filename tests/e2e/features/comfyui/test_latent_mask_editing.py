# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""End-to-end test for latent-mask editing serialization.

Exercises ``VLLMOmniClient.generate_video`` with a ``latent_edit`` payload
against a mock ``/v1/videos`` server and asserts the multipart fields the
client produced. Runs on CPU: the ComfyUI ``comfy_api`` / ``comfy_extras``
modules are mocked by this directory's ``conftest.py``.
"""

import asyncio
import json
import os
import socket
import subprocess
import sys
import time

import pytest
import torch
from comfy_api.input import VideoInput
from comfyui_vllm_omni.utils.api_client import VLLMOmniClient

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def mock_server(tmp_path):
    port = _free_port()
    state_file = tmp_path / "state.json"
    env = dict(os.environ, MOCK_STATE_FILE=str(state_file))
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "mock_videos_server:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=_TESTS_DIR,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1):
                    break
            except OSError:
                time.sleep(0.2)
        yield f"http://127.0.0.1:{port}/v1", state_file
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_latent_edit_serialization(mock_server):
    base_url, state_file = mock_server

    # A non-trivial video mask requires a source video. The mocked VideoInput
    # only needs to provide ``save_to`` for the client's multipart upload.
    source_video = VideoInput(b"mock_source_video")
    mask = torch.zeros(1, 120, 160)
    latent_edit = {"source_video": source_video, "video_mask": mask, "audio_mask": 0.5}

    async def run():
        client = VLLMOmniClient(base_url)
        return await client.generate_video(
            model="MiniMaxAI/MiniMax-H3",
            prompt="restyle the clip",
            width=160,
            height=120,
            num_frames=22,
            fps=24,
            latent_edit=latent_edit,
        )

    out = asyncio.run(run())
    assert out is not None

    with open(state_file) as f:
        fields = json.load(f)

    assert fields["source_video"]["file"] is True
    assert fields["source_video"]["content_type"] == "video/mp4"
    for name in ("video_noise_mask", "audio_noise_mask"):
        assert fields[name]["file"] is True
        assert fields[name]["content_type"] == "application/json"
    assert fields["audio_noise_mask"]["json"] == 0.5

    video_mask = fields["video_noise_mask"]["json"]
    assert len(video_mask) == 7  # num_frames=22 aligns to 22 (17n+5) -> Tv = 7
    assert len(video_mask[0]) == 6  # height 120 floors to 96 (multiple of 32) -> gh 6
    assert len(video_mask[0][0]) == 10  # width 160 -> gw 10
