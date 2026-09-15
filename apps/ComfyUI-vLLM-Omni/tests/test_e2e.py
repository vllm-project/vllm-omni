# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""End-to-end test for latent-mask editing serialization.

Exercises ``VLLMOmniClient.generate_video`` with a ``latent_edit`` payload
against the mock ``/v1/videos`` server and asserts the multipart fields the
client produced. Requires a ComfyUI checkout (``comfy_api``) and CUDA; the
module is skipped when either is unavailable.
"""

import asyncio
import json
import os
import socket
import subprocess
import sys
import time

import pytest

_EXT_ROOT = os.environ.get("VLLM_OMNI_EXT_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_COMFYUI_DIR = os.environ.get("COMFYUI_DIR", "")

for _p in (_EXT_ROOT, _COMFYUI_DIR):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import comfy_api.latest  # noqa: F401  # init first to avoid the input/_io circular import
    import torch  # noqa: F401
    from comfyui_vllm_omni.utils.api_client import VLLMOmniClient
    from comfyui_vllm_omni.utils.format import bytes_to_video
except ImportError:
    pytest.skip("ComfyUI / extension import unavailable", allow_module_level=True)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="comfy.model_management requires CUDA",
)

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _make_source_mp4(path: str) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x120:rate=24:duration=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=32000:duration=1",
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-shortest",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-ar",
            "32000",
            "-ac",
            "2",
            path,
        ],
        check=True,
        capture_output=True,
    )


@pytest.fixture
def mock_server(tmp_path):
    port = _free_port()
    state_file = tmp_path / "state.json"
    env = dict(os.environ, MOCK_STATE_FILE=str(state_file))
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "mock_server:app", "--host", "127.0.0.1", "--port", str(port)],
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


def test_latent_edit_serialization(mock_server, tmp_path):
    base_url, state_file = mock_server
    source = tmp_path / "source.mp4"
    _make_source_mp4(str(source))

    with open(source, "rb") as f:
        source_video = bytes_to_video(f.read())

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
    assert fields["audio_noise_mask"] == "0.5"

    video_mask = json.loads(fields["video_noise_mask"])
    assert len(video_mask) == 7  # num_frames=22 aligns to 22 (17n+5) -> Tv = 7
    assert len(video_mask[0]) == 6  # height 120 floors to 96 (multiple of 32) -> gh 6
    assert len(video_mask[0][0]) == 10  # width 160 -> gw 10
