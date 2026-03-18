# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Comprehensive e2e tests of diffusion features for Wan2.1-VACE in online serving mode.

Wan2.1-VACE supports: Cache-DiT, Ulysses-SP, Ring, CFG-Parallel, TP,
VAE-Patch-Parallel, HSDP. TeaCache is NOT supported for this model, so
Cache-DiT is used in place of TeaCache for single-card and CFG tests.

Uses the 1.3B variant for faster CI testing.

Feature matrix (following the Adding a Diffusion Model guide §5.2):
  Single GPU:
    - Cache-DiT + layerwise CPU offload
  Two GPUs:
    - Cache-DiT + Ulysses-SP = 2
    - Cache-DiT + Ring = 2
    - Cache-DiT + CFG-Parallel = 2
    - Cache-DiT + TP = 2 + VAE-Patch-Parallel = 2
    - Cache-DiT + HSDP = 2 + VAE-Patch-Parallel = 2
"""

import time

import pytest
import requests

from tests.conftest import OmniServer, OmniServerParams
from tests.utils import hardware_marks

MODEL = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
PROMPT = "A cat walking slowly across a sunlit garden path"
VIDEO_TIMEOUT_S = 900.0
VIDEO_POLL_INTERVAL_S = 2.0

SINGLE_CARD_FEATURE_MARKS = hardware_marks(res={"cuda": "H100"})
PARALLEL_FEATURE_MARKS = hardware_marks(res={"cuda": "H100"}, num_cards=2)


def _get_vace_feature_cases():
    return [
        # Single GPU: Cache-DiT + layerwise CPU offload
        pytest.param(
            OmniServerParams(
                model=MODEL,
                server_args=[
                    "--cache-backend",
                    "cache_dit",
                    "--enable-layerwise-offload",
                    "--vae-use-tiling",
                ],
            ),
            id="single_card_001",
            marks=SINGLE_CARD_FEATURE_MARKS,
        ),
        # 2 GPUs: Cache-DiT + Ulysses-SP = 2
        pytest.param(
            OmniServerParams(
                model=MODEL,
                server_args=[
                    "--cache-backend",
                    "cache_dit",
                    "--ulysses-degree",
                    "2",
                    "--vae-use-tiling",
                ],
            ),
            id="parallel_001",
            marks=PARALLEL_FEATURE_MARKS,
        ),
        # 2 GPUs: Cache-DiT + Ring = 2
        pytest.param(
            OmniServerParams(
                model=MODEL,
                server_args=[
                    "--cache-backend",
                    "cache_dit",
                    "--ring",
                    "2",
                    "--vae-use-tiling",
                ],
            ),
            id="parallel_002",
            marks=PARALLEL_FEATURE_MARKS,
        ),
        # 2 GPUs: Cache-DiT + CFG-Parallel = 2
        pytest.param(
            OmniServerParams(
                model=MODEL,
                server_args=[
                    "--cache-backend",
                    "cache_dit",
                    "--cfg-parallel-size",
                    "2",
                    "--vae-use-tiling",
                ],
            ),
            id="parallel_003",
            marks=PARALLEL_FEATURE_MARKS,
        ),
        # 2 GPUs: Cache-DiT + TP = 2 + VAE-Patch-Parallel = 2
        pytest.param(
            OmniServerParams(
                model=MODEL,
                server_args=[
                    "--cache-backend",
                    "cache_dit",
                    "--tensor-parallel-size",
                    "2",
                    "--vae-patch-parallel-size",
                    "2",
                    "--vae-use-tiling",
                ],
            ),
            id="parallel_004",
            marks=PARALLEL_FEATURE_MARKS,
        ),
        # 2 GPUs: Cache-DiT + HSDP = 2 + VAE-Patch-Parallel = 2
        pytest.param(
            OmniServerParams(
                model=MODEL,
                server_args=[
                    "--cache-backend",
                    "cache_dit",
                    "--hsdp-shard-size",
                    "2",
                    "--vae-patch-parallel-size",
                    "2",
                    "--vae-use-tiling",
                ],
            ),
            id="parallel_005",
            marks=PARALLEL_FEATURE_MARKS,
        ),
    ]


def _video_api_url(server: OmniServer, suffix: str = "") -> str:
    return f"http://{server.host}:{server.port}/v1/videos{suffix}"


def _create_video_job(server: OmniServer, *, prompt: str | None = None, **overrides) -> requests.Response:
    payload = {
        "prompt": prompt or PROMPT,
        "width": 480,
        "height": 320,
        "num_frames": 5,
        "fps": 8,
        "num_inference_steps": 2,
        "guidance_scale": 5.0,
        "seed": 42,
    }
    payload.update(overrides)
    fields = [(key, (None, str(value))) for key, value in payload.items() if value is not None]
    return requests.post(_video_api_url(server), files=fields, timeout=VIDEO_TIMEOUT_S)


def _wait_for_video_completed(server: OmniServer, video_id: str) -> dict:
    deadline = time.time() + VIDEO_TIMEOUT_S
    while time.time() < deadline:
        resp = requests.get(_video_api_url(server, f"/{video_id}"), timeout=VIDEO_TIMEOUT_S)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        if data["status"] == "completed":
            return data
        if data["status"] == "failed":
            raise AssertionError(f"Video job {video_id} failed: {data}")
        time.sleep(VIDEO_POLL_INTERVAL_S)
    raise AssertionError(f"Timed out waiting for video job {video_id}")


@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.parametrize(
    "omni_server",
    _get_vace_feature_cases(),
    indirect=True,
)
def test_wan_2_1_vace(omni_server: OmniServer):
    """Test VACE T2V generation with all supported diffusion acceleration features."""
    create_resp = _create_video_job(omni_server)
    assert create_resp.status_code == 200, create_resp.text

    created = create_resp.json()
    video_id = created["id"]

    try:
        completed = _wait_for_video_completed(omni_server, video_id)
        assert completed["file_name"] is not None

        # Download and verify the generated video is a valid MP4
        download_resp = requests.get(
            _video_api_url(omni_server, f"/{video_id}/content"),
            timeout=VIDEO_TIMEOUT_S,
        )
        assert download_resp.status_code == 200, download_resp.text
        assert download_resp.headers["content-type"].startswith("video/mp4")
        content = download_resp.content
        assert len(content) > 32, f"Video payload too small: {len(content)} bytes"
        assert content[4:8] == b"ftyp", "Not a valid MP4 file"
    finally:
        # Best-effort cleanup
        try:
            requests.delete(_video_api_url(omni_server, f"/{video_id}"), timeout=30)
        except Exception:
            pass
