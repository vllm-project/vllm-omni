# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
L4 e2e tests for LTX-2 in online serving mode.

Coverage:
- Cache-DiT (1 GPU)
- Cache-DiT + TP=2 + VAE patch parallel=2 (2 GPUs)

LTX-2 is served through the async video API (/v1/videos) in online serving mode.
"""

import time

import pytest
import requests

from tests.conftest import (
    OmniServer,
    OmniServerParams,
)
from tests.utils import hardware_marks

# Disable proxy for local test server requests
NO_PROXY = {"http": None, "https": None}

MODEL = "Lightricks/LTX-2"
PROMPT = "A cinematic close-up of ocean waves at golden hour."
NEGATIVE_PROMPT = "worst quality, inconsistent motion, blurry, jittery, distorted"

SINGLE_CARD_FEATURE_MARKS = hardware_marks(res={"cuda": "L4"})
PARALLEL_MARKS = hardware_marks(res={"cuda": "L4"}, num_cards=2)

VIDEO_TIMEOUT_S = 900.0
VIDEO_POLL_INTERVAL_S = 2.0


def _get_diffusion_feature_cases(model: str):
    """Return L4 diffusion feature cases for LTX-2."""
    return [
        # (1 GPU) Cache-DiT
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=[
                    "--cache-backend",
                    "cache_dit",
                ],
            ),
            id="single_card_cachedit",
            marks=SINGLE_CARD_FEATURE_MARKS,
        ),
        # (2 GPUs) Cache-DiT + TP=2 + VAE patch parallel=2
        pytest.param(
            OmniServerParams(
                model=model,
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
            id="parallel_cachedit_tp2_vae2",
            marks=PARALLEL_MARKS,
        ),
    ]


@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.parametrize(
    "omni_server",
    _get_diffusion_feature_cases(MODEL),
    indirect=True,
)
def test_ltx2(
    omni_server: OmniServer,
):
    """L4 diffusion feature coverage for LTX-2 on L4."""
    url = f"http://{omni_server.host}:{omni_server.port}/v1/videos"

    payload = {
        "prompt": PROMPT,
        "height": 512,
        "width": 768,
        "num_frames": 9,
        "num_inference_steps": 2,
        "negative_prompt": NEGATIVE_PROMPT,
        "guidance_scale": 4.0,
        "fps": 24,
        "seed": 42,
    }

    files = [(k, (None, str(v))) for k, v in payload.items()]

    create_resp = requests.post(url, files=files, timeout=VIDEO_TIMEOUT_S, proxies=NO_PROXY)
    assert create_resp.status_code == 200, create_resp.text

    data = create_resp.json()
    video_id = data["id"]
    assert data["status"] == "queued"
    assert data["model"] == omni_server.model

    # Poll for completion
    deadline = time.time() + VIDEO_TIMEOUT_S
    last_status = None
    while time.time() < deadline:
        status_resp = requests.get(f"{url}/{video_id}", timeout=30, proxies=NO_PROXY)
        assert status_resp.status_code == 200, status_resp.text
        status_data = status_resp.json()
        last_status = status_data["status"]
        if last_status == "completed":
            break
        if last_status == "failed":
            raise AssertionError(f"Video generation failed: {status_data}")
        time.sleep(VIDEO_POLL_INTERVAL_S)
    else:
        raise AssertionError(f"Timed out waiting for video generation. Last status: {last_status}")

    # Verify download returns a valid MP4
    download_resp = requests.get(
        f"{url}/{video_id}/content",
        timeout=VIDEO_TIMEOUT_S,
        proxies=NO_PROXY,
    )
    assert download_resp.status_code == 200, download_resp.text
    assert download_resp.headers["content-type"].startswith("video/mp4")
    assert len(download_resp.content) > 32, (
        f"Downloaded video payload is unexpectedly small: {len(download_resp.content)} bytes"
    )
    assert download_resp.content[4:8] == b"ftyp", "Downloaded payload does not look like an MP4 file."
