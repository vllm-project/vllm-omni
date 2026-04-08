# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E online serving tests for Index-AniSora I2V models.

Exercises the /v1/videos async job lifecycle (create → poll → download → delete)
for both V1 (5B CogVideoX) and V2 (14B Wan2.1) via the HTTP API.
"""

import base64
import io
import os
import time
import uuid
from pathlib import Path
from typing import Any

import PIL.Image
import pytest
import requests

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServer

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"

MODEL_V1 = "Disty0/Index-anisora-5B-diffusers"
MODEL_V2 = "aardsoul-music/Wan2.1-Anisora-14B"

VIDEO_POLL_INTERVAL_S = 2.0
VIDEO_TIMEOUT_S = 900.0

NUM_FRAMES = 5
HEIGHT = 480
WIDTH = 720


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dummy_image_b64() -> str:
    """Return a small solid-color image as a base64-encoded JPEG string."""
    img = PIL.Image.new("RGB", (WIDTH, HEIGHT), color=(100, 149, 237))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def _video_api_url(server: OmniServer, suffix: str = "") -> str:
    return f"http://{server.host}:{server.port}/v1/videos{suffix}"


def _multipart_fields(payload: dict[str, Any]) -> list[tuple[str, tuple[None, str]]]:
    return [(k, (None, str(v))) for k, v in payload.items() if v is not None]


def _create_video_job(server: OmniServer, image_b64: str, **overrides: Any) -> requests.Response:
    payload: dict[str, Any] = {
        "prompt": f"a cat sitting calmly {uuid.uuid4().hex[:6]}",
        "width": WIDTH,
        "height": HEIGHT,
        "num_frames": NUM_FRAMES,
        "fps": 8,
        "num_inference_steps": 2,
        "guidance_scale": 6.0,
        "seed": 42,
    }
    payload.update(overrides)
    fields = _multipart_fields(payload)
    # Pass image as a file-like field
    img_bytes = base64.b64decode(image_b64)
    files = fields + [("input_reference", ("input.jpg", img_bytes, "image/jpeg"))]
    return requests.post(_video_api_url(server), files=files, timeout=VIDEO_TIMEOUT_S)


def _wait_for_completion(server: OmniServer, video_id: str) -> dict[str, Any]:
    deadline = time.time() + VIDEO_TIMEOUT_S
    while time.time() < deadline:
        resp = requests.get(_video_api_url(server, f"/{video_id}"), timeout=VIDEO_TIMEOUT_S)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        if data["status"] == "completed":
            return data
        if data["status"] == "failed":
            raise AssertionError(f"Job {video_id} failed: {data}")
        time.sleep(VIDEO_POLL_INTERVAL_S)
    raise AssertionError(f"Timed out waiting for job {video_id}")


def _assert_mp4(content: bytes) -> None:
    assert len(content) > 32
    assert content[4:8] == b"ftyp", "Response is not a valid MP4"


def _best_effort_delete(server: OmniServer, video_id: str) -> None:
    try:
        requests.delete(_video_api_url(server, f"/{video_id}"), timeout=30)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# V1 (5B) server fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="function")
def anisora_v1_server():
    with OmniServer(MODEL_V1, ["--num-gpus", "1", "--disable-log-stats"]) as server:
        yield server


@pytest.fixture(scope="function")
def anisora_v1_tp2_server():
    with OmniServer(MODEL_V1, ["--num-gpus", "2", "--tensor-parallel-size", "2", "--disable-log-stats"]) as server:
        yield server


# ---------------------------------------------------------------------------
# V2 (14B) server fixture — needs >80GB GPU; skip if not available
# ---------------------------------------------------------------------------


@pytest.fixture(scope="function")
def anisora_v2_server():
    # V2 (14B) requires 2× GPUs; single A100 80 GB runs OOM
    with OmniServer(MODEL_V2, ["--num-gpus", "2", "--disable-log-stats"]) as server:
        yield server


# ---------------------------------------------------------------------------
# V1 tests
# ---------------------------------------------------------------------------


@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100", "rocm": "MI325"})
def test_anisora_v1_online_create_poll_download_delete(anisora_v1_server: OmniServer, tmp_path: Path):
    """V1: full job lifecycle — create → poll → download → delete."""
    image_b64 = _dummy_image_b64()
    video_id: str | None = None
    try:
        resp = _create_video_job(anisora_v1_server, image_b64)
        assert resp.status_code == 200, resp.text
        created = resp.json()
        video_id = created["id"]
        assert created["status"] == "queued"
        assert created["model"] == MODEL_V1

        completed = _wait_for_completion(anisora_v1_server, video_id)
        assert completed["file_name"] is not None
        assert completed["progress"] == 100

        dl = requests.get(
            _video_api_url(anisora_v1_server, f"/{video_id}/content"),
            timeout=VIDEO_TIMEOUT_S,
        )
        assert dl.status_code == 200, dl.text
        assert dl.headers["content-type"].startswith("video/mp4")
        _assert_mp4(dl.content)

        out = tmp_path / completed["file_name"]
        out.write_bytes(dl.content)
        assert out.stat().st_size == len(dl.content)
    finally:
        if video_id:
            _best_effort_delete(anisora_v1_server, video_id)


# ---------------------------------------------------------------------------
# V1 TP=2 tests
# ---------------------------------------------------------------------------


@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
def test_anisora_v1_online_tp2_create_poll_download_delete(anisora_v1_tp2_server: OmniServer, tmp_path: Path):
    """V1 TP=2: full job lifecycle with tensor parallelism — create → poll → download → delete."""
    image_b64 = _dummy_image_b64()
    video_id: str | None = None
    try:
        resp = _create_video_job(anisora_v1_tp2_server, image_b64)
        assert resp.status_code == 200, resp.text
        created = resp.json()
        video_id = created["id"]
        assert created["status"] == "queued"
        assert created["model"] == MODEL_V1

        completed = _wait_for_completion(anisora_v1_tp2_server, video_id)
        assert completed["file_name"] is not None
        assert completed["progress"] == 100

        dl = requests.get(
            _video_api_url(anisora_v1_tp2_server, f"/{video_id}/content"),
            timeout=VIDEO_TIMEOUT_S,
        )
        assert dl.status_code == 200, dl.text
        assert dl.headers["content-type"].startswith("video/mp4")
        _assert_mp4(dl.content)

        out = tmp_path / completed["file_name"]
        out.write_bytes(dl.content)
        assert out.stat().st_size == len(dl.content)
    finally:
        if video_id:
            _best_effort_delete(anisora_v1_tp2_server, video_id)


# ---------------------------------------------------------------------------
# V2 tests
# ---------------------------------------------------------------------------


@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100", "rocm": "MI325"})
def test_anisora_v2_online_create_poll_download_delete(anisora_v2_server: OmniServer, tmp_path: Path):
    """V2: full job lifecycle — create → poll → download → delete."""
    image_b64 = _dummy_image_b64()
    video_id: str | None = None
    try:
        resp = _create_video_job(anisora_v2_server, image_b64, guidance_scale=5.0)
        assert resp.status_code == 200, resp.text
        created = resp.json()
        video_id = created["id"]
        assert created["status"] == "queued"

        completed = _wait_for_completion(anisora_v2_server, video_id)
        assert completed["file_name"] is not None
        assert completed["progress"] == 100

        dl = requests.get(
            _video_api_url(anisora_v2_server, f"/{video_id}/content"),
            timeout=VIDEO_TIMEOUT_S,
        )
        assert dl.status_code == 200, dl.text
        _assert_mp4(dl.content)
    finally:
        if video_id:
            _best_effort_delete(anisora_v2_server, video_id)
