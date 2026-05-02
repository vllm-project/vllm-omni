# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E online serving tests for the async video generation API.

These tests exercise the real `/v1/videos` job lifecycle against a running
diffusion model, including create, retrieve, list, download, and delete flows.
"""

import os
import time
import uuid
from pathlib import Path
from typing import Any

import pytest
import requests

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServer

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"

MODEL = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
VIDEO_POLL_INTERVAL_S = 2.0
VIDEO_TIMEOUT_S = 900.0


@pytest.fixture(scope="module")
def video_server():
    """Start a real video generation server for async video API tests."""
    with OmniServer(
        MODEL,
        [
            "--num-gpus",
            "1",
            "--boundary-ratio",
            "0.875",
            "--flow-shift",
            "5.0",
            "--disable-log-stats",
        ],
    ) as server:
        yield server


def _video_api_url(server: OmniServer, suffix: str = "") -> str:
    return f"http://{server.host}:{server.port}/v1/videos{suffix}"


def _multipart_fields(payload: dict[str, Any]) -> list[tuple[str, tuple[None, str]]]:
    return [(key, (None, str(value))) for key, value in payload.items() if value is not None]


def _create_video_job(
    server: OmniServer,
    *,
    prompt: str | None = None,
    **overrides: Any,
) -> requests.Response:
    payload: dict[str, Any] = {
        "prompt": prompt or f"video api e2e {uuid.uuid4().hex[:8]}",
        "width": 640,
        "height": 480,
        "num_frames": 5,
        "fps": 8,
        "num_inference_steps": 2,
        "guidance_scale": 1.0,
        "guidance_scale_2": 1.0,
        "boundary_ratio": 0.875,
        "flow_shift": 5.0,
        "seed": 42,
    }
    payload.update(overrides)
    return requests.post(
        _video_api_url(server),
        files=_multipart_fields(payload),
        timeout=VIDEO_TIMEOUT_S,
    )


def _retrieve_video_job(server: OmniServer, video_id: str) -> requests.Response:
    return requests.get(_video_api_url(server, f"/{video_id}"), timeout=VIDEO_TIMEOUT_S)


def _wait_for_video_status(server: OmniServer, video_id: str, expected_status: str) -> dict[str, Any]:
    deadline = time.time() + VIDEO_TIMEOUT_S
    last_payload: dict[str, Any] | None = None

    while time.time() < deadline:
        response = _retrieve_video_job(server, video_id)
        assert response.status_code == 200, response.text
        last_payload = response.json()
        status = last_payload["status"]
        if status == expected_status:
            return last_payload
        if status == "failed":
            raise AssertionError(f"Video job {video_id} failed unexpectedly: {last_payload}")
        time.sleep(VIDEO_POLL_INTERVAL_S)

    raise AssertionError(
        f"Timed out waiting for video job {video_id} to reach status={expected_status}. Last payload: {last_payload}"
    )


def _wait_for_video_missing(server: OmniServer, video_id: str) -> None:
    deadline = time.time() + 30.0
    last_status: int | None = None

    while time.time() < deadline:
        response = _retrieve_video_job(server, video_id)
        last_status = response.status_code
        if response.status_code == 404:
            return
        time.sleep(0.5)

    raise AssertionError(f"Timed out waiting for video job {video_id} to disappear. Last status={last_status}")


def _delete_video_job(server: OmniServer, video_id: str) -> requests.Response:
    return requests.delete(_video_api_url(server, f"/{video_id}"), timeout=VIDEO_TIMEOUT_S)


def _delete_video_job_with_retry(server: OmniServer, video_id: str) -> requests.Response:
    deadline = time.time() + 30.0
    last_response: requests.Response | None = None

    while time.time() < deadline:
        response = _delete_video_job(server, video_id)
        last_response = response
        if response.status_code != 409:
            return response
        time.sleep(1.0)

    raise AssertionError(
        f"Timed out waiting to delete video job {video_id}. "
        f"Last response: {None if last_response is None else last_response.text}"
    )


def _best_effort_delete(server: OmniServer, video_id: str) -> None:
    try:
        response = _delete_video_job_with_retry(server, video_id)
        if response.status_code not in (200, 404):
            print(f"Cleanup delete for {video_id} returned {response.status_code}: {response.text}")
    except Exception as exc:
        print(f"Cleanup delete for {video_id} failed: {exc}")


def _assert_mp4_payload(content: bytes) -> None:
    assert len(content) > 32, f"Downloaded video payload is unexpectedly small: {len(content)} bytes"
    assert content[4:8] == b"ftyp", "Downloaded payload does not look like an MP4 file."


@pytest.mark.core_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100", "rocm": "MI325"})
def test_create_list_retrieve_download_video(video_server: OmniServer, tmp_path: Path) -> None:
    video_id: str | None = None
    try:
        create_response = _create_video_job(video_server, prompt="A small paper airplane gliding over a lake at dawn.")
        assert create_response.status_code == 200, create_response.text

        created = create_response.json()
        video_id = created["id"]
        assert created["object"] == "video"
        assert created["status"] == "queued"
        assert created["model"] == MODEL

        completed = _wait_for_video_status(video_server, video_id, "completed")
        assert completed["file_name"] is not None
        assert completed["completed_at"] is not None
        assert completed["progress"] == 100

        retrieve_response = _retrieve_video_job(video_server, video_id)
        assert retrieve_response.status_code == 200, retrieve_response.text
        retrieved = retrieve_response.json()
        assert retrieved["id"] == video_id
        assert retrieved["status"] == "completed"

        list_response = requests.get(_video_api_url(video_server), timeout=VIDEO_TIMEOUT_S)
        assert list_response.status_code == 200, list_response.text
        listed = list_response.json()
        assert listed["object"] == "list"
        assert any(item["id"] == video_id for item in listed["data"])

        download_response = requests.get(
            _video_api_url(video_server, f"/{video_id}/content"),
            timeout=VIDEO_TIMEOUT_S,
        )
        assert download_response.status_code == 200, download_response.text
        assert download_response.headers["content-type"].startswith("video/mp4")
        assert completed["file_name"] in download_response.headers.get("content-disposition", "")
        _assert_mp4_payload(download_response.content)

        output_path = tmp_path / completed["file_name"]
        output_path.write_bytes(download_response.content)
        assert output_path.stat().st_size == len(download_response.content)
    finally:
        if video_id is not None:
            _best_effort_delete(video_server, video_id)


@pytest.mark.core_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100", "rocm": "MI325"})
def test_delete_video_while_queued_or_in_progress(video_server: OmniServer) -> None:
    create_response = _create_video_job(
        video_server,
        prompt="A toy boat sailing through fog.",
        num_frames=13,
        num_inference_steps=4,
    )
    assert create_response.status_code == 200, create_response.text

    created = create_response.json()
    video_id = created["id"]
    assert created["status"] == "queued"

    # Delete immediately so the job is still queued or just entering execution.
    delete_response = _delete_video_job_with_retry(video_server, video_id)
    assert delete_response.status_code == 200, delete_response.text
    deleted = delete_response.json()
    assert deleted["id"] == video_id
    assert deleted["deleted"] is True
    assert deleted["object"] == "video.deleted"

    _wait_for_video_missing(video_server, video_id)


@pytest.mark.core_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100", "rocm": "MI325"})
def test_delete_video_after_processing(video_server: OmniServer) -> None:
    create_response = _create_video_job(
        video_server,
        prompt="A lantern drifting above a quiet forest.",
        seed=43,
    )
    assert create_response.status_code == 200, create_response.text

    video_id = create_response.json()["id"]
    completed = _wait_for_video_status(video_server, video_id, "completed")
    assert completed["file_name"] is not None

    delete_response = _delete_video_job_with_retry(video_server, video_id)
    assert delete_response.status_code == 200, delete_response.text
    deleted = delete_response.json()
    assert deleted["id"] == video_id
    assert deleted["deleted"] is True
    assert deleted["object"] == "video.deleted"

    _wait_for_video_missing(video_server, video_id)

    download_response = requests.get(
        _video_api_url(video_server, f"/{video_id}/content"),
        timeout=VIDEO_TIMEOUT_S,
    )
    assert download_response.status_code == 404, download_response.text
