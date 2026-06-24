# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""E2E online serving test for DreamZero OpenPI websocket serving."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from tests.dreamzero import openpi_client_helper as openpi_client
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams, get_open_port

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"

MODEL = "GEAR-Dreams/DreamZero-DROID"


test_params = [
    OmniServerParams(
        model=MODEL,
        port=8091,
        server_args=[
            "--deploy-config",
            "vllm_omni/deploy/dreamzero_tp1_cfg2.yaml",
        ],
        env_dict={
            "ATTENTION_BACKEND": "torch",
            "DIFFUSION_ATTENTION_BACKEND": "TORCH_SDPA",
            "VLLM_DISABLE_COMPILE_CACHE": "1",
            "MASTER_PORT": str(get_open_port()),
        },
    )
]


def _write_synthetic_video(path: Path, cv2_module, *, channel: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width, num_frames = 180, 320, 24
    writer = cv2_module.VideoWriter(str(path), cv2_module.VideoWriter_fourcc(*"mp4v"), 15.0, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {path}")
    try:
        for frame_idx in range(num_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[..., channel] = (frame_idx * 7) % 255
            frame[..., (channel + 1) % 3] = 64
            writer.write(cv2_module.cvtColor(frame, cv2_module.COLOR_RGB2BGR))
    finally:
        writer.release()


def _write_synthetic_dreamzero_videos(client_mod, video_dir: Path) -> None:
    for channel, file_name in enumerate(client_mod.CAMERA_FILES.values()):
        _write_synthetic_video(video_dir / file_name, client_mod.cv2, channel=channel)


@pytest.mark.full_model
@pytest.mark.diffusion
@pytest.mark.distributed_cuda
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_dreamzero_openpi_online(omni_server, tmp_path: Path) -> None:
    try:
        openpi_client.require_dependencies()
    except ModuleNotFoundError as exc:
        pytest.skip(str(exc))

    video_dir = tmp_path / "dreamzero_videos"
    _write_synthetic_dreamzero_videos(openpi_client, video_dir)
    result = openpi_client.run_policy_session(
        host=omni_server.host,
        port=omni_server.port,
        video_dir=video_dir,
        session_id="dreamzero-online-e2e",
    )

    openpi_client.validate_session_result(result)

    metadata = result["metadata"]
    assert metadata["needs_session_id"] is True
    assert metadata["needs_stereo_camera"] is False
    assert tuple(metadata["image_resolution"]) == (180, 320)
