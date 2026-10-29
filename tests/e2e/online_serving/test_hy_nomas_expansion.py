# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.helpers.assertions import assert_video_valid
from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServer, OmniServerParams

pytestmark = [pytest.mark.local_model, pytest.mark.diffusion, *hardware_marks(res={"cuda": "H200"}, num_cards=2)]


@pytest.mark.parametrize(
    "omni_server",
    [
        OmniServerParams(
            model=os.environ.get("HY_NOMAS_MODEL", "tencent/Hy-Nomas-Diffusers"),
            server_args=["--deploy-config", "vllm_omni/deploy/hy_nomas_ar_recent.yaml"],
            stage_init_timeout=1800,
            init_timeout=2100,
            startup_timeout=2100,
        )
    ],
    indirect=True,
)
def test_recent_history_delivers_120_seconds_through_shared_client(omni_server: OmniServer, tmp_path: Path):
    root = Path(__file__).resolve().parents[3]
    output = tmp_path / "long-stream.mp4"
    command = [
        sys.executable,
        str(root / "examples/online_serving/streaming_video_generation/streaming_video_client.py"),
        "--model",
        omni_server.model,
        "--host",
        omni_server.host,
        "--port",
        str(omni_server.port),
        "--no-helios-distilled-preset",
        "--prompt",
        "A forest beside a clear river, photorealistic, daylight, gentle camera motion.",
        "--height",
        "704",
        "--width",
        "1280",
        "--num-frames",
        "2881",
        "--fps",
        "24",
        "--num-inference-steps",
        "4",
        "--guidance-scale",
        "1",
        "--seed",
        "42",
        "--output",
        str(output),
    ]
    with (tmp_path / "client.log").open("w") as log:
        subprocess.run(command, check=True, timeout=3600, stdout=log, stderr=subprocess.STDOUT)
    assert_video_valid(output, num_frames=2881, width=1280, height=704, fps=24, fps_tolerance=0.01)
