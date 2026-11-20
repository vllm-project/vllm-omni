# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""MiniMax-H3 video-reference RainFusion online-serving case on A3."""

import io
import json
import os
import subprocess
from pathlib import Path

import av
import numpy as np
import pytest

from tests.dfx.conftest import _build_serve_args
from tests.helpers.assertions import assert_video_audio_valid
from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServer, OmniServerParams, OpenAIClientHandler

from ._common import assert_h3_video, run_ref2va_video

pytestmark = [pytest.mark.full_model, pytest.mark.diffusion]
TEST_NAME = "test_minimax_h3_ref2va_rainfusion_laser_4npu_a3"
CONFIG_PATH = Path(__file__).resolve().parents[3] / "dfx/perf/tests/test_minimax_h3_vllm_omni.json"
CONFIGS = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
MATCHES = [config for config in CONFIGS if config["test_name"] == TEST_NAME]
assert len(MATCHES) == 1, f"Expected one benchmark configuration for {TEST_NAME}"
CONFIG = MATCHES[0]
WORKLOAD = CONFIG["benchmark_params"][0]["extra_body"]
SERVE_ARGS = CONFIG["server_params"]["serve_args"]


@pytest.mark.parametrize(
    "omni_server",
    [
        pytest.param(
            OmniServerParams(
                model=os.environ.get("VLLM_TEST_MINIMAX_H3_REF2VA_MODEL") or "MiniMaxAI/MiniMax-H3",
                server_args=_build_serve_args(
                    {
                        key: value
                        for key, value in SERVE_ARGS.items()
                        if key not in {"stage-init-timeout", "init-timeout"}
                    }
                ),
                stage_init_timeout=SERVE_ARGS["stage-init-timeout"],
                init_timeout=SERVE_ARGS["init-timeout"],
                startup_timeout=2400,
            ),
            id=TEST_NAME,
            marks=hardware_marks(res={"npu": "A3"}, num_cards=4),
        )
    ],
    indirect=True,
)
def test_ref2va_rainfusion_laser_video(
    omni_server: OmniServer, openai_client: OpenAIClientHandler, tmp_path: Path
) -> None:
    """Generate video and audio from a video reference using the benchmark configuration."""
    assert os.environ.get("MINDIE_SD_FA_TYPE") == "ascend_laser_attention"
    assert SERVE_ARGS["diffusion-attention-config"]["default"]["block_sparse"]["start_step"] == 2
    assert WORKLOAD["num_inference_steps"] > 2
    reference = tmp_path / "reference.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=832x480:rate=30:duration=5",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(reference),
        ],
        check=True,
    )
    print("MINIMAX_H3_REF2VA_REQUEST_BEGIN", flush=True)
    video = run_ref2va_video(openai_client, omni_server.model, reference, WORKLOAD)
    assert_h3_video(video, width=WORKLOAD["width"], height=WORKLOAD["height"])
    assert_video_audio_valid(video, sample_rate=32000, channels=2)
    with av.open(io.BytesIO(video)) as container:
        frames = sum(1 for _ in container.decode(video=0))
    assert frames == WORKLOAD["num_frames"], f"Decoded {frames} video frames"
    with av.open(io.BytesIO(video)) as container:
        samples = 0
        for frame in container.decode(audio=0):
            samples += frame.samples
            assert np.isfinite(frame.to_ndarray()).all(), "Audio contains non-finite samples"
    assert samples > 0, "Ref2VA audio stream is empty"
