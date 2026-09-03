# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""MiniMax-H3 DLO + DP2 online-serving smoke test."""

from __future__ import annotations

import concurrent.futures
import io
import json
import os

import av
import pytest
import requests

from tests.helpers.assertions import assert_video_valid
from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServer, OmniServerParams, OpenAIClientHandler

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL = os.environ.get("VLLM_TEST_MINIMAX_H3_MODEL", "MiniMaxAI/MiniMax-H3")
WIDTH = 1344
HEIGHT = 768
FPS = 24
NUM_INFERENCE_STEPS = 4
REQUEST_TIMEOUT_SECONDS = 1800
H100_TWO_CARD_MARKS = hardware_marks(res={"cuda": "H100"}, num_cards=2)

SERVER_ARGS = [
    "--trust-remote-code",
    "--task-type",
    "fl2va",
    "--num-gpus",
    "2",
    "--tensor-parallel-size",
    "1",
    "--data-parallel-size",
    "2",
    "--request-batch-max-wait-ms",
    "500",
    "--usp",
    "1",
    "--ring",
    "1",
    "--text-encoder-tp-size",
    "1",
    "--vae-patch-parallel-size",
    "1",
    "--vae-parallel-mode",
    "tile",
    "--vae-use-tiling",
    "--enable-distributed-layerwise-offload",
]


def _assert_audio_stream_present(video: bytes) -> None:
    """Assert that the generated MP4 contains decodable audio samples."""
    with av.open(io.BytesIO(video)) as container:
        audio_streams = [stream for stream in container.streams if stream.type == "audio"]
        assert audio_streams, "MiniMax-H3 MP4 has no audio stream"
        audio_frame = next(container.decode(audio=0), None)
        assert audio_frame is not None and audio_frame.samples > 0, "MiniMax-H3 MP4 audio stream is empty"


def _run_t2va_request(client: OpenAIClientHandler, seed: int) -> bytes:
    """Submit one synchronous T2VA request and return its MP4 body."""
    request_data = {
        "model": MODEL,
        "prompt": "In a snowy blue-purple forest, a traveler walks past a sleeping giant; footsteps crunch in the snow while the creature softly breathes.",
        "width": str(WIDTH),
        "height": str(HEIGHT),
        "fps": str(FPS),
        "num_inference_steps": str(NUM_INFERENCE_STEPS),
        "flow_shift": "12",
        "seed": str(seed),
        "extra_params": json.dumps(
            {
                "task": "t2va",
                "duration": 4.0,
                "aspect_ratio": "16:9",
                "audio_flow_shift": 3.0,
            },
            separators=(",", ":"),
        ),
    }
    response = requests.post(
        f"{client.base_url.rstrip('/')}/v1/videos/sync",
        data=request_data,
        headers={"Accept": "video/mp4"},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    assert response.headers.get("content-type", "").startswith("video/mp4")
    assert response.content, "MiniMax-H3 returned an empty video body"
    return response.content


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.slow
@pytest.mark.parametrize(
    "omni_server",
    [
        pytest.param(
            OmniServerParams(
                model=MODEL,
                server_args=SERVER_ARGS,
                stage_init_timeout=1800,
                init_timeout=1800,
            ),
            id="minimax_h3_dlo_dp2_t2va",
            marks=H100_TWO_CARD_MARKS,
        )
    ],
    indirect=True,
)
def test_minimax_h3_dlo_dp2_t2va(omni_server: OmniServer, openai_client: OpenAIClientHandler) -> None:
    """Validate one complete DLO all-gather DP2 wave with two T2VA jobs."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(_run_t2va_request, openai_client, seed) for seed in (1101, 1102)]
        videos = [future.result() for future in futures]

    for video in videos:
        assert_video_valid(video, width=WIDTH, height=HEIGHT, fps=FPS)
        _assert_audio_stream_present(video)
