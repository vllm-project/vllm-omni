# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""MiniMax-H3 DLO + DP2 online-serving smoke test."""

from __future__ import annotations

import concurrent.futures
import io
import json
import os
from pathlib import Path

import av
import pytest
import requests

from tests.helpers.assertions import assert_video_valid
from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServer, OpenAIClientHandler, get_model_prefix, resolve_tiny_model_path

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


def _server_args(*, served_model_name: str | None = None) -> list[str]:
    args = [
        *SERVER_ARGS,
        "--stage-init-timeout",
        "1800",
        "--init-timeout",
        "1800",
    ]
    if served_model_name is not None:
        args.extend(("--served-model-name", served_model_name))
    return args


def _hwr_server_args(root: Path, mode: str, *, served_model_name: str | None = None) -> list[str]:
    return [
        *_server_args(served_model_name=served_model_name),
        "--host-weight-runtime-mode",
        mode,
        "--host-weight-runtime-root",
        str(root),
    ]


def _assert_audio_stream_present(video: bytes) -> None:
    """Assert that the generated MP4 contains decodable audio samples."""
    with av.open(io.BytesIO(video)) as container:
        audio_streams = [stream for stream in container.streams if stream.type == "audio"]
        assert audio_streams, "MiniMax-H3 MP4 has no audio stream"
        audio_frame = next(container.decode(audio=0), None)
        assert audio_frame is not None and audio_frame.samples > 0, "MiniMax-H3 MP4 audio stream is empty"


def _run_t2va_request(client: OpenAIClientHandler, seed: int, model: str = MODEL) -> bytes:
    """Submit one synchronous T2VA request and return its MP4 body."""
    request_data = {
        "model": model,
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


def _run_t2va_requests(server: OmniServer, model: str) -> list[bytes]:
    client = OpenAIClientHandler(host=server.host, port=server.port)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(_run_t2va_request, client, seed, model) for seed in (1101, 1102)]
        return [future.result() for future in futures]


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.slow
@pytest.mark.parametrize("_hardware", [pytest.param(None, marks=H100_TWO_CARD_MARKS)])
def test_minimax_h3_bf16_hwr_dlo_dp2_t2va(
    _hardware: None,
    tmp_path: Path,
    run_level: str,
) -> None:
    """Cover baseline, populate, and exact-hit BF16 DLO DP2 AllGather."""
    original_model = get_model_prefix() + MODEL
    server_model = resolve_tiny_model_path(original_model) if run_level == "core_model" else original_model
    served_model_name = original_model if server_model != original_model else None
    hwr_root = tmp_path / "minimax-h3-bf16-hwr"

    with OmniServer(server_model, _server_args(served_model_name=served_model_name)) as server:
        baseline_videos = _run_t2va_requests(server, original_model)

    # A preferred cold startup publishes the finalized representation. Server
    # readiness means publication completed before the producer exits.
    with OmniServer(
        server_model,
        _hwr_server_args(hwr_root, "preferred", served_model_name=served_model_name),
    ):
        pass
    assert tuple(hwr_root.rglob("*.safetensors")), "preferred startup did not publish an HWR artifact"

    # Required mode cannot fall back. Reaching readiness proves every rank hit,
    # restored, and committed the same artifact before DLO AllGather setup.
    with OmniServer(
        server_model,
        _hwr_server_args(hwr_root, "required", served_model_name=served_model_name),
    ) as server:
        hwr_videos = _run_t2va_requests(server, original_model)

    for video in (*baseline_videos, *hwr_videos):
        assert_video_valid(video, width=WIDTH, height=HEIGHT, fps=FPS)
        _assert_audio_stream_present(video)
