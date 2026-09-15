# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Shared request and server helpers for MiniMax-H3 L3 tests."""

from __future__ import annotations

import base64
import concurrent.futures
import io
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import av
import requests

from tests.helpers.assertions import assert_video_valid
from tests.helpers.media import generate_synthetic_image
from tests.helpers.runtime import OmniServerParams, OpenAIClientHandler

# DLO starts worker processes and uses shared-memory queues.  Always select
# spawn for these cases so a parent fork cannot inherit stale resource-tracker
# state from a preceding server lifecycle.
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL = os.environ.get("VLLM_TEST_MINIMAX_H3_MODEL", "MiniMaxAI/MiniMax-H3")
WIDTH = 1344
HEIGHT = 768
FPS = 24
NUM_INFERENCE_STEPS = 4
# Keep a failed collective/preprocessing regression bounded well below the
# per-file Buildkite timeout.  Healthy H100 runs complete in under two minutes
# after startup; ten minutes leaves ample room for a cold compile.
REQUEST_TIMEOUT_SECONDS = 600

# FastH3's four-step preview is a load-time fusion adapter.  Its model-level
# deltas cannot use the dynamic PEFT LoRA manager, so this test deliberately
# skips on branches that predate PR #6714 instead of starting a base-H3 server.
_REPO_ROOT = Path(__file__).resolve().parents[4]
FASTH3_SUPPORTED = (_REPO_ROOT / "vllm_omni/diffusion/models/minimax_h3/fasth3.py").is_file()
# The Turbo artifact is handled by the MiniMax-specific loader introduced in
# PR #6550; the generic diffusion LoRA manager cannot parse that checkpoint.
TURBO_SUPPORTED = (_REPO_ROOT / "vllm_omni/diffusion/models/minimax_h3/lora.py").is_file()
FASTH3_WIDTH = 1024
FASTH3_HEIGHT = 576
FASTH3_DURATION = 4.4

DLO_SERVER_ARGS = [
    "--trust-remote-code",
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

_FL2VA_IMAGE = base64.b64decode(generate_synthetic_image(WIDTH, HEIGHT, seed=42)["base64"])
# Keep the DLO/DP2 Ref2VA wave on the image-only path.  The separate video
# reference path performs per-request ffmpeg/Qwen video preprocessing and has
# repeatedly stalled this merge wave before the first pipeline log; it remains
# covered by the Ref2VA contract/accuracy suites.
_REF2VA_IMAGE = base64.b64decode(generate_synthetic_image(512, 288, seed=43)["base64"])


def resolve_turbo_lora() -> str | None:
    """Resolve the legacy Turbo LoRA artifact without network at collection."""
    configured = os.environ.get("VLLM_TEST_MINIMAX_H3_TURBO_LORA")
    if configured and Path(configured).is_file():
        return configured

    try:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(
            repo_id="lightx2v/Minimax-h3-Turbo",
            filename="minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors",
            local_files_only=True,
        )
    except Exception:
        return None


def resolve_fasth3_lora() -> str | None:
    """Resolve the dense/data-free FastH3 artifact without network at collection."""
    configured = os.environ.get("VLLM_TEST_MINIMAX_H3_FASTH3_LORA")
    if configured and Path(configured).is_file():
        return configured

    try:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(
            repo_id="FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA",
            filename="dense-datafree/adapter_model.safetensors",
            local_files_only=True,
        )
    except Exception:
        # Buildkite downloads the artifact before pytest.  Local collection
        # must stay offline-friendly when that optional cache is absent.
        return None


FASTH3_LORA = resolve_fasth3_lora()
TURBO_LORA = resolve_turbo_lora()


def _assert_audio_stream_present(video: bytes) -> None:
    """Assert that the generated MP4 contains decodable audio samples."""
    with av.open(io.BytesIO(video)) as container:
        audio_streams = [stream for stream in container.streams if stream.type == "audio"]
        assert audio_streams, "MiniMax-H3 MP4 has no audio stream"
        audio_frame = next(container.decode(audio=0), None)
        assert audio_frame is not None and audio_frame.samples > 0, "MiniMax-H3 MP4 audio stream is empty"


def assert_h3_video(video: bytes, *, width: int, height: int) -> None:
    """Check the common H3 video/audio response contract."""
    assert_video_valid(video, width=width, height=height, fps=FPS)
    _assert_audio_stream_present(video)


def post_sync(
    client: OpenAIClientHandler,
    form_data: dict[str, str],
    files: Any = None,
) -> bytes:
    """Submit one synchronous video request and return its MP4 body."""
    response = requests.post(
        f"{client.base_url.rstrip('/')}/v1/videos/sync",
        data=form_data,
        files=files,
        headers={"Accept": "video/mp4"},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    assert response.headers.get("content-type", "").startswith("video/mp4")
    assert response.content, "MiniMax-H3 returned an empty video body"
    return response.content


def _h3_form(task: str, seed: int) -> dict[str, str]:
    return {
        "model": MODEL,
        "prompt": (
            "A cinematic live-action scene with a clear subject moving naturally; "
            "the atmosphere includes synchronized environmental sound."
        ),
        "width": str(WIDTH),
        "height": str(HEIGHT),
        "fps": str(FPS),
        "num_inference_steps": str(NUM_INFERENCE_STEPS),
        "flow_shift": "12",
        "seed": str(seed),
        "extra_params": json.dumps(
            {
                "task": task,
                "duration": 4.0,
                "aspect_ratio": "16:9",
                "audio_flow_shift": 3.0,
            },
            separators=(",", ":"),
        ),
    }


def run_fl2va(client: OpenAIClientHandler, seed: int) -> bytes:
    return post_sync(
        client,
        _h3_form("fl2va", seed),
        files=fl2va_files(),
    )


def run_ref2va(client: OpenAIClientHandler, seed: int) -> bytes:
    return post_sync(
        client,
        _h3_form("ref2va", seed),
        files=[("input_references", ("reference.jpg", io.BytesIO(_REF2VA_IMAGE), "image/jpeg"))],
    )


def fl2va_files() -> dict[str, tuple[str, io.BytesIO, str]]:
    """Return a fresh multipart first-frame payload for an FL2VA request."""
    return {"input_reference": ("first_frame.jpg", io.BytesIO(_FL2VA_IMAGE), "image/jpeg")}


def run_dlo_wave(
    client: OpenAIClientHandler,
    request_fn: Callable[[OpenAIClientHandler, int], bytes],
) -> list[bytes]:
    """Run one complete two-request DLO/DP2 wave concurrently."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        return list(executor.map(request_fn, (client, client), (2101, 2102)))


def dlo_params(task_type: str) -> OmniServerParams:
    return OmniServerParams(
        model=MODEL,
        server_args=[*DLO_SERVER_ARGS, "--task-type", task_type],
        stage_init_timeout=1800,
        init_timeout=1800,
    )


def turbo_params(lora_path: str | None) -> OmniServerParams:
    """Build the two-card TP2 + DLO-no-AllGather Turbo layout."""
    return OmniServerParams(
        model=MODEL,
        server_args=[
            "--trust-remote-code",
            "--task-type",
            "fl2va",
            "--num-gpus",
            "2",
            "--tensor-parallel-size",
            "2",
            "--usp",
            "1",
            "--ring",
            "1",
            "--text-encoder-tp-size",
            "2",
            "--vae-patch-parallel-size",
            "2",
            "--vae-parallel-mode",
            "tile",
            "--vae-use-tiling",
            "--enable-distributed-layerwise-offload",
            "--dlo-no-use-allgather",
            "--lora-backend",
            "peft",
            "--lora-path",
            lora_path or "",
        ],
        stage_init_timeout=1800,
        init_timeout=1800,
    )


def turbo_form(seed: int) -> dict[str, str]:
    """Build the official five-sigma-point Turbo FL2VA request."""
    return {
        "model": MODEL,
        "prompt": (
            "A man stands beside a yellow car at night; the car drives away "
            "as he follows it with his eyes and begins singing sadly."
        ),
        "width": str(WIDTH),
        "height": str(HEIGHT),
        "fps": str(FPS),
        # The public Turbo artifact has four denoiser evaluations but its API
        # contract requests five sigma points.
        "num_inference_steps": "5",
        "flow_shift": "6",
        "seed": str(seed),
        "extra_params": json.dumps(
            {"task": "fl2va", "duration": FASTH3_DURATION, "aspect_ratio": "16:9", "audio_flow_shift": 3.0},
            separators=(",", ":"),
        ),
        "lora": json.dumps(
            {"name": "h3-turbo-v1.0", "path": TURBO_LORA, "scale": 1.0},
            separators=(",", ":"),
        ),
    }


def fasth3_params(lora_path: str | None) -> OmniServerParams:
    """Build the requested two-card FastH3 parallel layout.

    HSDP owns DiT sharding (the model tensor-parallel degree stays at its
    default 1), USP2 shards the sequence, text-encoder TP2 shards Qwen3-VL,
    and VAE patch parallel 2 is the project's VPP2 setting.
    """
    return OmniServerParams(
        model=MODEL,
        server_args=[
            "--trust-remote-code",
            "--task-type",
            "fl2va",
            "--num-gpus",
            "2",
            "--usp",
            "2",
            "--ring",
            "1",
            "--use-hsdp",
            "--hsdp-shard-size",
            "2",
            "--text-encoder-tp-size",
            "2",
            "--vae-patch-parallel-size",
            "2",
            "--vae-parallel-mode",
            "tile",
            "--vae-use-tiling",
            "--lora-path",
            lora_path or "",
        ],
        stage_init_timeout=1800,
        init_timeout=1800,
    )


def fasth3_form(seed: int) -> dict[str, str]:
    """Build the four-step T2VA request accepted by the FastH3 preview."""
    return {
        "model": MODEL,
        "prompt": (
            "A cinematic live-action scene with a clear subject moving naturally; "
            "the atmosphere includes synchronized environmental sound."
        ),
        "width": str(FASTH3_WIDTH),
        "height": str(FASTH3_HEIGHT),
        "fps": str(FPS),
        "num_inference_steps": "4",
        "seed": str(seed),
        # FastH3 is fused at startup, so a request-level `lora` field is both
        # unnecessary and rejected by the load-time adapter contract.
        "extra_params": json.dumps(
            {
                "task": "t2va",
                "duration": FASTH3_DURATION,
                "aspect_ratio": "16:9",
            },
            separators=(",", ":"),
        ),
    }


__all__ = [
    "FASTH3_LORA",
    "FASTH3_SUPPORTED",
    "FASTH3_DURATION",
    "FASTH3_HEIGHT",
    "FASTH3_WIDTH",
    "FPS",
    "MODEL",
    "assert_h3_video",
    "dlo_params",
    "fasth3_form",
    "fasth3_params",
    "fl2va_files",
    "post_sync",
    "run_dlo_wave",
    "run_fl2va",
    "run_ref2va",
    "TURBO_LORA",
    "TURBO_SUPPORTED",
    "turbo_form",
    "turbo_params",
]
