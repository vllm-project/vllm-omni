# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest
import requests
import torch
from huggingface_hub import snapshot_download

from tests.e2e.accuracy.helpers import (
    assert_video_metadata,
    assert_video_similarity_metrics,
    probe_binary,
    probe_video,
    reset_artifact_dir,
)
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServer

pytestmark = [pytest.mark.benchmark, pytest.mark.diffusion, pytest.mark.full_model]

MODEL_REPO_ID = "MiniMaxAI/MiniMax-H3"
# The previously pinned revision was removed from the Hugging Face repository.
MODEL_REVISION = "main"
MODEL_ENV_VAR = "VLLM_TEST_MINIMAX_H3_FL2VA_MODEL"
# Keep the official asset link visible in the test report. Hugging Face's
# ``blob`` URL is an HTML page, so resolve it to the raw asset for download.
REFERENCE_VIDEO_URL = "https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/assets/t2va.mp4"
REFERENCE_VIDEO_SHA256 = "d66903241362e224085cc93f7a5e70fba6ab378d0ac6ff6af83acf2559849a42"

# This is the official reproducible 768p T2VA prompt associated with
# ``assets/t2va.mp4`` in the MiniMax H3 repository.
PROMPT = """integrated_multimodal_description: [Shot 1] Cinematic, medium wide shot, pushing in slowly. In the cavernous, dimly lit bridge of a starship, sleek metallic consoles with glowing amber displays flank a massive, curved observation window. A female captain, in her late 40s with an athletic build and short silver-streaked black hair, stands in the center midground. She wears a structured, high-collared dark navy military tunic with silver chest insignias. Her back is to the camera, silhouetted against the cool, ambient starlight pouring through the thick glass. She stands perfectly still with her hands clasped tightly behind her back. Outside the window, a massive armada of jagged, dark grey dreadnoughts hovers in tight formation against a deep purple space nebula. The fleet's massive rear thrusters begin to glow with an intense, escalating bright blue light. [Shot 2] At 00:04.500, the camera cuts to a close-up of the captain's face and shakes strongly. The brilliant blue-white light from the fleet's gathering energy reflects vividly in her dark eyes. Suddenly, a blinding white flash floods through the window, completely washing out the background as the fleet jumps to hyperspace. The sheer spatial force violently jolts the bridge, causing the captain from Shot 1 to stagger slightly forward, her shoulders tensing as she visibly braces herself against the physical tremors. As the intense white light fades abruptly, leaving only the dim, empty expanse of the purple nebula reflected on her starkly lit skin, her jaw clenches, and she slowly closes her eyes in the newly emptied space.
overall_soundscape: A low, resonant hum of the ship's ambient life support systems serves as the baseline, soon drowned out by an audible, escalating, high-pitched electronic whine as the fleet outside charges its hyperdrives. A massive, deafening, bass-heavy boom and sharp crackle erupts during the blinding flash, accompanied by the loud metallic creaking, rattling, and deep thuds of the bridge's bulkheads vibrating under immense physical stress. The intense roaring impact then cuts abruptly back to a hollow, echoing room tone, leaving only the faint, steady hum of the isolated bridge.
non_diegetic_music: Cinematic space-opera orchestral score, slow tempo, featuring a solitary, mournful French horn melody over deep, sustained string dissonances that build rapidly in volume and intensity, swelling to a massive orchestral peak before snapping immediately into silence right after the jump."""

WIDTH = 1344
HEIGHT = 768
FPS = 24
NUM_FRAMES = 243
NUM_INFERENCE_STEPS = 50
FLOW_SHIFT = 12.0
AUDIO_FLOW_SHIFT = 3.0
DURATION_SECONDS = 10.0
SEED = 0
SSIM_THRESHOLD = 0.82
PSNR_THRESHOLD = 20.0
REQUEST_TIMEOUT_SECONDS = 60 * 60


def _model_name() -> str:
    configured = os.environ.get(MODEL_ENV_VAR)
    if configured:
        return configured

    snapshot_root = snapshot_download(
        repo_id=MODEL_REPO_ID,
        revision=MODEL_REVISION,
        allow_patterns=["FL2VA/**"],
    )
    return str(Path(snapshot_root) / "FL2VA")


def _server_args() -> list[str]:
    return [
        "--trust-remote-code",
        "--num-gpus",
        "4",
        "--usp",
        "4",
        "--ring",
        "1",
        "--use-hsdp",
        "--hsdp-shard-size",
        "4",
        "--text-encoder-tp-size",
        "4",
        "--vae-patch-parallel-size",
        "4",
        "--vae-parallel-mode",
        "tile",
        "--vae-use-tiling",
        "--diffusion-attention-backend",
        "FLASH_ATTN",
        "--stage-init-timeout",
        "1800",
        "--init-timeout",
        "1800",
    ]


def _download_reference_video(output_dir: Path) -> Path:
    reference_path = output_dir / "reference.mp4"
    download_url = REFERENCE_VIDEO_URL.replace("/blob/", "/resolve/", 1)
    response = requests.get(download_url, timeout=120)
    response.raise_for_status()
    content = response.content
    digest = hashlib.sha256(content).hexdigest()
    assert digest == REFERENCE_VIDEO_SHA256, (
        f"Official MiniMax H3 T2VA reference changed: got {digest}, expected {REFERENCE_VIDEO_SHA256}"
    )
    reference_path.write_bytes(content)
    return reference_path


def _probe_audio(path: Path) -> dict[str, str | int]:
    probe_binary("ffprobe")
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name,sample_rate,channels",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    streams = json.loads(result.stdout).get("streams", [])
    assert len(streams) == 1, f"Expected one audio stream in {path}, got {streams}"
    stream = streams[0]
    return {
        "codec_name": str(stream["codec_name"]),
        "sample_rate": int(stream["sample_rate"]),
        "channels": int(stream["channels"]),
    }


@hardware_test(res={"cuda": "H100"}, num_cards=4)
def test_minimax_h3_t2va_matches_official_reference(
    accuracy_artifact_root: Path,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("MiniMax H3 T2VA accuracy test requires CUDA.")

    probe_binary("ffmpeg")
    probe_binary("ffprobe")
    output_dir = reset_artifact_dir(accuracy_artifact_root / "minimax_h3_t2va")
    reference_path = _download_reference_video(output_dir)
    generated_path = output_dir / "vllm_omni.mp4"

    server_env = {
        "FLASHINFER_DISABLE_VERSION_CHECK": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_OMNI_VIDEO_SYNC_TIMEOUT": str(REQUEST_TIMEOUT_SECONDS),
        "VLLM_OMNI_STORAGE_PATH": str(output_dir / "storage"),
    }
    request_data = {
        "prompt": PROMPT,
        "width": str(WIDTH),
        "height": str(HEIGHT),
        "fps": str(FPS),
        "num_inference_steps": str(NUM_INFERENCE_STEPS),
        "flow_shift": str(FLOW_SHIFT),
        "seed": str(SEED),
        "extra_params": json.dumps(
            {
                "task": "t2va",
                "duration": DURATION_SECONDS,
                "aspect_ratio": "16:9",
                "audio_flow_shift": AUDIO_FLOW_SHIFT,
            }
        ),
    }

    with OmniServer(
        _model_name(),
        _server_args(),
        env_dict=server_env,
        use_omni=True,
    ) as server:
        response = requests.post(
            f"http://{server.host}:{server.port}/v1/videos/sync",
            data=request_data,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

    response.raise_for_status()
    assert response.headers["content-type"].startswith("video/mp4")
    generated_path.write_bytes(response.content)

    reference_metadata = probe_video(reference_path)
    generated_metadata = probe_video(generated_path)
    assert reference_metadata == generated_metadata
    assert_video_metadata(
        generated_metadata,
        width=WIDTH,
        height=HEIGHT,
        fps=FPS,
        frame_count=NUM_FRAMES,
    )

    expected_audio = {"codec_name": "aac", "sample_rate": 32000, "channels": 2}
    assert _probe_audio(reference_path) == expected_audio
    assert _probe_audio(generated_path) == expected_audio

    assert_video_similarity_metrics(
        label="minimax_h3_t2va_official_reference",
        online_path=generated_path,
        offline_path=reference_path,
        ssim_threshold=SSIM_THRESHOLD,
        psnr_threshold=PSNR_THRESHOLD,
    )
