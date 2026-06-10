from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tests.e2e.accuracy.helpers import (
    assert_video_metadata,
    assert_video_similarity_metrics,
    probe_binary,
    probe_video,
    send_video_request_with_timeout,
)
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams

pytestmark = [pytest.mark.diffusion, pytest.mark.full_model]

MODEL_NAME = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"
PROMPT = "A cat walks on the grass, realistic style."
WIDTH = 832
HEIGHT = 480
FPS = 24
NUM_FRAMES = 33
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 6.0
FLOW_SHIFT = 5.0
SEED = 42
SSIM_THRESHOLD = 0.94
PSNR_THRESHOLD = 28.0

REPO_ROOT = Path(__file__).resolve().parents[4]
WORKSPACE_ROOT = REPO_ROOT.parent
RUNNER_PATH = REPO_ROOT / "examples" / "offline_inference" / "text_to_video" / "text_to_video.py"
RESULT_ROOT = Path(__file__).parent / "result"
ARTIFACT_DIR = RESULT_ROOT / "cat_grass"
VIDEO_TIMEOUT_SECONDS = 60 * 60
SERVER_CASES = [
    pytest.param(
        OmniServerParams(
            model=MODEL_NAME,
            server_args=[
                "--flow-shift",
                str(FLOW_SHIFT),
            ],
            env_dict={"VLLM_OMNI_STORAGE_PATH": str(RESULT_ROOT / "storage")},
            use_omni=True,
        ),
        id="hunyuanvideo15_t2v_default",
    )
]


def _runner_env() -> dict[str, str]:
    env = os.environ.copy()
    pythonpath_parts = [
        str(REPO_ROOT),
        str(WORKSPACE_ROOT / "diffusers" / "src"),
    ]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    return env


def _artifact_paths() -> tuple[Path, Path]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACT_DIR / "online.mp4", ARTIFACT_DIR / "offline.mp4"


def _build_offline_command(*, output_path: Path) -> list[str]:
    return [
        sys.executable,
        str(RUNNER_PATH),
        "--model",
        MODEL_NAME,
        "--prompt",
        PROMPT,
        "--height",
        str(HEIGHT),
        "--width",
        str(WIDTH),
        "--num-frames",
        str(NUM_FRAMES),
        "--num-inference-steps",
        str(NUM_INFERENCE_STEPS),
        "--guidance-scale",
        str(GUIDANCE_SCALE),
        "--flow-shift",
        str(FLOW_SHIFT),
        "--fps",
        str(FPS),
        "--seed",
        str(SEED),
        "--output",
        str(output_path),
    ]


def _generate_offline_video() -> Path:
    _, offline_path = _artifact_paths()
    subprocess.run(
        _build_offline_command(output_path=offline_path),
        cwd=REPO_ROOT,
        env=_runner_env(),
        check=True,
        timeout=VIDEO_TIMEOUT_SECONDS,
    )
    return offline_path


def _generate_online_video(*, omni_server, openai_client, timeout_seconds: int) -> Path:
    online_path, _ = _artifact_paths()
    request_config = {
        "model": omni_server.model,
        "form_data": {
            "prompt": PROMPT,
            "height": HEIGHT,
            "width": WIDTH,
            "fps": FPS,
            "num_frames": NUM_FRAMES,
            "guidance_scale": GUIDANCE_SCALE,
            "flow_shift": FLOW_SHIFT,
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "seed": SEED,
        },
    }
    online_video_bytes = send_video_request_with_timeout(
        openai_client,
        request_config,
        timeout_seconds=timeout_seconds,
    )
    online_path.write_bytes(online_video_bytes)
    return online_path


@pytest.mark.benchmark
@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_hunyuanvideo15_t2v_diffusers_offline_generates_video() -> None:
    if not torch.cuda.is_available():
        pytest.skip("HunyuanVideo-1.5 T2V offline accuracy test requires CUDA.")

    probe_binary("ffprobe")
    if not RUNNER_PATH.exists():
        raise AssertionError(f"Offline runner does not exist: {RUNNER_PATH}")

    offline_path = _generate_offline_video()
    assert offline_path.exists(), f"Expected offline video artifact at {offline_path}"
    offline_metadata = probe_video(offline_path)
    assert_video_metadata(offline_metadata, width=WIDTH, height=HEIGHT, fps=FPS, frame_count=NUM_FRAMES)


@pytest.mark.benchmark
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_CASES, indirect=True)
def test_hunyuanvideo15_t2v_online_serving_generates_video(
    omni_server,
    openai_client,
    hunyuanvideo15_online_timeout_seconds: int,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("HunyuanVideo-1.5 T2V online accuracy test requires CUDA.")

    probe_binary("ffprobe")
    online_path = _generate_online_video(
        omni_server=omni_server,
        openai_client=openai_client,
        timeout_seconds=hunyuanvideo15_online_timeout_seconds,
    )
    assert online_path.exists(), f"Expected online video artifact at {online_path}"
    online_metadata = probe_video(online_path)
    assert_video_metadata(online_metadata, width=WIDTH, height=HEIGHT, fps=FPS, frame_count=NUM_FRAMES)


@pytest.mark.benchmark
@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_hunyuanvideo15_t2v_serving_matches_offline_video_similarity() -> None:
    if not torch.cuda.is_available():
        pytest.skip("HunyuanVideo-1.5 T2V video similarity test requires CUDA.")

    probe_binary("ffmpeg")
    probe_binary("ffprobe")
    online_path, offline_path = _artifact_paths()
    if not online_path.exists():
        pytest.skip(f"Missing online artifact from prerequisite test: {online_path}")
    if not offline_path.exists():
        pytest.skip(f"Missing offline artifact from prerequisite test: {offline_path}")

    online_metadata = probe_video(online_path)
    offline_metadata = probe_video(offline_path)
    assert online_metadata == offline_metadata, (
        f"Video metadata mismatch:\n"
        f"online={online_metadata}\n"
        f"offline={offline_metadata}\n"
        f"online_path={online_path}\n"
        f"offline_path={offline_path}"
    )
    assert_video_metadata(online_metadata, width=WIDTH, height=HEIGHT, fps=FPS, frame_count=NUM_FRAMES)
    assert_video_similarity_metrics(
        label="hunyuanvideo15_t2v",
        online_path=online_path,
        offline_path=offline_path,
        ssim_threshold=SSIM_THRESHOLD,
        psnr_threshold=PSNR_THRESHOLD,
    )
