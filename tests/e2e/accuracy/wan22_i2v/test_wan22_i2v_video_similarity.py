# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

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
    build_online_image_reference,
    online_timeout_seconds,
    probe_binary,
    probe_video,
    resolve_device_profile,
    resolve_similarity_thresholds,
    send_video_request_with_timeout,
    validate_image_source,
    video_artifact_dir,
)
from tests.e2e.accuracy.wan22_i2v.wan22_i2v_video_similarity_common import (
    FLOW_SHIFT,
    FPS,
    GUIDANCE_SCALE,
    GUIDANCE_SCALE_2,
    HEIGHT,
    MODEL_NAME,
    NEGATIVE_PROMPT,
    NUM_FRAMES,
    NUM_INFERENCE_STEPS,
    PROMPT,
    RABBIT_IMAGE_URL,
    SEED,
    SIMILARITY_THRESHOLDS_BY_DEVICE,
    SIZE,
    WIDTH,
)
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams

pytestmark = [pytest.mark.diffusion]

REPO_ROOT = Path(__file__).resolve().parents[4]
WORKSPACE_ROOT = REPO_ROOT.parent
RUNNER_PATH = Path(__file__).with_name("run_wan22_i2v_diffusers_cp.py")
RESULT_ROOT = Path(__file__).parent / "result"
VIDEO_TIMEOUT_SECONDS = 60 * 60
SERVER_CASES = [
    pytest.param(
        OmniServerParams(
            model=MODEL_NAME,
            server_args=[
                "--usp",
                "2",
                "--use-hsdp",
                "--hsdp-shard-size",
                "2",
            ],
            use_omni=True,
        ),
        id="wan22_i2v_usp2_hsdp2",
    )
]


def _build_diffusers_command(
    *,
    runner_path: Path,
    image_source: str,
    output_path: Path,
    metadata_path: Path,
) -> list[str]:
    return [
        sys.executable,
        str(runner_path),
        "--model",
        MODEL_NAME,
        "--image-source",
        image_source,
        "--prompt",
        PROMPT,
        "--negative-prompt",
        NEGATIVE_PROMPT,
        "--size",
        SIZE,
        "--fps",
        str(FPS),
        "--num-frames",
        str(NUM_FRAMES),
        "--guidance-scale",
        str(GUIDANCE_SCALE),
        "--guidance-scale-2",
        str(GUIDANCE_SCALE_2),
        "--flow-shift",
        str(FLOW_SHIFT),
        "--num-inference-steps",
        str(NUM_INFERENCE_STEPS),
        "--seed",
        str(SEED),
        "--output",
        str(output_path),
        "--metadata-output",
        str(metadata_path),
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

    return env


def _resolve_image_source(configured: str | None) -> str:
    configured = configured or RABBIT_IMAGE_URL
    candidate = Path(configured)
    if candidate.exists():
        return str(candidate.resolve())
    return configured


def _artifact_paths(image_source: str) -> tuple[Path, Path, Path]:
    artifact_dir = video_artifact_dir(RESULT_ROOT, image_source)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return (
        artifact_dir / "online.mp4",
        artifact_dir / "offline.mp4",
        artifact_dir / "offline_metadata.json",
    )


def _generate_online_video(
    *,
    omni_server,
    online_client,
    image_source: str,
    online_timeout_seconds_value: int,
) -> Path:
    online_path, _, _ = _artifact_paths(image_source)
    request_config = {
        "model": omni_server.model,
        "form_data": {
            "prompt": PROMPT,
            "negative_prompt": NEGATIVE_PROMPT,
            "size": SIZE,
            "fps": FPS,
            "num_frames": NUM_FRAMES,
            "guidance_scale": GUIDANCE_SCALE,
            "guidance_scale_2": GUIDANCE_SCALE_2,
            "flow_shift": FLOW_SHIFT,
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "seed": SEED,
        },
        "image_reference": build_online_image_reference(image_source),
    }
    online_video_bytes = send_video_request_with_timeout(
        online_client,
        request_config,
        timeout_seconds=online_timeout_seconds(online_timeout_seconds_value),
    )
    online_path.write_bytes(online_video_bytes)
    return online_path


def _generate_offline_video(*, image_source: str) -> tuple[Path, Path]:
    _, offline_path, offline_metadata_path = _artifact_paths(image_source)
    command = _build_diffusers_command(
        runner_path=RUNNER_PATH,
        image_source=image_source,
        output_path=offline_path,
        metadata_path=offline_metadata_path,
    )
    subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=_runner_env(),
        check=True,
        timeout=VIDEO_TIMEOUT_SECONDS,
    )
    return offline_path, offline_metadata_path


@pytest.mark.benchmark
@pytest.mark.full_model
@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_wan22_i2v_diffusers_offline_generates_video(
    wan22_i2v_image_source: str | None,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("Wan2.2 I2V diffusers offline test requires CUDA.")

    probe_binary("ffprobe")
    if not RUNNER_PATH.exists():
        raise AssertionError(f"Offline diffusers runner does not exist: {RUNNER_PATH}")

    image_source = _resolve_image_source(wan22_i2v_image_source)
    validate_image_source(image_source)
    offline_path, offline_metadata_path = _generate_offline_video(image_source=image_source)
    assert offline_path.exists(), f"Expected offline video artifact at {offline_path}"
    assert offline_metadata_path.exists(), f"Expected offline metadata artifact at {offline_metadata_path}"
    offline_metadata = probe_video(offline_path)
    assert_video_metadata(offline_metadata, width=WIDTH, height=HEIGHT, fps=FPS, frame_count=NUM_FRAMES)


@pytest.mark.benchmark
@pytest.mark.full_model
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_server", SERVER_CASES, indirect=True)
def test_wan22_i2v_online_serving_generates_video(
    omni_server,
    online_client,
    wan22_i2v_image_source: str | None,
    wan22_i2v_online_timeout_seconds: int,
) -> None:
    if not torch.cuda.is_available() or torch.accelerator.device_count() < 2:
        pytest.skip("Wan2.2 I2V similarity e2e test requires >= 2 CUDA GPUs.")

    probe_binary("ffprobe")
    image_source = _resolve_image_source(wan22_i2v_image_source)
    validate_image_source(image_source)
    online_path = _generate_online_video(
        omni_server=omni_server,
        online_client=online_client,
        image_source=image_source,
        online_timeout_seconds_value=wan22_i2v_online_timeout_seconds,
    )
    assert online_path.exists(), f"Expected online video artifact at {online_path}"
    online_metadata = probe_video(online_path)
    assert_video_metadata(online_metadata, width=WIDTH, height=HEIGHT, fps=FPS, frame_count=NUM_FRAMES)


@pytest.mark.benchmark
@pytest.mark.full_model
@hardware_test(res={"cuda": "H100"}, num_cards=2)
def test_wan22_i2v_serving_matches_diffusers_video_similarity(
    wan22_i2v_image_source: str | None,
) -> None:
    if not torch.cuda.is_available() or torch.accelerator.device_count() < 2:
        pytest.skip("Wan2.2 I2V similarity e2e test requires >= 2 CUDA GPUs.")

    probe_binary("ffmpeg")
    probe_binary("ffprobe")
    if not RUNNER_PATH.exists():
        raise AssertionError(f"Offline diffusers runner does not exist: {RUNNER_PATH}")

    image_source = _resolve_image_source(wan22_i2v_image_source)
    validate_image_source(image_source)
    online_path, offline_path, offline_metadata_path = _artifact_paths(image_source)

    if not online_path.exists():
        pytest.skip(f"Missing online artifact from prerequisite test: {online_path}")
    if not offline_path.exists() or not offline_metadata_path.exists():
        pytest.skip(f"Missing offline artifacts from prerequisite test: {offline_path}, {offline_metadata_path}")

    assert online_path.exists(), f"Expected online video artifact at {online_path}"
    assert offline_path.exists(), f"Expected offline video artifact at {offline_path}"
    assert offline_metadata_path.exists(), f"Expected offline metadata artifact at {offline_metadata_path}"

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
    device_profile = resolve_device_profile(profiles=SIMILARITY_THRESHOLDS_BY_DEVICE)
    thresholds = resolve_similarity_thresholds(SIMILARITY_THRESHOLDS_BY_DEVICE, device_profile)
    print(
        f"wan22_i2v similarity thresholds: device_profile={device_profile}, "
        f"ssim>={thresholds.ssim:.6f}, psnr>={thresholds.psnr:.6f}"
    )
    assert_video_similarity_metrics(
        label="wan22_i2v",
        online_path=online_path,
        offline_path=offline_path,
        ssim_threshold=thresholds.ssim,
        psnr_threshold=thresholds.psnr,
    )
    print(f"offline_metadata={offline_metadata_path}")
