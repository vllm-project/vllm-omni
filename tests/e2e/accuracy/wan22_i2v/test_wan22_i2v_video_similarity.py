from __future__ import annotations

import os
import subprocess
import sys
from base64 import b64decode
from pathlib import Path

import pytest
import requests
import torch
from diffusers import UniPCMultistepScheduler
from PIL import Image

from tests.e2e.accuracy.helpers import (
    assert_video_metadata,
    assert_video_similarity_metrics,
    video_artifact_dir,
)
from tests.e2e.accuracy.helpers import (
    build_online_image_reference as _build_online_image_reference,
)
from tests.e2e.accuracy.helpers import (
    online_timeout_seconds as _online_timeout_seconds,
)
from tests.e2e.accuracy.helpers import (
    parse_psnr_score as _parse_psnr_score,
)
from tests.e2e.accuracy.helpers import (
    parse_ssim_score as _parse_ssim_score,
)
from tests.e2e.accuracy.helpers import (
    parse_video_metadata as _parse_video_metadata,
)
from tests.e2e.accuracy.helpers import (
    probe_binary as _probe_binary,
)
from tests.e2e.accuracy.helpers import (
    probe_video as _probe_video,
)
from tests.e2e.accuracy.helpers import (
    send_video_request_with_timeout as _send_video_request_with_timeout,
)
from tests.e2e.accuracy.helpers import (
    validate_image_source as _validate_image_source,
)
from tests.e2e.accuracy.wan22_i2v.run_wan22_i2v_diffusers_cp import (
    _configure_scheduler,
    _ensure_wan_ftfy_fallback,
    _IdentityFtfy,
    _offline_cuda_device,
    _resize_to_target,
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
    PSNR_THRESHOLD,
    RABBIT_IMAGE_URL,
    SEED,
    SIZE,
    SSIM_THRESHOLD,
    WIDTH,
)
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams

pytestmark = [pytest.mark.diffusion, pytest.mark.full_model]


def test_parse_video_metadata_extracts_dimensions_and_fps() -> None:
    payload = {
        "streams": [
            {
                "width": 832,
                "height": 480,
                "avg_frame_rate": "16/1",
                "nb_read_frames": "93",
            }
        ]
    }

    metadata = _parse_video_metadata(payload)

    assert metadata["width"] == 832
    assert metadata["height"] == 480
    assert metadata["fps"] == 16.0
    assert metadata["frame_count"] == 93


def test_parse_ssim_summary_extracts_all_score() -> None:
    output = """
    [Parsed_ssim_0 @ 000001] SSIM Y:0.971903 (15.512007) U:0.965077 (14.569044) V:0.962414 (14.252637) All:0.968311 (15.035654)
    """

    assert _parse_ssim_score(output) == 0.968311


def test_parse_psnr_summary_extracts_average_score() -> None:
    output = """
    [Parsed_psnr_0 @ 000001] PSNR y:32.670157 u:31.844621 v:31.513839 average:32.148004 min:31.563744 max:33.201457
    """

    assert _parse_psnr_score(output) == 32.148004


def test_build_diffusers_command_uses_python_runner_path(tmp_path: Path) -> None:
    runner_path = tmp_path / "run_wan22_i2v_diffusers_cp.py"
    command = _build_diffusers_command(
        runner_path=runner_path,
        image_source=RABBIT_IMAGE_URL,
        output_path=tmp_path / "offline.mp4",
        metadata_path=tmp_path / "offline.json",
    )

    assert command[:2] == [
        sys.executable,
        str(runner_path),
    ]
    assert "--output" in command
    assert "--metadata-output" in command


def test_resolve_image_source_prefers_existing_local_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image_path = tmp_path / "rabbit.png"
    Image.new("RGB", (8, 8), color=(255, 128, 64)).save(image_path)

    assert _resolve_image_source(str(image_path)) == str(image_path.resolve())


def test_build_online_image_reference_uses_data_url_for_local_path(tmp_path: Path) -> None:
    image_path = tmp_path / "rabbit.png"
    Image.new("RGB", (4, 2), color=(10, 20, 30)).save(image_path)

    reference = _build_online_image_reference(str(image_path))

    assert reference.startswith("data:image/png;base64,")
    encoded = reference.split(",", 1)[1]
    assert len(b64decode(encoded)) > 0


def test_send_video_request_with_timeout_uses_requested_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, str]:
            return {"id": "video_123"}

    class DummyClient:
        def __init__(self) -> None:
            self.run_level = "core_model"
            self.wait_args: tuple[str, int, int] | None = None

        def _build_url(self, path: str) -> str:
            return f"http://localhost:8000{path}"

        def _wait_until_video_completed(
            self,
            video_id: str,
            poll_interval_seconds: int = 2,
            timeout_seconds: int = 300,
        ) -> None:
            self.wait_args = (video_id, poll_interval_seconds, timeout_seconds)

        def _download_video_content(self, video_id: str) -> bytes:
            return b"video-bytes"

    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: DummyResponse())

    client = DummyClient()
    result = _send_video_request_with_timeout(
        client,
        {
            "form_data": {"prompt": "test"},
            "image_reference": RABBIT_IMAGE_URL,
        },
        timeout_seconds=1200,
    )

    assert result == b"video-bytes"
    assert client.wait_args == ("video_123", 2, 1200)


def test_online_timeout_defaults_to_1200() -> None:
    assert _online_timeout_seconds(None) == 1200


def test_artifact_dir_is_under_repo_result_folder(tmp_path: Path) -> None:
    artifact_dir = _artifact_dir(str(tmp_path / "rabbit.png"))

    assert artifact_dir.parent == Path(__file__).parent / "result"
    assert artifact_dir.name.startswith("rabbit-")


def test_offline_cuda_device_uses_indexed_cuda_device() -> None:
    assert _offline_cuda_device() == torch.device("cuda:0")


def test_ensure_wan_ftfy_fallback_sets_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    from diffusers.pipelines.wan import pipeline_wan_i2v as wan_i2v_module

    monkeypatch.delattr(wan_i2v_module, "ftfy", raising=False)
    _ensure_wan_ftfy_fallback()

    assert hasattr(wan_i2v_module, "ftfy")
    assert isinstance(wan_i2v_module.ftfy, _IdentityFtfy)
    assert wan_i2v_module.ftfy.fix_text("abc") == "abc"


def test_resize_to_target_matches_requested_dimensions() -> None:
    image = Image.new("RGB", (100, 50), color=(1, 2, 3))

    resized = _resize_to_target(image, width=832, height=480)

    assert resized.size == (832, 480)


def test_configure_scheduler_switches_to_unipc() -> None:
    class DummyPipe:
        def __init__(self) -> None:
            self.scheduler = UniPCMultistepScheduler(num_train_timesteps=1000, flow_shift=1.0)

    pipe = DummyPipe()
    _configure_scheduler(pipe, flow_shift=5.0)

    assert isinstance(pipe.scheduler, UniPCMultistepScheduler)
    assert pipe.scheduler.config.flow_shift == 5.0


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
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    return env


def _resolve_image_source(configured: str | None) -> str:
    configured = configured or RABBIT_IMAGE_URL
    candidate = Path(configured)
    if candidate.exists():
        return str(candidate.resolve())
    return configured


def _artifact_dir(image_source: str) -> Path:
    return video_artifact_dir(RESULT_ROOT, image_source)


def _artifact_paths(image_source: str) -> tuple[Path, Path, Path]:
    artifact_dir = _artifact_dir(image_source)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return (
        artifact_dir / "online.mp4",
        artifact_dir / "offline.mp4",
        artifact_dir / "offline_metadata.json",
    )


def _generate_online_video(
    *,
    omni_server,
    openai_client,
    image_source: str,
    online_timeout_seconds: int,
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
        "image_reference": _build_online_image_reference(image_source),
    }
    online_video_bytes = _send_video_request_with_timeout(
        openai_client,
        request_config,
        timeout_seconds=_online_timeout_seconds(online_timeout_seconds),
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
@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_wan22_i2v_diffusers_offline_generates_video(
    wan22_i2v_image_source: str | None,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("Wan2.2 I2V diffusers offline test requires CUDA.")

    _probe_binary("ffprobe")
    if not RUNNER_PATH.exists():
        raise AssertionError(f"Offline diffusers runner does not exist: {RUNNER_PATH}")

    image_source = _resolve_image_source(wan22_i2v_image_source)
    _validate_image_source(image_source)
    offline_path, offline_metadata_path = _generate_offline_video(image_source=image_source)
    assert offline_path.exists(), f"Expected offline video artifact at {offline_path}"
    assert offline_metadata_path.exists(), f"Expected offline metadata artifact at {offline_metadata_path}"
    offline_metadata = _probe_video(offline_path)
    assert_video_metadata(offline_metadata, width=WIDTH, height=HEIGHT, fps=FPS, frame_count=NUM_FRAMES)


@pytest.mark.benchmark
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_server", SERVER_CASES, indirect=True)
def test_wan22_i2v_online_serving_generates_video(
    omni_server,
    openai_client,
    wan22_i2v_image_source: str | None,
    wan22_i2v_online_timeout_seconds: int,
) -> None:
    if not torch.cuda.is_available() or torch.accelerator.device_count() < 2:
        pytest.skip("Wan2.2 I2V similarity e2e test requires >= 2 CUDA GPUs.")

    _probe_binary("ffprobe")
    image_source = _resolve_image_source(wan22_i2v_image_source)
    _validate_image_source(image_source)
    online_path = _generate_online_video(
        omni_server=omni_server,
        openai_client=openai_client,
        image_source=image_source,
        online_timeout_seconds=wan22_i2v_online_timeout_seconds,
    )
    assert online_path.exists(), f"Expected online video artifact at {online_path}"
    online_metadata = _probe_video(online_path)
    assert_video_metadata(online_metadata, width=WIDTH, height=HEIGHT, fps=FPS, frame_count=NUM_FRAMES)


@pytest.mark.benchmark
@hardware_test(res={"cuda": "H100"}, num_cards=2)
def test_wan22_i2v_serving_matches_diffusers_video_similarity(
    wan22_i2v_image_source: str | None,
) -> None:
    if not torch.cuda.is_available() or torch.accelerator.device_count() < 2:
        pytest.skip("Wan2.2 I2V similarity e2e test requires >= 2 CUDA GPUs.")

    _probe_binary("ffmpeg")
    _probe_binary("ffprobe")
    if not RUNNER_PATH.exists():
        raise AssertionError(f"Offline diffusers runner does not exist: {RUNNER_PATH}")

    image_source = _resolve_image_source(wan22_i2v_image_source)
    _validate_image_source(image_source)
    online_path, offline_path, offline_metadata_path = _artifact_paths(image_source)

    if not online_path.exists():
        pytest.skip(f"Missing online artifact from prerequisite test: {online_path}")
    if not offline_path.exists() or not offline_metadata_path.exists():
        pytest.skip(f"Missing offline artifacts from prerequisite test: {offline_path}, {offline_metadata_path}")

    assert online_path.exists(), f"Expected online video artifact at {online_path}"
    assert offline_path.exists(), f"Expected offline video artifact at {offline_path}"
    assert offline_metadata_path.exists(), f"Expected offline metadata artifact at {offline_metadata_path}"

    online_metadata = _probe_video(online_path)
    offline_metadata = _probe_video(offline_path)
    assert online_metadata == offline_metadata, (
        f"Video metadata mismatch:\n"
        f"online={online_metadata}\n"
        f"offline={offline_metadata}\n"
        f"online_path={online_path}\n"
        f"offline_path={offline_path}"
    )
    assert_video_metadata(online_metadata, width=WIDTH, height=HEIGHT, fps=FPS, frame_count=NUM_FRAMES)
    assert_video_similarity_metrics(
        label="wan22_i2v",
        online_path=online_path,
        offline_path=offline_path,
        ssim_threshold=SSIM_THRESHOLD,
        psnr_threshold=PSNR_THRESHOLD,
    )
    print(f"offline_metadata={offline_metadata_path}")
