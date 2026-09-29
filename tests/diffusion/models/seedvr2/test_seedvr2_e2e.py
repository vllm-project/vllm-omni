# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""End-to-end native video restoration and the HTTP long-video route.

Set VLLM_TEST_SEEDVR2_MODEL_DIR to the directory containing the DiT, VAE,
conditioning tensor, and model_index.json to run the complete video test.
Each rank exits normally; the launcher does not terminate workers.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from fractions import Fraction
from pathlib import Path

import pytest

pytestmark = [pytest.mark.local_model, pytest.mark.cuda, pytest.mark.diffusion, pytest.mark.parallel]
MODEL_DIR_ENV = "VLLM_TEST_SEEDVR2_MODEL_DIR"
# Two model windows with one four-frame seam, at the smallest legal frame size.
LONG_FRAMES = 20
LONG_SIZE = 64
FPS = 24


@pytest.mark.skipif(not os.environ.get(MODEL_DIR_ENV), reason=f"set {MODEL_DIR_ENV} to an authorized model directory")
@pytest.mark.parametrize("method", ["lab", "wavelet", "adain", "none"])
def test_seedvr2_native_video_e2e(method: str) -> None:
    import numpy as np
    import torch

    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    model_dir = Path(os.environ[MODEL_DIR_ENV]).resolve()
    for name in ("seedvr2_ema_3b_fp16.safetensors", "ema_vae_fp16.safetensors", "pos_emb.pt", "model_index.json"):
        assert (model_dir / name).is_file(), f"Missing SeedVR2 model file: {name}"
    frames = torch.rand(5, 3, 64, 112, generator=torch.Generator().manual_seed(7723))
    engine = Omni(
        model=str(model_dir),
        model_class_name="SeedVR2Pipeline",
        dtype="float16",
        enforce_eager=True,
        vae_use_tiling=True,
    )
    try:
        output = engine.generate(
            {"prompt": " ", "multi_modal_data": {"video": frames}},
            OmniDiffusionSamplingParams(
                height=128,
                width=224,
                num_frames=5,
                fps=24,
                num_inference_steps=1,
                guidance_scale=1.0,
                seed=7723,
                output_type="np",
                extra_args={"color_correction_method": method},
            ),
            use_tqdm=False,
        )
    finally:
        engine.close()
    assert len(output) == 1
    restored = np.asarray(output[0].images[0])
    assert restored.shape == (1, 5, 128, 224, 3)
    assert np.isfinite(restored).all()


def _write_source(path: Path, frames: int) -> None:
    """A constant-rate source with audio, which the route requires."""
    import imageio_ffmpeg

    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-nostdin",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size={LONG_SIZE}x{LONG_SIZE}:rate={FPS}",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440",
            # The generators are endless, so bound the video by frame count and
            # the audio by a slightly longer wall time.
            "-frames:v",
            str(frames),
            "-t",
            str((frames + 1) / FPS),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(path),
        ],
        check=True,
    )


def _await_status(base: str, job_id: str, timeout: float) -> dict:
    """Poll until the job leaves the queued/running states."""
    import requests

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = requests.get(f"{base}/{job_id}", timeout=30).json()
        if record["status"] not in {"queued", "running"}:
            return record
        time.sleep(2)
    raise AssertionError(f"SeedVR2 long-video job stayed {record['status']} for {timeout}s")


@pytest.mark.skipif(not os.environ.get(MODEL_DIR_ENV), reason=f"set {MODEL_DIR_ENV} to an authorized model directory")
@pytest.mark.parametrize("degree", [1, 8])
def test_seedvr2_long_video_route_e2e(tmp_path: Path, degree: int) -> None:
    """Restore a clip spanning two model windows, then cancel a longer job."""
    import av
    import requests
    import torch

    if torch.accelerator.device_count() < degree:
        pytest.skip(f"SeedVR2 USP={degree} requires {degree} CUDA devices")

    from tests.helpers.runtime import OmniServer
    from vllm_omni.diffusion.models.seedvr2.long_video import MAX_FRAMES

    model_dir = Path(os.environ[MODEL_DIR_ENV]).resolve()
    source = tmp_path / "input.mp4"
    _write_source(source, LONG_FRAMES)
    server = OmniServer(
        model=str(model_dir),
        serve_args=[
            "--model-class-name",
            "SeedVR2Pipeline",
            "--dtype",
            "float16",
            "--enforce-eager",
            "--num-gpus",
            str(degree),
            "--vae-use-tiling",
            "--stage-overrides",
            json.dumps(
                {
                    "0": {
                        "ulysses_degree": degree,
                        "vae_patch_parallel_size": degree,
                        "vae_parallel_mode": "spatial_shard_height",
                    }
                }
            ),
            "--api-server-count",
            "1",
        ],
        env_dict={
            "VLLM_OMNI_SEEDVR2_LONG_OUTPUT_DIR": str(tmp_path / "jobs"),
            "VLLM_OMNI_SEEDVR2_LONG_MAX_WINDOW": "13",
        },
    )

    with server:
        base = f"http://{server.host}:{server.port}/v1/seedvr2/restore-long"

        def submit(**overrides: str) -> requests.Response:
            with source.open("rb") as upload:
                return requests.post(
                    base,
                    data={
                        "prompt": " ",
                        "size": f"{LONG_SIZE}x{LONG_SIZE}",
                        "num_frames": str(LONG_FRAMES),
                        "seed": "7723",
                        **overrides,
                    },
                    files={"input_references": ("input.mp4", upload, "video/mp4")},
                    timeout=120,
                )

        assert submit(num_frames=str(MAX_FRAMES + 1)).status_code == 400

        # Validation must happen before a job is admitted, through model-owned
        # extra_params on both the long route and the common video endpoint.
        for invalid in ("{", "[]", '{"color_correction_method":"invalid"}'):
            assert submit(extra_params=invalid).status_code == 400
        with source.open("rb") as upload:
            invalid = requests.post(
                f"http://{server.host}:{server.port}/v1/videos/sync",
                data={
                    "prompt": " ",
                    "size": f"{LONG_SIZE}x{LONG_SIZE}",
                    "num_inference_steps": "1",
                    "guidance_scale": "1",
                    "extra_params": '{"color_correction_method":"invalid"}',
                },
                files={"input_references": ("input.mp4", upload, "video/mp4")},
                timeout=120,
            )
        assert invalid.status_code == 400, invalid.text
        accepted = submit(extra_params='{"color_correction_method":"wavelet"}')
        assert accepted.status_code == 202, accepted.text
        job_id = accepted.json()["id"]
        record = _await_status(base, job_id, timeout=900)
        assert record["status"] == "completed", record["error"]
        assert record["frames"] == LONG_FRAMES

        restored = tmp_path / "restored.mp4"
        content = requests.get(f"{base}/{job_id}/content", timeout=300)
        assert content.status_code == 200
        restored.write_bytes(content.content)
        with av.open(str(restored)) as container:
            video = container.streams.video[0]
            timestamps = [frame.pts * frame.time_base for frame in container.decode(video=0)]
            assert (video.width, video.height) == (LONG_SIZE, LONG_SIZE)
            assert video.average_rate == Fraction(FPS)
            assert timestamps == [Fraction(index, FPS) for index in range(LONG_FRAMES)]
            assert container.streams.audio, "the route must carry the source audio through"

        # Many windows, so the cancel lands well before the job could finish.
        running = submit(num_frames="600", loop_input="true")
        assert running.status_code == 202, running.text
        cancelled_id = running.json()["id"]
        assert submit().status_code == 409, "the route runs one job per server"
        assert requests.delete(f"{base}/{cancelled_id}", timeout=30).status_code == 202
        settled = _await_status(base, cancelled_id, timeout=900)
        assert settled["status"] == "cancelled", settled
        assert settled["frames"] < 600
        assert requests.get(f"{base}/{cancelled_id}/content", timeout=30).status_code == 404
