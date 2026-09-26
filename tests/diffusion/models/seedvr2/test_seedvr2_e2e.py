# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Full-checkpoint DiT parity, native video restoration, and the long route.

Set VLLM_TEST_SEEDVR2_MODEL to the released 3B FP16 safetensors file.
Set VLLM_TEST_SEEDVR2_MODEL_DIR to the directory containing the DiT, VAE,
conditioning tensor, and model_index.json to run the complete video test.
Each rank exits normally; the launcher does not terminate workers.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

import pytest

pytestmark = [pytest.mark.local_model, pytest.mark.cuda, pytest.mark.diffusion, pytest.mark.parallel]
MODEL_ENV = "VLLM_TEST_SEEDVR2_MODEL"
MODEL_DIR_ENV = "VLLM_TEST_SEEDVR2_MODEL_DIR"
# Two model windows with one four-frame seam, at the smallest legal frame size.
LONG_FRAMES = 20
LONG_SIZE = 64
FPS = 24


@pytest.mark.skipif(not os.environ.get(MODEL_ENV), reason=f"set {MODEL_ENV} to an authorized 3B checkpoint path")
@pytest.mark.parametrize("degree", [1, 2, 4])
def test_seedvr2_checkpoint_sequence_parallel(degree: int, tmp_path: Path) -> None:
    import torch

    if torch.accelerator.device_count() < degree:
        pytest.skip(f"SeedVR2 SP={degree} requires {degree} CUDA devices")
    checkpoint = Path(os.environ[MODEL_ENV]).resolve()
    assert checkpoint.is_file(), f"Checkpoint does not exist: {checkpoint}"
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    processes = []
    for rank in range(degree):
        env = os.environ | {
            MODEL_ENV: str(checkpoint),
            "PYTHONPATH": str(Path(__file__).resolve().parents[4]) + os.pathsep + os.environ.get("PYTHONPATH", ""),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "WORLD_SIZE": str(degree),
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
        }
        with (tmp_path / f"rank-{rank}.log").open("w") as log:
            processes.append(
                subprocess.Popen(
                    [sys.executable, str(Path(__file__).resolve()), str(tmp_path)],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            )
    codes = [process.wait() for process in processes]
    logs = "\n".join((tmp_path / f"rank-{rank}.log").read_text() for rank in range(degree))
    assert codes == [0] * degree, logs
    for rank in range(degree):
        report = json.loads((tmp_path / f"rank-{rank}.json").read_text())
        assert report["shape"] == [4096, 16]
        assert report["finite"]
        assert report["relative_l2"] < 0.02
        assert report["layout_transitions"] == 31
        assert report["text_all_reduces"] == (32 if degree > 1 else 0)
        if degree > 1:
            assert report["network_transitions"] > 0


@pytest.mark.skipif(not os.environ.get(MODEL_DIR_ENV), reason=f"set {MODEL_DIR_ENV} to an authorized model directory")
def test_seedvr2_native_video_e2e() -> None:
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
def test_seedvr2_long_video_route_e2e(tmp_path: Path) -> None:
    """Restore a clip spanning two model windows, then cancel a longer job."""
    import av
    import requests

    from tests.helpers.runtime import OmniServer
    from vllm_omni.entrypoints.openai.video.seedvr2_long import MAX_FRAMES

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
            "1",
            "--api-server-count",
            "1",
        ],
        env_dict={"VLLM_OMNI_SEEDVR2_LONG_OUTPUT_DIR": str(tmp_path / "jobs")},
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

        accepted = submit()
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


def _run_rank(output_dir: Path) -> None:
    import torch
    from safetensors.torch import load_file

    from vllm_omni.diffusion.data import DiffusionParallelConfig
    from vllm_omni.diffusion.distributed.parallel_state import (
        destroy_distributed_env,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm_omni.diffusion.models.seedvr2.nadit import SEEDVR2_3B_CONFIG, SeedVR2NaDiT

    rank, degree = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.set_num_threads(4)
    torch.accelerator.set_device_index(rank)
    init_distributed_environment(world_size=degree, rank=rank, local_rank=rank)
    initialize_model_parallel(sequence_parallel_size=degree, ulysses_degree=degree)
    try:
        device = torch.device("cuda", rank)
        with torch.device("meta"):
            model = SeedVR2NaDiT(**SEEDVR2_3B_CONFIG, use_varlen_kernel=False)
        state = load_file(os.environ[MODEL_ENV])
        # The release nests the RoPE buffer one level below the port.
        normalized = {
            key.removesuffix(".rope.rope.freqs") + ".rope.freqs" if key.endswith(".rope.rope.freqs") else key: value
            for key, value in state.items()
        }
        assert len(normalized) == len(state)
        model.load_state_dict(normalized, strict=True, assign=True)
        model = model.to(device=device, dtype=torch.float16).eval()
        del state, normalized
        generator = torch.Generator().manual_seed(7723)
        inputs = {
            "vid": torch.randn(4096, 33, generator=generator).to(device, torch.float16),
            "txt": torch.randn(58, 5120, generator=generator).to(device, torch.float16),
            "vid_shape": torch.tensor([[1, 64, 64]], device=device),
            "txt_shape": torch.tensor([[58]], device=device),
            "timestep": torch.tensor([1000.0], device=device, dtype=torch.float16),
        }
        runtime = model.build_runtime(
            model.token_grid_for(inputs["vid_shape"]),
            text_len=58,
            parallel_config=DiffusionParallelConfig(ulysses_degree=degree),
        )
        with torch.inference_mode():
            reference = model(**inputs).vid_sample
            actual = model(**inputs, runtime=runtime).vid_sample
        torch.testing.assert_close(actual, reference, atol=0.02, rtol=0.02)
        relative_l2 = (actual.float() - reference.float()).norm() / reference.float().norm().clamp_min(1e-12)
        report = {
            "shape": list(actual.shape),
            "finite": bool(torch.isfinite(actual).all()),
            "relative_l2": relative_l2.item(),
            "max_abs_error": (actual.float() - reference.float()).abs().max().item(),
            **runtime.stats,
        }
        (output_dir / f"rank-{rank}.json").write_text(json.dumps(report, indent=2))
    finally:
        destroy_distributed_env()


if __name__ == "__main__":
    _run_rank(Path(sys.argv[1]))
