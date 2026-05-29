# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tests.helpers.mark import hardware_test

REPO_ROOT = Path(__file__).resolve().parents[3]
T2V_EXAMPLE = REPO_ROOT / "examples" / "offline_inference" / "text_to_video" / "text_to_video.py"
I2V_EXAMPLE = REPO_ROOT / "examples" / "offline_inference" / "image_to_video" / "image_to_video.py"

T2V_PROMPT = (
    "At sunrise, a glowing paper lantern boat drifts through a narrow canal between mossy stone walls, "
    "soft fog above the water, the camera slowly gliding forward as golden reflections shimmer across "
    "the ripples, cinematic, realistic, highly detailed."
)
T2V_NEGATIVE_PROMPT = "worst quality, blurry, jittery motion, distorted, oversaturated, artifacts"
I2V_PROMPT = "A cinematic dolly shot of a boat drifting on calm water at sunset"
I2V_NEGATIVE_PROMPT = "worst quality, blurry, jittery motion"

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


def _get_ltx2_model() -> str:
    return "Lightricks/LTX-2"


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _make_deterministic_test_image(path: Path) -> None:
    """Create a deterministic 256x256 test image for I2V tests."""
    rng = np.random.RandomState(42)
    img = Image.fromarray(rng.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    img.save(path)


def _run_and_check(cmd: list[str], env: dict, output_path: Path, expected_md5: str) -> None:
    result = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 0, (
        f"Command failed (exit {result.returncode}).\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    generated_md5 = _md5(output_path)
    assert generated_md5 == expected_md5, (
        f"Unexpected output md5: {generated_md5} != {expected_md5}.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


# ── T2V tests ──


@pytest.mark.full_model
@pytest.mark.diffusion
@pytest.mark.parallel
@pytest.mark.slow
@hardware_test(res={"cuda": "L4"}, num_cards=2)
def test_ltx2_t2v_cfg_parallel(tmp_path: Path):
    """T2V with CFG=4.0, cfg-parallel-size=2."""
    output = tmp_path / "t2v_cfg4.mp4"
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
    cmd = [
        sys.executable,
        str(T2V_EXAMPLE),
        "--model",
        _get_ltx2_model(),
        "--prompt",
        T2V_PROMPT,
        "--negative-prompt",
        T2V_NEGATIVE_PROMPT,
        "--height",
        "256",
        "--width",
        "256",
        "--num-frames",
        "145",
        "--num-inference-steps",
        "6",
        "--guidance-scale",
        "4.0",
        "--frame-rate",
        "24",
        "--fps",
        "24",
        "--seed",
        "42",
        "--cfg-parallel-size",
        "2",
        "--enforce-eager",
        "--output",
        str(output),
    ]
    _run_and_check(cmd, env, output, expected_md5="08e606b9c522fee4b6f30cee8b77db40")


@pytest.mark.full_model
@pytest.mark.diffusion
@pytest.mark.slow
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_ltx2_t2v_no_cfg(tmp_path: Path):
    """T2V with CFG=1.0 (no classifier-free guidance)."""
    output = tmp_path / "t2v_nocfg.mp4"
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    cmd = [
        sys.executable,
        str(T2V_EXAMPLE),
        "--model",
        _get_ltx2_model(),
        "--prompt",
        T2V_PROMPT,
        "--height",
        "256",
        "--width",
        "256",
        "--num-frames",
        "145",
        "--num-inference-steps",
        "6",
        "--guidance-scale",
        "1.0",
        "--frame-rate",
        "24",
        "--fps",
        "24",
        "--seed",
        "42",
        "--enforce-eager",
        "--output",
        str(output),
    ]
    _run_and_check(cmd, env, output, expected_md5="a83994b94b6e67c54a524e0383c45ce8")


# ── I2V tests ──


@pytest.mark.full_model
@pytest.mark.diffusion
@pytest.mark.parallel
@pytest.mark.slow
@hardware_test(res={"cuda": "L4"}, num_cards=2)
def test_ltx2_i2v_cfg_parallel(tmp_path: Path):
    """I2V with CFG=4.0, cfg-parallel-size=2."""
    test_image = tmp_path / "test_input.png"
    _make_deterministic_test_image(test_image)
    output = tmp_path / "i2v_cfg4.mp4"
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
    cmd = [
        sys.executable,
        str(I2V_EXAMPLE),
        "--model",
        _get_ltx2_model(),
        "--model-class-name",
        "LTX2ImageToVideoPipeline",
        "--image",
        str(test_image),
        "--prompt",
        I2V_PROMPT,
        "--negative-prompt",
        I2V_NEGATIVE_PROMPT,
        "--height",
        "256",
        "--width",
        "256",
        "--num-frames",
        "73",
        "--num-inference-steps",
        "6",
        "--guidance-scale",
        "4.0",
        "--frame-rate",
        "24",
        "--fps",
        "24",
        "--seed",
        "42",
        "--cfg-parallel-size",
        "2",
        "--enforce-eager",
        "--output",
        str(output),
    ]
    _run_and_check(cmd, env, output, expected_md5="aed7e56084b36373244d8f839b16d115")


@pytest.mark.full_model
@pytest.mark.diffusion
@pytest.mark.slow
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_ltx2_i2v_no_cfg(tmp_path: Path):
    """I2V with CFG=1.0 (no classifier-free guidance)."""
    test_image = tmp_path / "test_input.png"
    _make_deterministic_test_image(test_image)
    output = tmp_path / "i2v_nocfg.mp4"
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    cmd = [
        sys.executable,
        str(I2V_EXAMPLE),
        "--model",
        _get_ltx2_model(),
        "--model-class-name",
        "LTX2ImageToVideoPipeline",
        "--image",
        str(test_image),
        "--prompt",
        I2V_PROMPT,
        "--height",
        "256",
        "--width",
        "256",
        "--num-frames",
        "73",
        "--num-inference-steps",
        "6",
        "--guidance-scale",
        "1.0",
        "--frame-rate",
        "24",
        "--fps",
        "24",
        "--seed",
        "42",
        "--enforce-eager",
        "--output",
        str(output),
    ]
    _run_and_check(cmd, env, output, expected_md5="81b21ede12753e9e14a357a6c548b666")
