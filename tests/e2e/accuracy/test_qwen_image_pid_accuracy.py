# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""L4 accuracy for PiD 4x super-resolution on Qwen-Image (nightly).

Two quality guard lines:
1. Structure preservation: PiD 4x output downsampled to 512 vs the VAE 512
   baseline from the same server (content must not drift or hallucinate).
2. Distill determinism (golden): PiD 4x output vs the nvidia/PiD official
   reference for the same prompt/seed (guards regression of the distilled
   student). Skipped when the golden reference image is absent.

Thresholds are initial suggestions; calibrate on H100 before enabling.
"""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path

import pytest
import requests
from PIL import Image

from tests.e2e.accuracy.helpers import assert_similarity, model_output_dir
from tests.helpers.runtime import OmniServer

pytestmark = [pytest.mark.full_model, pytest.mark.diffusion]

MODEL_ID = os.environ.get("QWEN_IMAGE_MODEL", "Qwen/Qwen-Image")
PID_CKPT = os.environ.get("PID_CKPT", "")
PID_GEMMA = os.environ.get("PID_GEMMA", "Efficient-Large-Model/gemma-2-2b-it")
PROMPT = "A photo of a cat sitting on a laptop keyboard, digital art style."
SIZE = "512x512"
SSIM_STRUCTURE, PSNR_STRUCTURE = 0.90, 28.0
SSIM_GOLDEN, PSNR_GOLDEN = 0.94, 30.0

_pid_args = ["--pid-enable", "--pid-gemma", PID_GEMMA]
if PID_CKPT:
    _pid_args += ["--pid-checkpoint", PID_CKPT]


def _omni_pid(*, enabled: bool) -> Image.Image:
    server_args = ["--num-gpus", "1", "--stage-init-timeout", "300", "--init-timeout", "900", *_pid_args]
    with OmniServer(MODEL_ID, server_args, use_omni=True) as omni_server:
        resp = requests.post(
            f"http://{omni_server.host}:{omni_server.port}/v1/images/generations",
            json={
                "model": omni_server.model,
                "prompt": PROMPT,
                "size": SIZE,
                "n": 1,
                "response_format": "b64_json",
                "num_inference_steps": 20,
                "true_cfg_scale": 4.0,
                "seed": 42,
                "pid_decode": {"enabled": enabled, "scale": 4, "num_steps": 4, "seed": 42},
            },
            timeout=600,
        )
        resp.raise_for_status()
        data = resp.json()["data"][0]["b64_json"]
        img = Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")
        img.load()
        return img


def _golden_reference() -> Image.Image | None:
    # Produced once with nvidia/PiD scripts/pipeline_demo.py for the same prompt/seed.
    golden = model_output_dir(Path(__file__).parent, "qwen_image_pid") / "golden_4x.png"
    return Image.open(golden).convert("RGB") if golden.exists() else None


def test_pid_4x_structure_preserved() -> None:
    """PiD 4x downsampled to 512 matches the VAE 512 baseline (no content drift)."""
    pid_4x = _omni_pid(enabled=True)
    assert pid_4x.size == (2048, 2048)
    baseline = _omni_pid(enabled=False)
    pid_down = pid_4x.resize(baseline.size, Image.LANCZOS)
    assert_similarity(
        model_name="qwen_image_pid",
        vllm_image=pid_down,
        diffusers_image=baseline,
        ssim_threshold=SSIM_STRUCTURE,
        psnr_threshold=PSNR_STRUCTURE,
    )


def test_pid_4x_golden_match() -> None:
    """PiD 4x output matches the nvidia/PiD official reference (distill determinism)."""
    golden = _golden_reference()
    if golden is None:
        pytest.skip("missing nvidia/PiD official reference output; skipping golden comparison")
    pid_4x = _omni_pid(enabled=True)
    assert_similarity(
        model_name="qwen_image_pid",
        vllm_image=pid_4x,
        diffusers_image=golden,
        ssim_threshold=SSIM_GOLDEN,
        psnr_threshold=PSNR_GOLDEN,
    )
