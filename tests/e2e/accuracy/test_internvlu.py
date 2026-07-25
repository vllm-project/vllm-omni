# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end serving pixel parity for InternVL-U text-to-image and editing.

The official InternVL-U reference stack pins ``transformers==4.52.3``, which
cannot be installed alongside this repository's requirements, so the official
pipeline cannot run in-process the way the Qwen-Image accuracy tests do.
Instead, this test compares served outputs against committed golden images
generated once with the official repository (rev ``018a59c1``, checkpoint
``InternVL-U/InternVL-U``) in a pinned ``transformers==4.52.3`` /
``diffusers==0.35.1`` environment on A100, bf16, seed 42, 20 steps:

- ``internvlu_t2i_golden.png``: official T2I output for ``T2I_PROMPT`` at
  1024x1024.
- ``internvlu_edit_golden.png``: official edit of the T2I golden with
  ``EDIT_PROMPT`` at 1024x1024 (the golden also serves as the edit input, so
  no third-party image is needed).

Thresholds carry margin below the cross-implementation scores measured when
the goldens were created, to absorb attention-backend and GPU-generation
numeric drift.
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
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServer
from tests.helpers.stage_config import get_deploy_config_path

pytestmark = [pytest.mark.slow, pytest.mark.diffusion]

MODEL_ID = "InternVL-U/InternVL-U"
MODEL_ENV_VAR = "INTERNVLU_MODEL"
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
T2I_GOLDEN = ASSETS_DIR / "internvlu_t2i_golden.png"
EDIT_GOLDEN = ASSETS_DIR / "internvlu_edit_golden.png"

T2I_PROMPT = (
    "A photo of an orange tabby cat sitting on a wooden table next to a cup of "
    "coffee, warm morning light, shallow depth of field, photorealistic."
)
EDIT_PROMPT = "Add a red knitted scarf around the cat's neck."
WIDTH = 1024
HEIGHT = 1024
# Divisible by 16 but not by 32: guards the released stride-16 size contract.
STRIDE16_WIDTH = 1024
STRIDE16_HEIGHT = 1008
NUM_INFERENCE_STEPS = 20
SEED = 42
# Measured vs the goldens at creation time (A100, TORCH_SDPA): T2I SSIM 0.979
# / PSNR 34.2; edit SSIM 0.994 / PSNR 43.6. A second A100 environment (torch
# 2.11) reproduced the edit scores exactly but measured T2I at SSIM 0.954 /
# PSNR 28.1 (bit-deterministic across runs, visually near-identical): the
# from-noise T2I trajectory amplifies per-step kernel differences over 20
# steps, while edit outputs stay anchored to the reference conditioning. The
# T2I bar therefore carries margin below the weakest observed environment;
# the edit bar keeps the Qwen-Image accuracy level.
T2I_SSIM_THRESHOLD = 0.93
T2I_PSNR_THRESHOLD = 26.0
EDIT_SSIM_THRESHOLD = 0.94
EDIT_PSNR_THRESHOLD = 30.0


def _model_name() -> str:
    return os.environ.get(MODEL_ENV_VAR, MODEL_ID)


def _decode_single_image(payload: dict) -> Image.Image:
    assert len(payload["data"]) == 1
    image_bytes = base64.b64decode(payload["data"][0]["b64_json"])
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image.load()
    return image


def _generate(omni_server: OmniServer, *, width: int, height: int, steps: int) -> Image.Image:
    response = requests.post(
        f"http://{omni_server.host}:{omni_server.port}/v1/images/generations",
        json={
            "model": omni_server.model,
            "prompt": T2I_PROMPT,
            "size": f"{width}x{height}",
            "n": 1,
            "response_format": "b64_json",
            "num_inference_steps": steps,
            "seed": SEED,
        },
        timeout=900,
    )
    response.raise_for_status()
    return _decode_single_image(response.json())


def _edit(omni_server: OmniServer, input_image: Image.Image) -> Image.Image:
    png = io.BytesIO()
    input_image.save(png, format="PNG")
    response = requests.post(
        f"http://{omni_server.host}:{omni_server.port}/v1/images/edits",
        data={
            "model": omni_server.model,
            "prompt": EDIT_PROMPT,
            "size": f"{WIDTH}x{HEIGHT}",
            "n": 1,
            "response_format": "b64_json",
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "seed": SEED,
        },
        files=[("image", ("input.png", png.getvalue(), "image/png"))],
        timeout=900,
    )
    response.raise_for_status()
    return _decode_single_image(response.json())


@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_internvlu_t2i_and_edit_serving(accuracy_artifact_root: Path) -> None:
    model = _model_name()
    output_dir = model_output_dir(accuracy_artifact_root, MODEL_ID)
    t2i_golden = Image.open(T2I_GOLDEN).convert("RGB")
    edit_golden = Image.open(EDIT_GOLDEN).convert("RGB")
    server_args = [
        "--deploy-config",
        get_deploy_config_path("internvlu_chat.yaml"),
        "--stage-init-timeout",
        "600",
        "--init-timeout",
        "900",
    ]

    with OmniServer(model, server_args, use_omni=True) as omni_server:
        t2i_image = _generate(omni_server, width=WIDTH, height=HEIGHT, steps=NUM_INFERENCE_STEPS)
        t2i_image.save(output_dir / "internvlu_t2i.png")

        stride16_image = _generate(omni_server, width=STRIDE16_WIDTH, height=STRIDE16_HEIGHT, steps=4)
        stride16_image.save(output_dir / "internvlu_t2i_stride16.png")
        assert stride16_image.size == (STRIDE16_WIDTH, STRIDE16_HEIGHT)

        edit_image = _edit(omni_server, t2i_golden)
        edit_image.save(output_dir / "internvlu_edit.png")

    assert_similarity(
        model_name="internvlu-t2i",
        vllm_image=t2i_image,
        diffusers_image=t2i_golden,
        ssim_threshold=T2I_SSIM_THRESHOLD,
        psnr_threshold=T2I_PSNR_THRESHOLD,
    )
    assert_similarity(
        model_name="internvlu-edit",
        vllm_image=edit_image,
        diffusers_image=edit_golden,
        ssim_threshold=EDIT_SSIM_THRESHOLD,
        psnr_threshold=EDIT_PSNR_THRESHOLD,
    )


@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_internvlu_think_text_then_image_serving(accuracy_artifact_root: Path) -> None:
    """Think deployment: Stage 0 writes a description before image generation.

    Functionality-level (no golden): the CoT text must surface as
    ``cot_output`` and the image must come back at the requested size.  Think
    mode is a single deployment-static knob (internvlu_chat_think.yaml); the
    request carries no CoT fields.
    """
    model = _model_name()
    output_dir = model_output_dir(accuracy_artifact_root, MODEL_ID)
    server_args = [
        "--deploy-config",
        get_deploy_config_path("internvlu_chat_think.yaml"),
        "--stage-init-timeout",
        "600",
        "--init-timeout",
        "900",
    ]

    with OmniServer(model, server_args, use_omni=True) as omni_server:
        response = requests.post(
            f"http://{omni_server.host}:{omni_server.port}/v1/images/generations",
            json={
                "model": omni_server.model,
                "prompt": T2I_PROMPT,
                "size": f"{WIDTH}x{HEIGHT}",
                "n": 1,
                "response_format": "b64_json",
                "num_inference_steps": NUM_INFERENCE_STEPS,
                "seed": SEED,
            },
            timeout=900,
        )
        response.raise_for_status()
        payload = response.json()

    cot_output = payload.get("cot_output")
    assert isinstance(cot_output, str) and cot_output.strip(), "think deployment must surface non-empty cot_output"

    think_image = _decode_single_image(payload)
    assert think_image.size == (WIDTH, HEIGHT)
    think_image.save(output_dir / "internvlu_think_t2i.png")
