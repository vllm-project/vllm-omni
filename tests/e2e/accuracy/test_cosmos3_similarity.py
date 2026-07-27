# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path

import pytest
import requests
import torch
from PIL import Image

from tests.e2e.accuracy.helpers import model_output_dir
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServer

pytestmark = [pytest.mark.full_model, pytest.mark.diffusion]

MODEL_ENV_VAR = "VLLM_TEST_COSMOS3_MODEL"
MODEL_ID = "cosmos3"
PROMPT = "A small warehouse robot moves a blue box across a clean floor."
NEGATIVE_PROMPT = "blurry, distorted, low quality"
SEED = 42
WIDTH = HEIGHT = 256
NUM_INFERENCE_STEPS = 2


def _model_name() -> str:
    model = os.environ.get(MODEL_ENV_VAR)
    if not model:
        pytest.skip(f"Set {MODEL_ENV_VAR} to run Cosmos3 full-model smoke tests.")
    if not torch.cuda.is_available():
        pytest.skip("Cosmos3 full-model smoke tests require CUDA.")
    return model


def _server_args() -> list[str]:
    return [
        "--num-gpus",
        "1",
        "--stage-init-timeout",
        "900",
        "--init-timeout",
        "1200",
    ]


def _image_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


@pytest.mark.benchmark
@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_cosmos3_t2i_serving_smoke(accuracy_artifact_root: Path) -> None:
    output_dir = model_output_dir(accuracy_artifact_root, MODEL_ID)
    with OmniServer(_model_name(), _server_args(), use_omni=True) as server:
        response = requests.post(
            f"http://{server.host}:{server.port}/v1/images/generations",
            json={
                "model": server.model,
                "prompt": PROMPT,
                "negative_prompt": NEGATIVE_PROMPT,
                "size": f"{WIDTH}x{HEIGHT}",
                "n": 1,
                "response_format": "b64_json",
                "num_inference_steps": NUM_INFERENCE_STEPS,
                "guidance_scale": 1.0,
                "seed": SEED,
            },
            timeout=1800,
        )

    response.raise_for_status()
    payload = response.json()
    assert len(payload["data"]) == 1
    image = Image.open(io.BytesIO(base64.b64decode(payload["data"][0]["b64_json"]))).convert("RGB")
    image.save(output_dir / "cosmos3_t2i.png")
    assert image.size == (WIDTH, HEIGHT)


@pytest.mark.parametrize(
    ("name", "prompt", "num_frames", "image_reference"),
    [
        ("t2v", PROMPT, "1", None),
        (
            "i2v",
            "The blue rectangle moves slowly forward.",
            "5",
            Image.new("RGB", (96, 64), color=(40, 80, 160)),
        ),
    ],
)
@pytest.mark.benchmark
@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_cosmos3_video_serving_smoke(
    accuracy_artifact_root: Path,
    name: str,
    prompt: str,
    num_frames: str,
    image_reference: Image.Image | None,
) -> None:
    output_dir = model_output_dir(accuracy_artifact_root, MODEL_ID)
    data = {
        "model": "",
        "prompt": prompt,
        "negative_prompt": NEGATIVE_PROMPT,
        "size": f"{WIDTH}x{HEIGHT}",
        "num_frames": num_frames,
        "fps": "1",
        "num_inference_steps": str(NUM_INFERENCE_STEPS),
        "guidance_scale": "1.0",
        "seed": str(SEED),
    }
    if image_reference is not None:
        data["image_reference"] = json.dumps({"image_url": _image_data_url(image_reference)})

    with OmniServer(_model_name(), _server_args(), use_omni=True) as server:
        data["model"] = server.model
        response = requests.post(f"http://{server.host}:{server.port}/v1/videos/sync", data=data, timeout=1800)

    response.raise_for_status()
    assert response.headers["content-type"].startswith("video/mp4")
    assert response.content
    (output_dir / f"cosmos3_{name}.mp4").write_bytes(response.content)
