# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import base64
import io
import os
import time
from pathlib import Path

import pytest
import requests
from PIL import Image

from tests.e2e.accuracy.helpers import assert_similarity, model_output_dir
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServer

pytestmark = [pytest.mark.full_model, pytest.mark.diffusion]

MODEL_ID = "Boogu/Boogu-Image-0.1-Base"
MODEL_ENV_VAR = "BOOGU_IMAGE_MODEL"
PROMPT = "A mountain lake at sunset, photorealistic, cinematic lighting"
NEGATIVE_PROMPT = ""
WIDTH = 512
HEIGHT = 512
NUM_INFERENCE_STEPS = 28
GUIDANCE_SCALE = 4.0
SEED = 42
SSIM_THRESHOLD = 0.90
PSNR_THRESHOLD = 30.0

CACHE_DIT_ARGS = [
    "--cache-backend",
    "cache_dit",
    "--cache-config",
    '{"max_warmup_steps":4,"max_continuous_cached_steps":6,"residual_diff_threshold":0.12}',
]


def _model_name() -> str:
    return os.environ.get(MODEL_ENV_VAR, MODEL_ID)


def _wait_until_healthy(server: OmniServer, client: requests.Session) -> None:
    """Wait past the early TCP-listen phase until application startup ends."""
    deadline = time.monotonic() + 180
    health_url = f"http://{server.host}:{server.port}/health"
    while time.monotonic() < deadline:
        try:
            if client.get(health_url, timeout=5).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise TimeoutError(f"Boogu server did not become healthy: {health_url}")


def _run_boogu(*, model: str, output_path: Path, cache_dit: bool) -> Image.Image:
    server_args = [
        "--num-gpus",
        "1",
        "--stage-init-timeout",
        "300",
        "--init-timeout",
        "900",
        "--vae-use-slicing",
        "--vae-use-tiling",
    ]
    if cache_dit:
        server_args.extend(CACHE_DIT_ARGS)

    with requests.Session() as client:
        # Loopback test traffic must not be routed through a developer's HTTP_PROXY.
        client.trust_env = False
        with OmniServer(model, server_args, use_omni=True) as server:
            _wait_until_healthy(server, client)
            response = client.post(
                f"http://{server.host}:{server.port}/v1/images/generations",
                json={
                    "model": server.model,
                    "prompt": PROMPT,
                    "negative_prompt": NEGATIVE_PROMPT,
                    "size": f"{WIDTH}x{HEIGHT}",
                    "n": 1,
                    "response_format": "b64_json",
                    "num_inference_steps": NUM_INFERENCE_STEPS,
                    "guidance_scale": GUIDANCE_SCALE,
                    "seed": SEED,
                },
                timeout=1200,
            )

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise requests.HTTPError(f"{exc}; response body: {response.text}", response=response) from exc
    payload = response.json()
    assert len(payload["data"]) == 1
    image = Image.open(io.BytesIO(base64.b64decode(payload["data"][0]["b64_json"]))).convert("RGB")
    image.load()
    image.save(output_path)
    return image


@pytest.mark.benchmark
@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_boogu_cache_dit_matches_dense(accuracy_artifact_root: Path) -> None:
    model = _model_name()
    output_dir = model_output_dir(accuracy_artifact_root, MODEL_ID)

    dense = _run_boogu(
        model=model,
        output_path=output_dir / "dense.png",
        cache_dit=False,
    )
    cached = _run_boogu(
        model=model,
        output_path=output_dir / "cache_dit.png",
        cache_dit=True,
    )

    assert_similarity(
        model_name=f"{MODEL_ID}-cache-dit-vs-dense",
        vllm_image=cached,
        diffusers_image=dense,
        width=WIDTH,
        height=HEIGHT,
        ssim_threshold=SSIM_THRESHOLD,
        psnr_threshold=PSNR_THRESHOLD,
    )
