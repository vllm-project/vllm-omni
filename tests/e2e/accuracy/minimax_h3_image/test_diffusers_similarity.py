# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Calibrate H3-Image output similarity against the official Diffusers pipeline."""

import base64
import io
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest
import requests
from huggingface_hub import constants
from PIL import Image

from tests.e2e.accuracy.helpers import compute_image_ssim_psnr, write_json_artifact
from tests.helpers.mark import hardware_test
from tests.helpers.media import get_asset_path
from tests.helpers.runtime import OmniServer
from vllm_omni.platforms import current_omni_platform
from vllm_omni.transformers_utils.repo_utils import hf_api

pytestmark = [pytest.mark.full_model, pytest.mark.diffusion]

MODEL_ID = "MiniMaxAI/MiniMax-H3-Image"
MODEL_REVISION = "57f313e135ccfdec4e7e6de9b7ba4579c2b7c663"
PROMPTS = {
    "t2i": "A red bicycle leaning against a pale stone wall in warm afternoon sunlight.",
    "edit": "Restyle <Picture 1> into a warm oil painting while preserving the subject and composition.",
}


@hardware_test(res={"cuda": "H100", "npu": "A3"}, num_cards=1)
def test_diffusers_similarity(accuracy_artifact_root):
    python = os.environ.get("H3_IMAGE_REFERENCE_PYTHON")
    assert python and Path(python).is_file(), "Source setup_reference.sh first"
    model = hf_api().snapshot_download(
        MODEL_ID,
        revision=MODEL_REVISION,
        allow_patterns=[
            "model_index.json",
            "modular_model_index.json",
            "pdd_*",
            "transformer/*",
            "transformer_diffusers/*",
            "vae/*",
            "vae_diffusers/*",
            "scheduler/*",
            "text_encoder/*",
            "tokenizer/*",
            "processor/*",
        ],
        local_files_only=constants.HF_HUB_OFFLINE,
    )
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (accuracy_artifact_root / "minimax_h3_image" / run_id).resolve()
    output.mkdir(parents=True)
    source = get_asset_path("qwen_image_edit/qwen_image_edit_2511_test1.png")
    reference_bytes = source.read_bytes()
    (output / "edit-input.png").write_bytes(reference_bytes)
    common = {
        "model": model,
        "size": "2048x2048",
        "seed": 42,
        "generator_device": "cpu",
        "num_inference_steps": 6,
        "response_format": "b64_json",
        "output_format": "png",
    }
    device_type = current_omni_platform.device_type
    request_file = output / "request.json"
    write_json_artifact(
        request_file, {**common, "prompts": PROMPTS, "revision": MODEL_REVISION, "device_type": device_type}
    )
    command = [python, str(Path(__file__).with_name("run_reference.py")), str(request_file)]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    for preflight in (True, False):
        log_name = "reference-preflight.log" if preflight else "reference.log"
        with (output / log_name).open("w") as log:
            subprocess.run(
                [*command, *(["--preflight"] if preflight else [])],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=300 if preflight else 3600,
            )
    args = [
        "--num-gpus",
        "1",
        "--diffusion-offload-config",
        '{"mode":"layer","components":["dit","text_encoder"]}',
        "--init-timeout",
        "1200",
    ]
    if device_type == "npu":
        args.append("--enforce-eager")
    with OmniServer(model, args, startup_timeout=1200) as server:
        url = f"http://{server.host}:{server.port}/v1/images/"
        for task, prompt in PROMPTS.items():
            request = {**common, "prompt": prompt}
            for draw in range(2):
                if task == "edit":
                    response = requests.post(
                        url + "edits",
                        data=request,
                        files=[("image", ("reference.png", reference_bytes, "image/png"))],
                        timeout=600,
                    )
                else:
                    response = requests.post(url + "generations", json=request, timeout=600)
                response.raise_for_status()
                images = response.json()["data"]
                assert len(images) == 1
                with Image.open(io.BytesIO(base64.b64decode(images[0]["b64_json"], validate=True))) as image:
                    assert image.format == "PNG" and image.size == (2048, 2048)
                    rgb = image.convert("RGB")
                    assert any(low != high for low, high in rgb.getextrema())
                    rgb.save(output / f"native-{task}-{draw}.png")

    metrics = {}
    for task in PROMPTS:
        comparisons = {
            "native_vs_diffusers_first": ("native", 0, "diffusers", 0),
            "native_vs_diffusers_second": ("native", 1, "diffusers", 1),
            "diffusers_repeatability": ("diffusers", 1, "diffusers", 0),
            "native_repeatability": ("native", 1, "native", 0),
        }
        metrics[task] = {}
        for name, (candidate, candidate_draw, reference, reference_draw) in comparisons.items():
            with (
                Image.open(output / f"{candidate}-{task}-{candidate_draw}.png") as prediction,
                Image.open(output / f"{reference}-{task}-{reference_draw}.png") as baseline,
            ):
                ssim, psnr = compute_image_ssim_psnr(prediction=prediction, reference=baseline)
            metrics[task][name] = {"ssim": ssim, "psnr_db": psnr if math.isfinite(psnr) else "inf"}
    write_json_artifact(output / "metrics.json", metrics)
    pytest.xfail(f"Diffusers similarity calibration collected at {output}")
