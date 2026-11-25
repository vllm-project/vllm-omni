# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Calibrate H3-Image output similarity against the official Diffusers pipeline."""

import base64
import io
import json
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest
import requests
from huggingface_hub import constants
from PIL import Image

from tests.e2e.accuracy.helpers import (
    SimilarityThresholds,
    assert_similarity_metrics,
    baseline_from_similarity_thresholds,
    compute_image_ssim_psnr,
    resolve_similarity_thresholds,
    write_accuracy_result,
    write_json_artifact,
)
from tests.helpers.mark import hardware_test
from tests.helpers.media import get_asset_path
from tests.helpers.runtime import OmniServer
from vllm_omni.platforms import current_omni_platform
from vllm_omni.transformers_utils.repo_utils import hf_api

pytestmark = [pytest.mark.full_model, pytest.mark.diffusion]

MODEL_ID = "MiniMaxAI/MiniMax-H3-Image"
PROMPTS = {
    "t2i": "A red bicycle leaning against a pale stone wall in warm afternoon sunlight.",
    "edit": "Restyle <Picture 1> into a warm oil painting while preserving the subject and composition.",
}
# The configured timeouts budget 45 minutes on CUDA and 50 on NPU, leaving time
# for environment setup, snapshot resolution, and artifact handling.
REFERENCE_PREFLIGHT_TIMEOUT_S = 180
REFERENCE_GENERATION_TIMEOUT_S = 1200
NPU_REFERENCE_GENERATION_TIMEOUT_S = 1500
OMNI_STARTUP_TIMEOUT_S = 600
IMAGE_REQUEST_TIMEOUT_S = 180


@hardware_test(res={"cuda": "H100", "npu": "A3"}, num_cards=1)
def test_diffusers_similarity(accuracy_artifact_root):
    device_type = current_omni_platform.device_type
    calibration = os.environ.get("H3_IMAGE_CALIBRATE", "1")
    assert calibration in {"0", "1"}, "H3_IMAGE_CALIBRATE must be 0 or 1"
    policy = json.loads(Path(__file__).with_name("thresholds.json").read_text(encoding="utf-8"))
    tables = {}
    thresholds = {}
    if calibration == "0":
        assert policy["status"] == "selected", (
            "Select device-specific bounds in thresholds.json after calibration before setting H3_IMAGE_CALIBRATE=0"
        )
        for task in PROMPTS:
            tables[task] = {profile: SimilarityThresholds(**row) for profile, row in policy[task].items()}
            floors = resolve_similarity_thresholds(tables[task])
            assert type(floors.ssim) in (float, int) and -1 <= floors.ssim <= 1, f"Invalid SSIM floor for {task}"
            assert type(floors.psnr) in (float, int) and math.isfinite(floors.psnr) and floors.psnr >= 0, (
                f"Invalid PSNR floor for {task}"
            )
            thresholds[task] = floors
    python = os.environ.get("H3_IMAGE_REFERENCE_PYTHON")
    assert python and Path(python).is_file(), "Source setup_reference.sh first"
    model = hf_api().snapshot_download(
        MODEL_ID,
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
    revision = Path(model).name  # The Hub returns a snapshots/<commit> directory.
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
    request_file = output / "request.json"
    write_json_artifact(request_file, {**common, "prompts": PROMPTS, "revision": revision, "device_type": device_type})
    command = [python, str(Path(__file__).with_name("run_reference.py")), str(request_file)]
    env = dict(os.environ)
    pythonpath = env.pop("PYTHONPATH", "")
    reference_timeout = REFERENCE_GENERATION_TIMEOUT_S
    if device_type == "npu":
        reference_timeout = NPU_REFERENCE_GENERATION_TIMEOUT_S
        env["PATH"] = f"{Path(python).parent}{os.pathsep}{env.get('PATH', os.defpath)}"
        # Keep the CANN compiler modules visible without inheriting serving packages.
        cann_home = env.get("ASCEND_HOME_PATH") or env.get("ASCEND_TOOLKIT_HOME")
        if cann_home:
            cann_root = Path(cann_home).resolve()
            cann_paths = [p for p in pythonpath.split(os.pathsep) if p and Path(p).resolve().is_relative_to(cann_root)]
            if cann_paths:
                env["PYTHONPATH"] = os.pathsep.join(cann_paths)
    for preflight in (True, False):
        log_name = "reference-preflight.log" if preflight else "reference.log"
        with (output / log_name).open("w") as log:
            subprocess.run(
                [*command, *(["--preflight"] if preflight else [])],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=REFERENCE_PREFLIGHT_TIMEOUT_S if preflight else reference_timeout,
            )
    args = [
        "--num-gpus",
        "1",
        "--diffusion-offload-config",
        '{"mode":"layer","components":["dit","text_encoder"]}',
        "--init-timeout",
        "480",
    ]
    if device_type == "npu":
        args.append("--enforce-eager")
    with OmniServer(model, args, startup_timeout=OMNI_STARTUP_TIMEOUT_S) as server:
        url = f"http://{server.host}:{server.port}/v1/images/"
        for task, prompt in PROMPTS.items():
            request = {**common, "prompt": prompt}
            for draw in range(2):
                if task == "edit":
                    response = requests.post(
                        url + "edits",
                        data=request,
                        files=[("image", ("reference.png", reference_bytes, "image/png"))],
                        timeout=IMAGE_REQUEST_TIMEOUT_S,
                    )
                else:
                    response = requests.post(url + "generations", json=request, timeout=IMAGE_REQUEST_TIMEOUT_S)
                response.raise_for_status()
                images = response.json()["data"]
                assert len(images) == 1
                with Image.open(io.BytesIO(base64.b64decode(images[0]["b64_json"], validate=True))) as image:
                    assert image.format == "PNG" and image.size == (2048, 2048)
                    rgb = image.convert("RGB")
                    assert any(low != high for low, high in rgb.getextrema())
                    rgb.save(output / f"native-{task}-{draw}.png")

    metrics: dict[str, dict[str, dict[str, float | str]]] = {}
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
            metrics[task][name] = {"ssim": ssim, "psnr_db": "inf" if psnr == math.inf else psnr}
    write_json_artifact(output / "metrics.json", metrics)
    for task in PROMPTS:
        scores = metrics[task]["native_vs_diffusers_first"]
        write_accuracy_result(
            model_id=MODEL_ID,
            dataset=f"image_similarity_{task}",
            metrics={"ssim": scores["ssim"], "psnr": float(scores["psnr_db"])},
            baseline=baseline_from_similarity_thresholds(tables[task]) if task in tables else None,
            result_dir=output,
            extra={
                "case": task,
                "revision": revision,
                "device_type": device_type,
                "calibration_only": calibration == "1",
                "threshold_status": policy["status"],
                "comparisons": metrics[task],
            },
        )
    if calibration == "1":
        pytest.skip(f"Diffusers calibration collected at {output}; accuracy thresholds were not enforced.")

    for task, floors in thresholds.items():
        for draw_name in ("first", "second"):
            scores = metrics[task][f"native_vs_diffusers_{draw_name}"]
            assert_similarity_metrics(
                model_name=f"MiniMax-H3-Image {task} ({draw_name} draw)",
                ssim_score=float(scores["ssim"]),
                psnr_score=float(scores["psnr_db"]),
                ssim_threshold=floors.ssim,
                psnr_threshold=floors.psnr,
            )
