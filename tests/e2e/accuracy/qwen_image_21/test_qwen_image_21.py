# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Real-checkpoint IT2I/T2I comparison against the pinned Diffusers reference.

Run from the repository root in the installed serving environment::

    source tests/e2e/accuracy/qwen_image_21/setup_reference.sh
    pytest -sv \
        tests/e2e/accuracy/qwen_image_21/test_qwen_image_21.py \
        -m 'full_model and diffusion and H100 and cards_1' --run-level full_model

Model setup follows main (cached main in offline mode), unless QWEN_IMAGE_21_MODEL
names a complete local snapshot. Both runtimes use the same resolved directory.

For NPU A3 nightly, use '-m "full_model and diffusion and npu and A3 and cards_1"'.
The NPU native uses TORCH_SDPA with dense-mask prefill; the upstream reference
retains segmented PyTorch NPU SDPA prefill. Both use CPU RNG. This known numerical
difference requires separate NPU IT2I calibration; see thresholds.json for the
observed baseline and selection limits. CUDA retains cuDNN and CUDA RNG.
NPU calibration does not establish CUDA thresholds.

Normal execution enforces the user-selected bounds in thresholds.json with one
reference generation. A documented expected-XFAIL profile still executes both
runtimes and retains metrics/artifacts, but does not claim a pixel-accuracy pass.
Set QWEN_IMAGE_21_CALIBRATE=1 to generate two references and collect calibration
evidence without enforcing bounds; this reports XFAIL.
"""

from __future__ import annotations

import base64
import json
import math
import os
import shutil
import subprocess
import tempfile
from importlib.metadata import version
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import requests
from PIL import Image

from tests.e2e.accuracy.helpers import (
    SimilarityThresholds,
    assert_similarity_metrics,
    baseline_from_similarity_thresholds,
    measure_image_similarity,
    model_output_dir,
    resolve_device_profile,
    resolve_similarity_thresholds,
    write_accuracy_result,
)
from tests.e2e.accuracy.qwen_image_21.contract import (
    CASES,
    EXECUTION_SETTINGS,
    INPUTS,
    MODEL_ID,
    ROOT,
    device_preflight,
    sha256,
    validate_model,
)
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServer
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.full_model, pytest.mark.diffusion]
HERE = Path(__file__).resolve().parent
SERVER_ARGS = [
    "--num-gpus",
    "1",
    "--dtype",
    "bfloat16",
    "--enforce-eager",
    "--max-num-seqs",
    "1",
    "--stage-init-timeout",
    "600",
    "--init-timeout",
    "1200",
]


def _reference(request_path: Path, output: Path, *, preflight: bool, repeats: int = 1) -> dict:
    python = os.environ.get("QWEN_IMAGE_21_REFERENCE_PYTHON")
    assert python and Path(python).is_file(), "Source setup_reference.sh before running accuracy"
    command = [python, str(HERE / "run_reference.py"), "--request", str(request_path), "--output-dir", str(output)]
    if preflight:
        command.append("--preflight-only")
    else:
        command.extend(["--repeats", str(repeats)])
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    name = "preflight" if preflight else "reference"
    with (output / f"{name}.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=3600)
    assert result.returncode == 0, f"Reference {name} failed: {output / (name + '.log')}"
    return json.loads((output / "reference_preflight.json").read_text(encoding="utf-8"))


def _native(request: dict, output: Path) -> None:
    payload = {
        "prompt": request["prompt"],
        "negative_prompt": request["negative_prompt"],
        "size": f"{request['width']}x{request['height']}",
        "n": 1,
        "response_format": "b64_json",
        "output_format": "png",
        "num_inference_steps": request["num_inference_steps"],
        "true_cfg_scale": request["true_cfg_scale"],
        "seed": request["seed"],
        "generator_device": request["generator_device"],
    }
    payload = {key: value for key, value in payload.items() if value is not None}
    serve_args = SERVER_ARGS + ["--diffusion-attention-backend", request["attention_backend"]]
    with OmniServer(model=request["model"], serve_args=serve_args, startup_timeout=1200) as server:
        payload["model"] = server.model
        if request["inputs"]:
            response = requests.post(
                f"http://{server.host}:{server.port}/v1/images/edits",
                data=payload,
                files=[
                    ("image", (Path(entry["path"]).name, Path(entry["path"]).read_bytes(), "image/png"))
                    for entry in request["inputs"]
                ],
                timeout=1200,
            )
        else:
            response = requests.post(
                f"http://{server.host}:{server.port}/v1/images/generations",
                json=payload,
                timeout=1200,
            )
        response.raise_for_status()
        body = response.json()
        assert isinstance(body, dict) and isinstance(body.get("data"), list), body
        assert len(body["data"]) == 1, "Exactly one RGBA output is required"
        raw = base64.b64decode(body["data"][0]["b64_json"], validate=True)
        (output / "native.png").write_bytes(raw)
        with Image.open(BytesIO(raw)) as image:
            image.load()
            assert image.format == "PNG" and image.mode == "RGBA", (image.format, image.mode)
            assert image.size == (request["width"], request["height"]), image.size


def _scores(prediction: Image.Image, reference: Image.Image) -> dict[str, float]:
    ssim, psnr = measure_image_similarity(
        model_name="Qwen-Image 2.1",
        vllm_image=prediction,
        diffusers_image=reference,
        compare_mode="RGBA",
    )
    alpha = (
        np.abs(np.asarray(prediction, dtype=np.float32)[..., 3] - np.asarray(reference, dtype=np.float32)[..., 3]) / 255
    )
    assert math.isfinite(ssim) and not math.isnan(psnr)
    return {"ssim": ssim, "psnr": psnr, "alpha_mae": float(alpha.mean())}


@pytest.mark.parametrize("case_name", ["it2i", "t2i"])
@hardware_test(res={"cuda": "H100", "npu": "A3", "rocm": "MI325"}, num_cards=1)
def test_qwen_image_21_matches_reference(accuracy_artifact_root: Path, run_level: str, case_name: str) -> None:
    """Own both generations and fresh artifacts; no prerequisite test or old output reuse."""
    assert run_level == "full_model", "Accuracy requires --run-level full_model"
    parent = model_output_dir(accuracy_artifact_root, "qwen_image_21")
    output = Path(tempfile.mkdtemp(prefix=f"{case_name}-", dir=parent))
    print(f"Qwen-Image 2.1 artifacts: {output}", flush=True)
    calibration = os.environ.get("QWEN_IMAGE_21_CALIBRATE") == "1"
    policy = json.loads((HERE / "thresholds.json").read_text(encoding="utf-8"))
    bounds = policy[case_name]
    device_type = "rocm" if current_omni_platform.is_rocm() else current_omni_platform.device_type
    device = device_preflight(device_type)
    device_name = device["device"]
    profile = resolve_device_profile(device_name, profiles=bounds)
    xfail_profiles = policy.get("expected_xfail", {}).get(case_name, {})
    xfail_profile = resolve_device_profile(device_name, profiles=xfail_profiles)
    expected_xfail = xfail_profiles.get(xfail_profile)
    if expected_xfail:
        assert isinstance(expected_xfail, str) and expected_xfail
        assert f"{xfail_profile}.{case_name}" in policy["calibration_notes"]
    if not calibration:
        assert policy["status"] == "selected", (
            "No user-selected Qwen-Image 2.1 gate. Select bounds in thresholds.json after calibration; "
            "QWEN_IMAGE_21_CALIBRATE=1 collects evidence and reports XFAIL, never an accuracy pass."
        )
        assert device_type != "npu" or profile != "default", "NPU requires an explicit device threshold profile"
        assert all(isinstance(v, (float, int)) and math.isfinite(v) for v in bounds[profile].values()), (
            f"No finite, user-selected thresholds for {device_name} ({profile})"
        )
    model_value = os.environ.get("QWEN_IMAGE_21_MODEL")
    assert model_value, "Source setup_reference.sh, or set QWEN_IMAGE_21_MODEL to a frozen complete snapshot"
    model = Path(model_value).resolve()
    assert model.is_dir(), model
    request = dict(
        CASES[case_name],
        model=str(model),
        seed=42,
        num_inference_steps=40,
        inputs=[],
        device_type=device_type,
        **EXECUTION_SETTINGS[device_type],
    )
    for name, expected_hash in INPUTS if case_name == "it2i" else ():
        source = ROOT / "tests/assets/qwen_image_edit" / name
        assert sha256(source) == expected_hash, f"Input changed: {source}"
        target = output / name
        shutil.copyfile(source, target)
        with Image.open(target) as image:
            image.load()
            w, h = image.size
            processed_w = round(math.sqrt(1024**2 * w / h) / 32) * 32
            processed_h = round(math.sqrt(1024**2 * h / w) / 32) * 32
            request["inputs"].append(
                {
                    "path": str(target),
                    "sha256": expected_hash,
                    "size": [w, h],
                    "mode": image.mode,
                    "processed_size": [processed_w, processed_h],
                }
            )
    # Ephemeral IPC for run_reference.py; not retained as a case artifact.
    request_file = tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False)
    try:
        json.dump(request, request_file)
        request_file.close()
        request_path = Path(request_file.name)
        native_assets = validate_model(model)
        # Run the actual reference interpreter's import and asset checks before either model loads.
        preflight = _reference(request_path, output, preflight=True)
        assert preflight["assets"] == native_assets
        shared_packages = ["torch", "transformers", "accelerate", "Pillow", "huggingface-hub"]
        if device_type == "npu":
            shared_packages.append("torch-npu")
        for package in shared_packages:
            assert preflight["versions"][package] == version(package), f"Mismatched shared dependency: {package}"
        assert all(preflight[key] == value for key, value in device.items())
        _native(request, output)
        reference_repeats = 2 if calibration else 1
        _reference(request_path, output, preflight=False, repeats=reference_repeats)
    finally:
        Path(request_file.name).unlink(missing_ok=True)
    with (
        Image.open(output / "native.png") as candidate,
        Image.open(output / "reference_0.png") as reference,
    ):
        for image in (candidate, reference):
            image.load()
            assert image.mode == "RGBA" and image.size == (request["width"], request["height"])
        candidate_scores = _scores(candidate, reference)
        comparisons = {"candidate_vs_reference": candidate_scores}
        if calibration:
            with Image.open(output / "reference_1.png") as repeated:
                repeated.load()
                assert repeated.mode == "RGBA" and repeated.size == reference.size
                comparisons["reference_repeatability"] = _scores(repeated, reference)
    selected = None
    table = None
    if not calibration:
        # Uncalibrated profiles may remain null while a different device is gated.
        table = {key: SimilarityThresholds(**row) for key, row in bounds.items()}
        selected = resolve_similarity_thresholds(table, device_name=device_name)
    write_accuracy_result(
        model_id=MODEL_ID,
        dataset="image_similarity",
        metrics={
            "ssim": candidate_scores["ssim"],
            "psnr": candidate_scores["psnr"],
            "alpha_mae": candidate_scores.get("alpha_mae"),
        },
        baseline=baseline_from_similarity_thresholds(table) if table is not None else None,
        extra={
            "case": case_name,
            "device": device_name,
            "device_type": device_type,
            "profile": profile,
            "threshold_status": policy["status"],
            "resolved_bounds": bounds.get(profile, bounds["default"]),
            "expected_xfail_profile": xfail_profile if expected_xfail else None,
            "expected_xfail_reason": expected_xfail,
            "calibration_only": calibration,
            "reference_generations": reference_repeats,
            "comparisons": comparisons,
            "artifacts": str(output),
        },
    )
    if calibration:
        pytest.xfail(f"Calibration collected; thresholds were not enforced. Artifacts: {output}")
    if expected_xfail:
        pytest.xfail(f"{expected_xfail} Artifacts: {output}")
    assert selected is not None
    assert_similarity_metrics(
        model_name=f"Qwen-Image 2.1 {case_name}; artifacts: {output}",
        ssim_score=candidate_scores["ssim"],
        psnr_score=candidate_scores["psnr"],
        ssim_threshold=selected.ssim,
        psnr_threshold=selected.psnr,
    )
