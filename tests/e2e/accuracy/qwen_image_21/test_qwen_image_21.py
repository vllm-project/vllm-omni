# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Real-checkpoint IT2I/T2I comparison against the pinned Diffusers reference.

Run from the repository root in the installed serving environment::

    source tests/e2e/accuracy/qwen_image_21/setup_reference.sh
    pytest -sv \
        tests/e2e/accuracy/qwen_image_21/test_qwen_image_21.py \
        -m 'full_model and diffusion and H100 and cards_1' --run-level full_model

Model setup follows main (cached main in offline mode), unless QWEN_IMAGE_21_MODEL
names a complete local snapshot. Both runtimes use the same resolved directory;
model_assets.json records its path and snapshot revision when available.

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
import sys
import tempfile
from importlib.metadata import version
from io import BytesIO
from pathlib import Path

import huggingface_hub
import numpy as np
import pytest
import requests
import torch
from PIL import Image

from tests.e2e.accuracy.helpers import (
    SimilarityThresholds,
    assert_similarity_metrics,
    compute_image_ssim_psnr,
    model_output_dir,
    resolve_device_profile,
    resolve_similarity_thresholds,
    write_json_artifact,
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
        try:
            result = subprocess.run(command, cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=3600)
        except subprocess.TimeoutExpired:
            write_json_artifact(
                output / f"{name}_exit.json", {"command": command, "exit_status": "timeout", "timeout_s": 3600}
            )
            raise
    write_json_artifact(output / f"{name}_exit.json", {"command": command, "exit_status": result.returncode})
    assert result.returncode == 0, f"Reference {name} failed: {output / (name + '.log')}"
    return json.loads((output / "reference_preflight.json").read_text(encoding="utf-8"))


def _source_provenance() -> dict[str, str]:
    try:
        return {
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
            ).strip(),
            "git_status": subprocess.check_output(
                ["git", "status", "--short"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
            ),
        }
    except (OSError, subprocess.CalledProcessError):
        buildkite_commit = os.environ.get("BUILDKITE_COMMIT")
        if not buildkite_commit:
            raise RuntimeError("Cannot identify the tested source: no Git checkout or BUILDKITE_COMMIT") from None
        return {
            "git_head": buildkite_commit,
            "git_status": "unavailable: repository metadata is not mounted",
        }


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
    write_json_artifact(
        output / "native_launch.json",
        {
            "model": request["model"],
            "serve_args": serve_args,
            "request": payload,
            "python": sys.executable,
            **_source_provenance(),
            "versions": {name: version(name) for name in ("vllm", "diffusers", "torch", "transformers", "Pillow")},
        },
    )
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
        (output / "http_response.json").write_text(response.text, encoding="utf-8")
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
    ssim, psnr = compute_image_ssim_psnr(prediction=prediction, reference=reference, compare_mode="RGBA")
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
    write_json_artifact(output / "threshold_policy.json", policy)
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
    request_path = output / "request.json"
    write_json_artifact(request_path, request)
    native_assets = validate_model(model)
    write_json_artifact(
        output / "model_assets.json",
        {
            "expected_public_model": MODEL_ID,
            "resolved_revision": model.name if model.parent.name == "snapshots" else None,
            **native_assets,
        },
    )
    # Run the actual reference interpreter's import and asset checks before either model loads.
    preflight = _reference(request_path, output, preflight=True)
    assert preflight["assets"] == native_assets
    shared_packages = ["torch", "transformers", "accelerate", "Pillow", "huggingface-hub"]
    if device_type == "npu":
        shared_packages.append("torch-npu")
    for package in shared_packages:
        assert preflight["versions"][package] == version(package), f"Mismatched shared dependency: {package}"
    assert all(preflight[key] == value for key, value in device.items())
    write_json_artifact(
        output / "comparison_preflight.json",
        {
            **device,
            "cuda": torch.version.cuda,
            "torchmetrics": version("torchmetrics"),
            "numpy": version("numpy"),
            "huggingface_hub_path": str(Path(huggingface_hub.__file__).resolve()),
            "python": sys.executable,
        },
    )
    _native(request, output)
    reference_repeats = 2 if calibration else 1
    _reference(request_path, output, preflight=False, repeats=reference_repeats)
    artifact_names = ["native.png", "reference_0.png"]
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
            artifact_names.append("reference_1.png")
    selected = None
    if not calibration:
        # Uncalibrated profiles may remain null while a different device is gated.
        table = {key: SimilarityThresholds(**row) for key, row in bounds.items()}
        selected = resolve_similarity_thresholds(table, device_name=device_name)
    write_json_artifact(
        output / "metrics.json",
        {
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
            "representation": "RGBA uint8 / 255, no compositing or resizing",
            **{
                comparison: {metric: str(value) if math.isinf(value) else value for metric, value in scores.items()}
                for comparison, scores in comparisons.items()
            },
            "artifacts": {name: sha256(output / name) for name in artifact_names},
        },
    )
    if calibration:
        pytest.xfail(f"Calibration collected; thresholds were not enforced. Metrics: {output / 'metrics.json'}")
    if expected_xfail:
        pytest.xfail(f"{expected_xfail} Metrics: {output / 'metrics.json'}")
    assert selected is not None
    assert_similarity_metrics(
        model_name=f"Qwen-Image 2.1 {case_name}; artifacts: {output}",
        ssim_score=candidate_scores["ssim"],
        psnr_score=candidate_scores["psnr"],
        ssim_threshold=selected.ssim,
        psnr_threshold=selected.psnr,
    )
