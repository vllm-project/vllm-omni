# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Real-weight native/upstream T2I parity; see README.md."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

import diffusers
import numpy as np
import pytest
import requests
import torch
from huggingface_hub import snapshot_download
from PIL import Image

from tests.e2e.accuracy.helpers import compute_image_ssim_psnr, model_output_dir
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServer

pytestmark = [pytest.mark.full_model, pytest.mark.diffusion]

MODEL_ID = "Qwen/Qwen-Image-2.1"
MODEL_REVISION = "790c92633540aa0cb11d9abf19eb46d861714758"
REFERENCE_REVISION = "8d3c30bfda9b511c00992f40cff4170a5502814d"
GENERATION = {
    "prompt": "A ceramic teapot on a wooden table",
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 50,
    "true_cfg_scale": 1.0,
    "seed": 42,
    "generator_device": "cuda",
}
# Fixed fidelity target: PSNR 40 dB corresponds to normalized pixel RMSE 0.01.
# Provisional proposal, awaiting user selection; evidence and scope are in README.md.
SSIM_MIN = 0.99
PSNR_MIN = 40.0
ALPHA_MAE_MAX = 1 / 255


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _load_rgba(path: Path) -> Image.Image:
    with Image.open(path) as image:
        assert image.format == "PNG", (path, image.format)
        assert image.size == (GENERATION["width"], GENERATION["height"]), (path, image.size)
        assert image.mode == "RGBA", (path, image.mode)
        image.load()
        return image.copy()


def _reference(output_dir: Path, *, preflight: bool) -> None:
    root_value = os.environ.get("QWEN_IMAGE_21_REFERENCE_ROOT")
    assert root_value, "Prepare QWEN_IMAGE_21_REFERENCE_ROOT as described in README.md"
    root = Path(root_value).resolve()
    assert _git(root, "rev-parse", "HEAD") == REFERENCE_REVISION, "Unexpected reference revision"
    assert not _git(root, "status", "--porcelain", "--", "src/diffusers"), "Reference source must be clean"
    env = os.environ.copy()
    # Keep reference-only dependencies/source out of the native server's imports.
    env["PYTHONPATH"] = str(root / "src")
    command = [
        os.environ.get("QWEN_IMAGE_21_REFERENCE_PYTHON", sys.executable),
        str(Path(__file__).with_name("run_reference.py")),
        "--request",
        str(output_dir / "request.json"),
        "--output-dir",
        str(output_dir),
        "--reference-root",
        str(root),
    ]
    if preflight:
        command.append("--preflight")
    print(f"Reference command: {command}", flush=True)
    with (output_dir / ("preflight.log" if preflight else "reference.log")).open("w") as log:
        subprocess.run(command, env=env, cwd=output_dir, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)


def _native(model: Path, output_dir: Path) -> Image.Image:
    server_args = [
        "--num-gpus",
        "1",
        "--dtype",
        "bfloat16",
        "--enforce-eager",
        "--no-enable-cuda-graph-decode",
        "--diffusion-attention-backend",
        "CUDNN_ATTN",
        "--stage-init-timeout",
        "600",
        "--init-timeout",
        "900",
    ]
    request = {key: value for key, value in GENERATION.items() if key not in ("width", "height")} | {
        "model": str(model),
        "size": f"{GENERATION['width']}x{GENERATION['height']}",
        "n": 1,
        "response_format": "b64_json",
        "output_format": "png",
    }
    _write_json(output_dir / "native_request.json", {"server_args": server_args, "request": request})
    with OmniServer(
        str(model),
        server_args,
        use_omni=True,
        env_dict={"DIFFUSION_ATTENTION_BACKEND": "CUDNN_ATTN"},
        startup_timeout=1000,
    ) as server:
        requests.get(f"http://{server.host}:{server.port}/health", timeout=30).raise_for_status()
        response = requests.post(
            f"http://{server.host}:{server.port}/v1/images/generations", json=request, timeout=1200
        )
        response.raise_for_status()
        payload = response.json()
        assert isinstance(payload.get("data"), list) and len(payload["data"]) == 1, payload.keys()
        (output_dir / "native.png").write_bytes(base64.b64decode(payload["data"][0]["b64_json"], validate=True))
        return _load_rgba(output_dir / "native.png")


@hardware_test(res={"cuda": ["H100", "H200"]}, num_cards=1)
def test_qwen_image_21_matches_diffusers(accuracy_artifact_root: Path) -> None:
    assert torch.cuda.is_available(), "This real-weight accuracy gate requires CUDA"
    override = os.environ.get("QWEN_IMAGE_21_MODEL")
    if override:
        model = Path(override).resolve()
        revision = os.environ.get("QWEN_IMAGE_21_MODEL_REVISION", model.name)
        assert revision == MODEL_REVISION, "Local model override must identify the pinned checkpoint revision"
    else:
        model = Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION)).resolve()
    output_dir = Path(tempfile.mkdtemp(prefix="t2i-", dir=model_output_dir(accuracy_artifact_root, MODEL_ID)))
    repo = Path(__file__).resolve().parents[4]
    _write_json(
        output_dir / "request.json",
        {
            "model": str(model),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "reference_revision": REFERENCE_REVISION,
            "generation": GENERATION,
            "dtype": "bfloat16",
            "attention": "transformer-only cuDNN SDPA",
            "use_kv_cache": True,
            "native_enforce_eager": True,
            "device": torch.cuda.get_device_name(),
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "native_diffusers_source": diffusers.__file__,
            "native_diffusers_version": diffusers.__version__,
            "native_revision": _git(repo, "rev-parse", "HEAD"),
            "native_worktree": _git(repo, "status", "--short"),
            "test_source_sha256": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (Path(__file__), Path(__file__).with_name("run_reference.py"))
            },
            "packages": {
                name: version(name) for name in ("torch", "vllm", "diffusers", "transformers", "torchmetrics")
            },
        },
    )
    print(f"Qwen-Image 2.1 artifacts: {output_dir}", flush=True)
    # Import/asset checks finish in the actual reference environment before either model loads.
    _reference(output_dir, preflight=True)
    _reference(output_dir, preflight=False)
    reference = _load_rgba(output_dir / "reference.png")
    repeat = _load_rgba(output_dir / "reference_repeat.png")
    repeat_equal = np.array_equal(np.asarray(reference), np.asarray(repeat))
    _write_json(output_dir / "repeatability.json", {"identical_rgba_pixels": repeat_equal})
    assert repeat_equal, f"Reference is not repeatable; inspect {output_dir}"
    native = _native(model, output_dir)
    metrics = {}
    for mode in ("RGB", "RGBA"):
        ssim, psnr = compute_image_ssim_psnr(prediction=native, reference=reference, compare_mode=mode)
        metrics[mode] = {"ssim": ssim, "psnr_db": psnr if np.isfinite(psnr) else "inf"}
    alpha_error = float(
        np.abs(
            np.asarray(native.getchannel("A"), dtype=np.float32)
            - np.asarray(reference.getchannel("A"), dtype=np.float32)
        ).mean()
        / 255
    )
    metrics.update(
        {
            "alpha_mean_abs_error": alpha_error,
            "reference_repeat_identical": repeat_equal,
            "threshold_status": "proposal_awaiting_selection",
            "thresholds": {"ssim_min": SSIM_MIN, "psnr_db_min": PSNR_MIN, "alpha_mae_max": ALPHA_MAE_MAX},
            "output_sha256": {
                name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest()
                for name in ("native.png", "reference.png", "reference_repeat.png")
            },
        }
    )
    _write_json(output_dir / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2), flush=True)
    for mode in ("RGB", "RGBA"):
        assert metrics[mode]["ssim"] >= SSIM_MIN, f"{mode} SSIM below gate; artifacts: {output_dir}"
        assert float(metrics[mode]["psnr_db"]) >= PSNR_MIN, f"{mode} PSNR below gate; artifacts: {output_dir}"
    assert alpha_error <= ALPHA_MAE_MAX, f"Alpha mismatch; artifacts: {output_dir}"
