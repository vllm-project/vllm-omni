# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E offline inference tests for LingBot-MAP 3D reconstruction.

Runs the LingbotMapPipeline against a fixed subset of the courthouse example
images and compares the predicted 3D outputs (depth, depth confidence, camera
extrinsic/intrinsic, pose encoding) against a stored baseline. Test passes
when every tensor matches the baseline within a per-key tolerance.

The LingbotMap pipeline emits its 3D predictions through
``OmniRequestOutput.custom_output``; ``multimodal_output`` is intentionally
checked too, so this test guards against future regressions in the
custom→multimodal merge path.

Maintainer controls:
    LINGBOT_REGENERATE_BASELINE=1 — overwrite the baseline file from this run
    LINGBOT_MODEL_DIR=...         — override the checkpoint location
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import pytest
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_TEST_DIR = Path(__file__).resolve().parent
_ASSETS_DIR = _TEST_DIR.parents[1] / "assets" / "lingbot_map"
_IMAGE_SUBSET = _ASSETS_DIR / "courthouse_subset"
_BASELINE_DIR = _ASSETS_DIR / "baselines" / "courthouse_subset"
_BASELINE_FILE = _BASELINE_DIR / "predictions.pt"

# Local checkpoint location. Override via ``LINGBOT_MODEL_DIR`` env var.
_DEFAULT_MODEL_DIR = "/tmp/lingbot-map-local"


# ---------------------------------------------------------------------------
# Per-key tolerances
# ---------------------------------------------------------------------------
#
# 3D reconstruction is numerically noisy: depth scale, pose encoding, and
# confidence values vary slightly run-to-run (cuDNN nondeterminism, autocast
# rounding). Tolerances are sized to absorb that noise without masking a real
# regression. Per-key keeps a single bad output from being hidden by a loose
# global tolerance.
#
_TOLERANCES: dict[str, dict[str, float]] = {
    # Frame-aligned dense tensors — moderate atol because depth ranges can be
    # in the tens of meters.
    "depth":             {"rtol": 1e-2, "atol": 5e-2},
    "depth_conf":        {"rtol": 1e-2, "atol": 1e-2},
    # Camera matrices — small, tightly clustered values, so tighten atol.
    "extrinsic":         {"rtol": 1e-3, "atol": 1e-3},
    "intrinsic":         {"rtol": 1e-3, "atol": 1e-3},
    # 9-dim pose encoding before extrinsic/intrinsic conversion.
    "pose_enc":          {"rtol": 1e-3, "atol": 1e-3},
}


# ---------------------------------------------------------------------------
# Skip guards (resolved at collection time)
# ---------------------------------------------------------------------------

def _model_dir() -> str:
    return os.environ.get("LINGBOT_MODEL_DIR", _DEFAULT_MODEL_DIR)


def _skip_reasons() -> list[str]:
    reasons: list[str] = []
    md = _model_dir()
    if not Path(md).is_dir():
        reasons.append(f"LingBot-MAP checkpoint dir not present: {md}")
    if not _IMAGE_SUBSET.is_dir() or not any(_IMAGE_SUBSET.glob("*.png")):
        reasons.append(f"Test image subset missing: {_IMAGE_SUBSET}")
    if not torch.cuda.is_available():
        reasons.append("CUDA device required for LingBot-MAP inference")
    return reasons


pytestmark = pytest.mark.skipif(
    bool(_skip_reasons()), reason="; ".join(_skip_reasons()) or "skipped",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_image_paths() -> list[str]:
    """Sorted image paths (matches infer_lingbot.py's discovery order)."""
    exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
    paths = sorted(
        p for p in glob.glob(str(_IMAGE_SUBSET / "*"))
        if p.endswith(exts)
    )
    assert paths, f"No images found in {_IMAGE_SUBSET}"
    return paths


def _extract_output_tensors(request_output) -> dict[str, torch.Tensor]:
    """Collect 3D tensors from both multimodal_output and custom_output.

    LingbotMapPipeline currently writes its 3D predictions into
    ``custom_output``; ``multimodal_output`` is the documented destination.
    Reading from both makes the test robust to either side of that gap.
    """
    out: dict[str, torch.Tensor] = {}
    mm = getattr(request_output, "multimodal_output", None)
    if isinstance(mm, dict):
        for k, v in mm.items():
            if isinstance(v, torch.Tensor):
                out[k] = v
    co = getattr(request_output, "custom_output", None)
    if isinstance(co, dict):
        for k, v in co.items():
            if isinstance(v, torch.Tensor) and k not in out:
                out[k] = v
    return out


def _run_inference() -> tuple[dict[str, torch.Tensor], object]:
    """Run inference once; return (tensors, request_output)."""
    # Imported lazily so this module is collectable without vllm_omni installed.
    from vllm_omni.entrypoints.omni import Omni

    pil_images = [Image.open(p).convert("RGB") for p in _load_image_paths()]
    prompt = {"prompt": "", "multi_modal_data": {"image": pil_images}}

    omni = Omni(model=_model_dir())
    outputs = omni.generate(prompt)
    request_output = outputs[0].request_output
    tensors = _extract_output_tensors(request_output)
    assert tensors, (
        f"Expected at least one 3D output tensor from LingbotMap pipeline; "
        f"got multimodal_output={getattr(request_output, 'multimodal_output', None)!r} "
        f"custom_output={getattr(request_output, 'custom_output', None)!r}"
    )
    return tensors, request_output


def _to_cpu_tensor(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", torch.float32).contiguous()
    return torch.as_tensor(value, dtype=torch.float32)


def _compare_tensor(name: str, predicted, expected) -> None:
    """Compare one tensor against its baseline using the per-key tolerance."""
    pred = _to_cpu_tensor(predicted)
    exp = _to_cpu_tensor(expected)
    assert pred.shape == exp.shape, (
        f"[{name}] shape mismatch: got {tuple(pred.shape)}, expected {tuple(exp.shape)}"
    )

    tol = _TOLERANCES.get(name, {"rtol": 1e-3, "atol": 1e-3})

    if torch.allclose(pred, exp, rtol=tol["rtol"], atol=tol["atol"], equal_nan=True):
        return

    diff = (pred - exp).abs()
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    scale = float(exp.abs().max())
    pytest.fail(
        f"[{name}] tensor mismatch beyond tolerance "
        f"(rtol={tol['rtol']}, atol={tol['atol']}):\n"
        f"  shape={tuple(pred.shape)} dtype={pred.dtype}\n"
        f"  max|Δ|={max_abs:.6g}  mean|Δ|={mean_abs:.6g}  max|expected|={scale:.6g}"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

# Keys the LingbotMap pipeline produces today. Update when the model changes.
_REQUIRED_KEYS: frozenset[str] = frozenset(
    {"depth", "depth_conf", "extrinsic", "intrinsic", "pose_enc"}
)


def test_lingbot_map_inference_matches_baseline():
    """Run LingBot-MAP on the image subset; compare each output to baseline.

    With ``LINGBOT_REGENERATE_BASELINE=1`` set, the current outputs overwrite
    the baseline file instead of being compared.
    """
    tensors, _ = _run_inference()

    missing = _REQUIRED_KEYS - set(tensors.keys())
    assert not missing, (
        f"Inference output missing required keys: {sorted(missing)}. "
        f"Got: {sorted(tensors.keys())}"
    )

    if os.environ.get("LINGBOT_REGENERATE_BASELINE") == "1":
        _BASELINE_DIR.mkdir(parents=True, exist_ok=True)
        baseline = {k: _to_cpu_tensor(tensors[k]) for k in _REQUIRED_KEYS}
        torch.save(baseline, _BASELINE_FILE)
        pytest.skip(f"Baseline regenerated at {_BASELINE_FILE}; rerun without "
                    f"LINGBOT_REGENERATE_BASELINE to compare")

    assert _BASELINE_FILE.is_file(), (
        f"Baseline file missing: {_BASELINE_FILE}\n"
        f"Generate it with: LINGBOT_REGENERATE_BASELINE=1 pytest "
        f"tests/e2e/offline_inference/test_lingbot_map.py"
    )
    baseline = torch.load(_BASELINE_FILE, map_location="cpu", weights_only=True)

    for key in sorted(_REQUIRED_KEYS):
        assert key in baseline, f"Baseline missing key: {key}"
        _compare_tensor(key, tensors[key], baseline[key])


def test_lingbot_map_output_schema():
    """Schema check — verifies output ranks/shapes are internally consistent."""
    tensors, _ = _run_inference()

    # depth: [S, H, W, 1]
    depth = _to_cpu_tensor(tensors["depth"])
    assert depth.ndim == 4 and depth.shape[-1] == 1, (
        f"depth must be [S, H, W, 1], got {tuple(depth.shape)}"
    )
    S, H, W, _ = depth.shape

    # depth_conf: [S, H, W]
    depth_conf = _to_cpu_tensor(tensors["depth_conf"])
    assert depth_conf.shape == (S, H, W), (
        f"depth_conf must be [{S}, {H}, {W}], got {tuple(depth_conf.shape)}"
    )

    # extrinsic: [S, 3, 4] camera-to-world
    extrinsic = _to_cpu_tensor(tensors["extrinsic"])
    assert extrinsic.shape == (S, 3, 4), (
        f"extrinsic must be [{S}, 3, 4], got {tuple(extrinsic.shape)}"
    )

    # intrinsic: [S, 3, 3]
    intrinsic = _to_cpu_tensor(tensors["intrinsic"])
    assert intrinsic.shape == (S, 3, 3), (
        f"intrinsic must be [{S}, 3, 3], got {tuple(intrinsic.shape)}"
    )

    # pose_enc: [S, 9]
    pose_enc = _to_cpu_tensor(tensors["pose_enc"])
    assert pose_enc.shape == (S, 9), (
        f"pose_enc must be [{S}, 9], got {tuple(pose_enc.shape)}"
    )

    # Number of frames must equal the number of input images.
    assert S == len(_load_image_paths()), (
        f"Frame count mismatch: depth has {S} frames, "
        f"but {len(_load_image_paths())} images were submitted"
    )
