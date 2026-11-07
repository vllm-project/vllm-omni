# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Replay ROCm device metadata without changing the observed pixel results."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from tests.e2e.accuracy.qwen_image_21 import test_qwen_image_21 as accuracy

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize(
    "name,architecture,expected",
    [
        ("AMD Instinct MI300X", "gfx942:sramecc+:xnack-", "gfx942"),
        ("gfx942:sramecc+:xnack-", "gfx942:sramecc+:xnack-", "gfx942"),
        ("", "gfx942:sramecc+:xnack-", "gfx942"),
        ("gfx942:sramecc+:xnack-", None, "gfx942"),
        ("AMD Instinct MI300X", None, "default"),
        ("AMD Instinct MI300X", "gfx950:sramecc+:xnack-", "default"),
        ("gfx942", "gfx950", "default"),
        ("gfx942", "gfx9420", "default"),
    ],
)
def test_only_the_existing_exact_rocm_architecture_selects_observational_policy(name, architecture, expected):
    policy = json.loads((accuracy.HERE / "thresholds.json").read_text())
    profiles = policy["expected_xfail"]["it2i"]
    device = {"device_type": "rocm", "device": name}
    if architecture is not None:
        device["device_arch"] = architecture
    before = dict(device)
    assert accuracy._expected_xfail_profile(device, profiles) == expected
    assert device == before
    # The architecture has no observational T2I rule; its strict gate remains.
    assert accuracy._expected_xfail_profile(device, policy["expected_xfail"].get("t2i", {})) == "default"


@pytest.mark.parametrize("platform,name", [("cuda", "NVIDIA H100 80GB HBM3"), ("npu", "Ascend910B1")])
def test_other_platforms_keep_marketing_name_profile_lookup(platform, name):
    profiles = {"H100": "CUDA rule", "Ascend": "NPU rule", "gfx942": "ROCm rule"}
    device = {"device_type": platform, "device": name, "device_arch": "gfx942"}
    assert accuracy._expected_xfail_profile(device, profiles) == ("H100" if platform == "cuda" else "Ascend")


@pytest.mark.parametrize(
    "case,architecture,observational",
    [("it2i", "gfx942:sramecc+:xnack-", True), ("t2i", "gfx942:sramecc+:xnack-", False), ("it2i", "gfx950", False)],
)
def test_comparison_and_failed_metrics_are_retained_before_existing_xfail(
    tmp_path, monkeypatch, case, architecture, observational
):
    device = {
        "device_type": "rocm",
        "device": "AMD Instinct MI300X",
        "device_arch": architecture,
        "memory_bytes": 206141652992,
    }
    calls = []
    monkeypatch.setenv("QWEN_IMAGE_21_MODEL", str(tmp_path))
    monkeypatch.setenv("ACC_BENCH_RESULT_DIR", str(tmp_path / "results"))
    monkeypatch.delenv("QWEN_IMAGE_21_CALIBRATE", raising=False)
    monkeypatch.setattr(accuracy, "current_omni_platform", SimpleNamespace(is_rocm=lambda: True, device_type="cuda"))
    monkeypatch.setattr(accuracy, "device_preflight", lambda _: dict(device))
    monkeypatch.setattr(accuracy, "validate_model", lambda _: {})
    monkeypatch.setattr(accuracy, "version", lambda _: "test-version")
    monkeypatch.setattr(accuracy, "INPUTS", ())
    measured = {"ssim": 0.9820287227630615, "psnr": 33.04202651977539, "alpha_mae": 0.002607607515528798}
    monkeypatch.setattr(accuracy, "_scores", lambda *_: dict(measured))

    def native(request, output):
        calls.append("native")
        Image.new("RGBA", (request["width"], request["height"])).save(output / "native.png")

    def reference(request_path, output, *, preflight, repeats=1):
        calls.append("preflight" if preflight else "reference")
        request = json.loads(request_path.read_text())
        if not preflight:
            assert repeats == 1
            Image.new("RGBA", (request["width"], request["height"])).save(output / "reference_0.png")
        return {
            **device,
            "assets": {},
            "versions": {
                package: "test-version"
                for package in ("torch", "transformers", "accelerate", "Pillow", "huggingface-hub")
            },
        }

    monkeypatch.setattr(accuracy, "_native", native)
    monkeypatch.setattr(accuracy, "_reference", reference)
    expected_error = pytest.xfail.Exception if observational else AssertionError
    with pytest.raises(expected_error):
        accuracy.test_qwen_image_21_matches_reference(tmp_path / "artifacts", "full_model", case)
    assert calls == ["preflight", "native", "reference"]
    report = json.loads(next((tmp_path / "results").glob("result_acc_*.json")).read_text())
    assert report["comparisons"]["candidate_vs_reference"] == measured
    assert {metric: report[metric] for metric in measured} == measured
    assert report["resolved_bounds"] == {"ssim": 0.99, "psnr": 40.0}
    assert report["device"] == device["device"]
    assert report["expected_xfail_profile"] == ("gfx942" if observational else None)
    assert not report["calibration_only"] and report["reference_generations"] == 1
    artifact_dir = Path(report["artifacts"])
    assert artifact_dir.is_relative_to(tmp_path / "artifacts")
    assert {path.name for path in artifact_dir.glob("*.png")} == {"native.png", "reference_0.png"}
