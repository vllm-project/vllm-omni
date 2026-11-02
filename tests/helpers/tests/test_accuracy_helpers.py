# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
import math
from datetime import datetime
from functools import partial

import pytest
from PIL import Image

from tests.e2e.accuracy import helpers

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    ("device_name", "profile"),
    [("NVIDIA H100 80GB HBM3", "H100"), ("NVIDIA B200", "default"), ("Ascend A3", "default")],
)
def test_similarity_thresholds_device_and_default(device_name, profile):
    table = {
        "H100": helpers.SimilarityThresholds(ssim=0.98, psnr=35.0),
        "default": helpers.SimilarityThresholds(ssim=0.99, psnr=40.0),
    }
    resolved_profile = helpers.resolve_device_profile(device_name, profiles=table)
    assert resolved_profile == profile
    assert helpers.resolve_similarity_thresholds(table, resolved_profile) == table[profile]
    assert helpers.resolve_similarity_thresholds(table, device_name=device_name) == table[profile]


def test_similarity_thresholds_require_default():
    with pytest.raises(ValueError, match="must include a 'default'"):
        helpers.resolve_similarity_thresholds({"H100": helpers.SimilarityThresholds(ssim=0.99, psnr=40.0)})


def test_write_json_artifact_round_trip(tmp_path):
    path = tmp_path / "metrics.json"
    value = {"prompt": "茶壶", "RGB": {"ssim": 1.0, "psnr_db": "inf"}}
    helpers.write_json_artifact(path, value)
    text = path.read_bytes().decode("utf-8")
    assert json.loads(text) == value
    assert '\n  "RGB": {' in text
    assert text.endswith("\n")


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
def test_write_json_artifact_rejects_nonfinite_without_overwriting(tmp_path, nonfinite):
    path = tmp_path / "metrics.json"
    helpers.write_json_artifact(path, {"ssim": 0.99})
    with pytest.raises(ValueError, match="Out of range float values"):
        helpers.write_json_artifact(path, {"ssim": nonfinite})
    assert json.loads(path.read_text(encoding="utf-8")) == {"ssim": 0.99}


@pytest.mark.parametrize("psnr", [40.0, float("inf")])
def test_similarity_metrics_accept_inclusive_bounds_and_identical_psnr(psnr):
    helpers.assert_similarity_metrics(
        model_name="example (RGBA)",
        ssim_score=0.99,
        psnr_score=psnr,
        ssim_threshold=0.99,
        psnr_threshold=40.0,
    )


@pytest.mark.parametrize(
    ("ssim", "psnr", "failed_metric"),
    [
        (math.nextafter(0.99, 0.0), 40.0, "SSIM"),
        (0.99, math.nextafter(40.0, 0.0), "PSNR"),
        (float("nan"), 40.0, "SSIM"),
        (0.99, float("nan"), "PSNR"),
    ],
)
def test_similarity_metrics_reject_below_bounds_and_nan(ssim, psnr, failed_metric):
    with pytest.raises(AssertionError, match=rf"{failed_metric} below threshold for example \(RGBA\)"):
        helpers.assert_similarity_metrics(
            model_name="example (RGBA)",
            ssim_score=ssim,
            psnr_score=psnr,
            ssim_threshold=0.99,
            psnr_threshold=40.0,
        )


def test_write_accuracy_result_persists_scores_and_metadata(tmp_path, monkeypatch):
    import tests.helpers.hardware as hardware

    monkeypatch.setattr(hardware, "get_runtime_resource_label", lambda: "H100")
    monkeypatch.setenv("BUILDKITE_BRANCH", "fix/diffusion-accuracy-write-results")
    when = datetime(2026, 11, 2, 9, 30, 0)
    path = helpers.write_accuracy_result(
        model_id="Qwen/Qwen-Image",
        dataset="image_similarity",
        metrics={"ssim": 0.991, "psnr": float("inf"), "seed_tts_content_error_mean": 0.12},
        baseline={
            "H100": {"ssim": 0.99, "psnr": 40.0, "max_seed_tts_mean_wer": 0.2},
            "A3": {"ssim": 0.95, "psnr": 35.0, "max_seed_tts_mean_wer": 0.25},
        },
        result_dir=tmp_path,
        when=when,
        extra={"label": "qwen_image"},
    )

    assert path.name == "result_acc_Qwen-Image_image_similarity_20261102_093000.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["model_id"] == "Qwen/Qwen-Image"
    assert saved["dataset"] == "image_similarity"
    assert saved["ssim"] == 0.991
    assert saved["psnr"] == "inf"
    assert saved["seed_tts_content_error_mean"] == 0.12
    assert saved["baseline"] == {
        "H100": {"ssim": 0.99, "psnr": 40.0, "max_seed_tts_mean_wer": 0.2},
        "A3": {"ssim": 0.95, "psnr": 35.0, "max_seed_tts_mean_wer": 0.25},
    }
    assert saved["Hardware"] == "H100"
    assert saved["branch"] == "fix/diffusion-accuracy-write-results"
    assert saved["label"] == "qwen_image"


def test_write_accuracy_result_wraps_flat_baseline_under_hardware(tmp_path, monkeypatch):
    import tests.helpers.hardware as hardware

    monkeypatch.setattr(hardware, "get_runtime_resource_label", lambda: "L4")
    monkeypatch.delenv("BUILDKITE_BRANCH", raising=False)
    helpers.write_accuracy_result(
        model_id="Qwen-Image",
        dataset="image_similarity",
        metrics={"ssim": 0.5, "psnr": 20.0},
        baseline={"ssim": 0.99, "psnr": 40.0},
        result_dir=tmp_path,
        when=datetime(2026, 11, 2, 10, 0, 0),
    )
    with pytest.raises(AssertionError, match="SSIM below threshold"):
        helpers.assert_similarity_metrics(
            model_name="Qwen-Image",
            ssim_score=0.5,
            psnr_score=20.0,
            ssim_threshold=0.99,
            psnr_threshold=40.0,
        )

    paths = list(tmp_path.glob("result_acc_Qwen-Image_image_similarity_*.json"))
    assert len(paths) == 1
    saved = json.loads(paths[0].read_text(encoding="utf-8"))
    assert saved["ssim"] == 0.5
    assert saved["psnr"] == 20.0
    assert saved["baseline"] == {"L4": {"ssim": 0.99, "psnr": 40.0}}
    assert saved["Hardware"] == "L4"
    assert saved["branch"] == ""


def test_baseline_from_similarity_thresholds_keeps_all_profiles():
    table = {
        "B200": helpers.SimilarityThresholds(ssim=0.93, psnr=28.0),
        "default": helpers.SimilarityThresholds(ssim=0.94, psnr=28.0),
    }
    assert helpers.baseline_from_similarity_thresholds(table) == {
        "B200": {"ssim": 0.93, "psnr": 28.0},
        "default": {"ssim": 0.94, "psnr": 28.0},
    }


@pytest.mark.parametrize("ssim", [0.99, 0.98])
def test_assert_similarity_preserves_interface_and_computes_once(monkeypatch, ssim):
    prediction = Image.new("RGBA", (16, 16))
    reference = Image.new("RGBA", (16, 16))
    calls = []

    def compute(**kwargs):
        calls.append(kwargs)
        return ssim, 40.0

    monkeypatch.setattr(helpers, "compute_image_ssim_psnr", compute)
    check = partial(
        helpers.assert_similarity,
        model_name="example",
        vllm_image=prediction,
        diffusers_image=reference,
        ssim_threshold=0.99,
        psnr_threshold=40.0,
        width=16,
        height=16,
        compare_mode="RGBA",
    )
    if ssim < 0.99:
        with pytest.raises(AssertionError, match="SSIM below threshold for example"):
            check()
    else:
        check()
    assert calls == [{"prediction": prediction, "reference": reference, "compare_mode": "RGBA"}]


def test_assert_similarity_preserves_geometry_checks():
    check = partial(
        helpers.assert_similarity,
        model_name="example",
        vllm_image=Image.new("RGBA", (16, 16)),
        diffusers_image=Image.new("RGBA", (32, 32)),
        ssim_threshold=0.99,
        psnr_threshold=40.0,
    )
    with pytest.raises(pytest.skip.Exception, match="baseline output is corrupt"):
        check(width=16, height=16)
    with pytest.raises(AssertionError, match="output sizes mismatch"):
        check()
