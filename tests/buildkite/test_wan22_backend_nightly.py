# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Exercise the real comparison control flow without importing GPU runtimes."""

import ast
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    ("profile", "latency", "ssim", "psnr", "failure"),
    [
        ("ROCm", 100.0, 0.95, 30.0, None),
        ("ROCm", 1.0, 0.949, 30.0, "SSIM"),
        ("ROCm", 1.0, 0.95, 29.9, "PSNR"),
        ("ROCm", 1.0, 1.0, float("inf"), None),
        ("H100", 1.3, 0.95, 30.0, None),
        ("H100", 1.301, 0.95, 30.0, "latency"),
        ("B200", 1.359, 0.95, 30.0, None),
        ("B200", 1.361, 0.95, 30.0, "latency"),
        ("H100", 1.0, 0.949, 30.0, "SSIM"),
        ("B200", 1.0, 0.95, 29.9, "PSNR"),
    ],
)
def test_video_comparison_preserves_quality_and_cuda_latency_gates(tmp_path, profile, latency, ssim, psnr, failure):
    source = Path("tests/e2e/accuracy/test_diffusers_backend_similarity.py")
    tree = ast.parse(source.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "test_diffusers_backend_i2v_matches_diffusers"
    )
    function.decorator_list = []

    def run_candidate(*, model, output_path, conditioning_image):
        output_path.write_bytes(b"candidate video")
        return latency

    def run_reference(*, model, output_path, conditioning_image):
        output_path.write_bytes(b"reference video")
        return 1.0

    def resolve_threshold(thresholds, *, label):
        assert profile != "ROCm", "ROCm must not inherit a CUDA allowance"
        assert thresholds == {"H100": 0.3, "B200": 0.36}
        return profile, thresholds[profile]

    accuracy_results = []

    def write_accuracy_result(**kwargs):
        accuracy_results.append(kwargs)

    def assert_video_similarity_metrics(*, ssim_threshold, psnr_threshold, ssim_score, psnr_score, **_):
        assert ssim_score >= ssim_threshold, "SSIM below threshold"
        assert psnr_score >= psnr_threshold, "PSNR below threshold"

    namespace = {
        "Path": Path,
        "Image": SimpleNamespace(Image=object, Resampling=SimpleNamespace(LANCZOS=1)),
        "torch": SimpleNamespace(version=SimpleNamespace(hip="6.4" if profile == "ROCm" else None)),
        "model_output_dir": lambda root, model: root,
        "_run_vllm_omni_wan22_i2v": run_candidate,
        "_run_diffusers_wan22_i2v": run_reference,
        "resolve_device_threshold": resolve_threshold,
        "_run_ffmpeg_similarity": lambda metric, *_: str(ssim if metric == "ssim" else psnr),
        "_parse_ssim_score": float,
        "_parse_psnr_score": float,
        "assert_video_similarity_metrics": assert_video_similarity_metrics,
        "write_accuracy_result": write_accuracy_result,
        "write_json_artifact": lambda path, value: path.write_text(json.dumps(value, allow_nan=False)),
    }
    # Use the source constants, so reducing a quality threshold fails this test.
    constants = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {"VIDEO_WIDTH", "VIDEO_HEIGHT", "VIDEO_SSIM_THRESHOLD", "VIDEO_PSNR_THRESHOLD"}
    ]
    exec(compile(ast.Module(body=[*constants, function], type_ignores=[]), str(source), "exec"), namespace)
    run = namespace[function.name]
    assert callable(run)
    image = SimpleNamespace(resize=lambda *_: object())
    if failure:
        with pytest.raises(AssertionError, match=failure):
            run("Wan-AI/Wan2.2-I2V-A14B-Diffusers", tmp_path, image)
    else:
        run("Wan-AI/Wan2.2-I2V-A14B-Diffusers", tmp_path, image)

    assert (tmp_path / "vllm_omni.mp4").read_bytes() == b"candidate video"
    assert (tmp_path / "diffusers.mp4").read_bytes() == b"reference video"
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["ssim_threshold"] == 0.95
    assert metrics["psnr_threshold_db"] == 30.0
    assert metrics["latency_gate"] == ("unqualified" if profile == "ROCm" else "enforced")
    assert (metrics["latency_threshold_ms"] is None) == (profile == "ROCm")
    assert len(accuracy_results) == 1
    accuracy = accuracy_results[0]
    assert accuracy["metrics"] == {"ssim": ssim, "psnr": psnr}
    assert accuracy["extra"]["latency_gate"] == metrics["latency_gate"]
    assert accuracy["extra"]["diffusers_latency_ms"] == 1000.0
    assert (tmp_path / "ssim.txt").is_file()
    assert (tmp_path / "psnr.txt").is_file()


@pytest.mark.parametrize(
    ("test_status", "missing", "expected"), [(0, "", 0), (1, "", 1), (124, "", 124), (0, "diffusers.mp4", 5)]
)
def test_nightly_preserves_comparison_artifacts_on_failure(tmp_path, test_status, missing, expected):
    pipeline = yaml.safe_load(Path(".buildkite/amd/test-amd-nightly.yml").read_text())
    step = next(
        step
        for group in pipeline["steps"]
        for step in group.get("steps", [])
        if step.get("label") == "Wan2.2 Diffusers Backend Similarity"
    )
    runner = tmp_path / ".buildkite/amd/scripts/run-rocm-full-model-nightly.sh"
    runner.parent.mkdir(parents=True)
    runner.write_text(
        "#!/bin/bash\nset -euo pipefail\n"
        "source_dir=tests/e2e/accuracy/artifacts/Wan2_2-I2V-A14B-Diffusers-diffusers-backend\n"
        'mkdir -p "$source_dir"\n'
        "for artifact in vllm_omni.mp4 diffusers.mp4 metrics.json ssim.txt psnr.txt; do\n"
        '  if [[ "$artifact" != "$MISSING" ]]; then echo evidence > "$source_dir/$artifact"; fi\n'
        'done\nexit "$TEST_STATUS"\n'
    )
    checkout = tmp_path / "checkout"
    result = subprocess.run(
        ["bash", "-c", step["commands"][0].replace("$$", "$")],
        cwd=tmp_path,
        env={
            **os.environ,
            "BUILDKITE_BUILD_CHECKOUT_PATH": str(checkout),
            "TEST_STATUS": str(test_status),
            "MISSING": missing,
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == expected, result.stdout + result.stderr
    artifacts = checkout / "artifacts/rocm-wan22-backend-similarity/accuracy"
    assert (artifacts / "vllm_omni.mp4").is_file()
    assert (artifacts / "diffusers.mp4").is_file() == (missing != "diffusers.mp4")
    assert (artifacts / "metrics.json").is_file()
