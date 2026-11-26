# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Exercise the real comparison control flow without importing GPU runtimes."""

import os
import subprocess
from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


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
