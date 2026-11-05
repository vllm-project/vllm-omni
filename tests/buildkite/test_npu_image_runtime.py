# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Keep the vendor stack intact when choosing the NPU device image."""

import json
import os
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("suffix", "digest"),
    [
        ("", "4a666fd818058c26ce445fa890e31ba781d0d43c721631b0d43e2eb550fbef99"),
        ("-a3", "ec7cd7c98616d55500e21a17b63be3a99603d520087f41dc37b60ae6c75d0730"),
        ("-a5", "51dc15c552d8de7141f84815de9d8ff030bc97b413280db742640aa467d632db"),
        ("-310p", "22f665099114310858daf90d072eb46ce97ebbbdce1f808879aee68a427de76f"),
    ],
)
def test_device_image_build_uses_matching_frozen_digest(tmp_path, suffix, digest):
    capture = tmp_path / "arguments.json"
    buildctl = tmp_path / "buildctl"
    buildctl.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['CAPTURE_FILE']).write_text(json.dumps(sys.argv[1:]))\n"
    )
    buildctl.chmod(0o755)
    result = subprocess.run(
        ["bash", str(ROOT / ".buildkite/npu/scripts/build-npu-image.sh"), "Dockerfile.npu.ci", "candidate", suffix],
        env={**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"], "CAPTURE_FILE": str(capture)},
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    arguments = json.loads(capture.read_text())
    assert "build-arg:VLLM_ASCEND_IMAGE=quay.io/atlas-ci/vllm-ascend" in arguments
    assert "build-arg:VLLM_ASCEND_VERSION=v0.33.0" in arguments
    assert f"build-arg:VLLM_ASCEND_DEVICE_SUFFIX={suffix}" in arguments
    assert f"build-arg:VLLM_ASCEND_DIGEST=sha256:{digest}" in arguments


def test_unknown_device_does_not_start_an_image_build(tmp_path):
    buildctl = tmp_path / "buildctl"
    buildctl.write_text("#!/bin/sh\nexit 99\n")
    buildctl.chmod(0o755)
    result = subprocess.run(
        ["bash", str(ROOT / ".buildkite/npu/scripts/build-npu-image.sh"), "Dockerfile.npu.ci", "candidate", "-unknown"],
        env={**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"]},
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 2
    assert "Unsupported Ascend device suffix" in result.stderr
