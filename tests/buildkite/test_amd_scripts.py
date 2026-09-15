# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

REPO_ROOT = Path(__file__).resolve().parents[2]
AMD_TEST_SCRIPT = REPO_ROOT / ".buildkite/amd/scripts/run-amd-test.sh"


def test_amd_test_script_fails_when_no_tests_are_collected(tmp_path: Path) -> None:
    """Exit code 5 must remain a failure even with the legacy env variable."""
    (tmp_path / "rocminfo").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    (tmp_path / "python3").write_text("#!/bin/sh\ncat >/dev/null\nexit 0\n", encoding="utf-8")
    for command in ("rocminfo", "python3"):
        (tmp_path / command).chmod(0o755)

    env = os.environ.copy()
    env.update(
        PATH=f"{tmp_path}:{env['PATH']}",
        HF_HOME=str(tmp_path / "huggingface"),
        VLLM_CI_DOCKER_DISABLED="1",
        VLLM_CI_ALLOW_NO_TESTS="1",
    )
    result = subprocess.run(
        ["bash", str(AMD_TEST_SCRIPT), "exit 5"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 5
    assert "treating exit code 5 as success" not in result.stdout
