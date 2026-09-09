# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Validate the project's runner-mode guard without installing a CUDA runtime."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("requested,expected_code", [(None, 0), ("0", 0), ("1", 2)])
def test_wrapper_uses_consistent_v1_runner_or_rejects_v2(tmp_path, requested, expected_code):
    source = Path(__file__).resolve().parents[2] / "tools" / "run_fullduplex_028.sh"
    wrapper = tmp_path / "tools" / source.name
    wrapper.parent.mkdir()
    shutil.copyfile(source, wrapper)
    python = tmp_path / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    # Stand in for Python only: the shell guard runs before the runtime's
    # dependency checks, which have their own real-environment validation.
    python.write_text('#!/bin/sh\nprintf "RUNNER_V2=%s\\n" "${VLLM_USE_V2_MODEL_RUNNER-unset}"\n')
    python.chmod(0o755)
    env = dict(os.environ)
    env.pop("VLLM_USE_V2_MODEL_RUNNER", None)
    if requested is not None:
        env["VLLM_USE_V2_MODEL_RUNNER"] = requested
    result = subprocess.run(["bash", str(wrapper), "-c", "pass"], env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == expected_code, result.stderr
    if expected_code == 0:
        assert result.stdout.splitlines() == ["RUNNER_V2=0", "RUNNER_V2=0"]
    else:
        assert "Model Runner V1" in result.stderr
        assert "RUNNER_V2=" not in result.stdout
