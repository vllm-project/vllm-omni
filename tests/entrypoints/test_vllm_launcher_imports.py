# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Omni entrypoints must not import vLLM's deprecated ``vllm.entrypoints.openai.api_server`` shim."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_entrypoints_do_not_import_deprecated_vllm_api_server():
    # A fresh interpreter, because other tests in this process may already have imported the shim.
    script = """
import sys

import vllm_omni.entrypoints.cli.serve
import vllm_omni.entrypoints.openai.api_server

assert "vllm.entrypoints.openai.api_server" not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_REPO_ROOT,
        env=dict(os.environ, PYTHONPATH=str(_REPO_ROOT)),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
