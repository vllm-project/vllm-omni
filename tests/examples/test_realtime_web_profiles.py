# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_realtime_profiles_and_shell_lifecycle():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required for browser protocol tests")
    assert node is not None
    subprocess.run(
        [node, "--test", str(Path(__file__).with_suffix(".cjs"))],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
