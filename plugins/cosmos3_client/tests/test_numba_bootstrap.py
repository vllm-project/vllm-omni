# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The native control client must compile with the legacy optimization level."""

import os
import subprocess
import sys

import pytest


@pytest.mark.cpu
def test_client_import_preserves_level3_control_jit(tmp_path):
    pytest.importorskip("numba")
    pytest.importorskip("robosuite")
    code = """
import resource
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
from vllm_omni_cosmos3_client import closed_loop_eval
closed_loop_eval._configure_mujoco_env('osmesa')
closed_loop_eval._import_libero()
from robosuite.utils.transform_utils import mat2quat
import numba
import numpy as np
assert numba.config.DISABLE_JIT == 0
assert numba.config.OPT == 3
np.testing.assert_array_equal(mat2quat(np.eye(3)), np.array([0., 0., 0., 1.], dtype=np.float32))
assert mat2quat.signatures
"""
    environment = {
        **os.environ,
        "NUMBA_DISABLE_JIT": "0",
        "NUMBA_OPT": "3",
        "NUMBA_CACHE_DIR": str(tmp_path / "numba_cache"),
        "MUJOCO_GL": "osmesa",
        "PYOPENGL_PLATFORM": "osmesa",
        "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
    }
    result = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", code],
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
