# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import subprocess
import sys


def test_batched_runtime_import_does_not_require_optional_opus_dependency() -> None:
    script = """
import builtins

real_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name == "sphn":
        raise ModuleNotFoundError("blocked optional dependency: sphn")
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
from vllm_omni.experimental.fullduplex.personaplex.serving.batched import BatchedSessionManager
assert BatchedSessionManager is not None
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
