# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Regression tests for the exit path of vLLM's model-inspection subprocess.

vLLM 0.32 spawns ``python -m vllm.model_executor.models.registry`` as a child
process to inspect model architectures; the child imports ``vllm_omni`` through
``load_general_plugins()``. On NPU the child's interpreter aborts while
finalizing (SIGABRT, "corrupted size vs. prev_size", exit code 134), and vLLM
0.32 treats that exit code as a hard failure ("Model architectures ... failed
to be inspected") and refuses to start the engine.

``vllm_omni`` therefore bypasses finalization from an ``atexit`` handler.
That bypass must only fire on the success path -- i.e. after the architecture
check *and* the registry serialization have both completed -- and must leave
stdout/stderr intact. A failing inspection has to keep its non-zero exit code
and its original traceback on stderr, because vLLM's parent calls
``check_returncode()`` before it reads the pickled output: masking the exit
status would swallow the real error and turn it into a confusing
"output file missing" failure.

Both tests really spawn the subprocess, so they need an NPU and skip otherwise.
"""

from __future__ import annotations

import pickle
import subprocess
import sys
from pathlib import Path

import cloudpickle
import pytest
import torch

from tests.helpers.mark import hardware_marks

_STDOUT_MARKER = "VLLM_OMNI_REGISTRY_STDOUT_MARKER"
_FAILURE_MESSAGE = "VLLM_OMNI_REGISTRY_INSPECTION_BOOM"

pytestmark = [
    pytest.mark.core_model,
    *hardware_marks(res={"npu": "A3"}, num_cards=1),
    pytest.mark.skipif(
        not (hasattr(torch, "npu") and torch.npu.is_available()),
        reason="Requires an available Ascend NPU.",
    ),
]


def _spawn_registry_child(fn, output_file: Path) -> subprocess.CompletedProcess:
    """Run the inspection child exactly the way vLLM's ``_run_in_subprocess`` does.

    ``fn`` is a lambda on purpose: cloudpickle always pickles lambdas by value,
    so the child never has to import this test module.
    """
    payload = cloudpickle.dumps((fn, str(output_file)))
    return subprocess.run(
        [sys.executable, "-m", "vllm.model_executor.models.registry"],
        input=payload,
        capture_output=True,
        timeout=900,
    )


def test_registry_inspection_subprocess_exits_zero_on_success(tmp_path):
    """Success path: check + serialization completed, so the child exits 0.

    Without the bypass the child aborts during finalization (exit code 134) and
    vLLM 0.32 refuses to start the engine.
    """
    output_file = tmp_path / "registry_output.tmp"

    completed = _spawn_registry_child(
        lambda: print(_STDOUT_MARKER)
        or sorted(
            __import__(
                "vllm.model_executor.models.registry", fromlist=["ModelRegistry"]
            ).ModelRegistry.get_supported_archs()
        ),
        output_file,
    )

    assert completed.returncode == 0, completed.stderr.decode(errors="replace")
    stdout = completed.stdout.decode(errors="replace")
    assert _STDOUT_MARKER in stdout, f"child stdout was swallowed: {stdout!r}"
    with open(output_file, "rb") as f:
        archs = pickle.load(f)
    assert archs, "registry was not serialized by the child"


def test_registry_inspection_subprocess_keeps_failure_exit_code_and_stderr(tmp_path):
    """Failure path: a failing check must stay visible to the parent."""
    output_file = tmp_path / "registry_output.tmp"

    completed = _spawn_registry_child(
        lambda: (_ for _ in ()).throw(RuntimeError(_FAILURE_MESSAGE)),
        output_file,
    )

    assert completed.returncode != 0
    stderr = completed.stderr.decode(errors="replace")
    assert _FAILURE_MESSAGE in stderr, stderr
    assert "RuntimeError" in stderr, stderr
    assert not output_file.exists()
