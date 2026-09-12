# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression test for the NVFP4 NaN-clamp install guard in
``vllm_omni.patch`` swallowing only ``ImportError``.

https://github.com/vllm-project/vllm-omni/issues/7232 hit a weekly-CI
crash where importing ``vllm_omni`` raised
``vllm.third_party.pynvml.NVMLError_InvalidArgument``. That error comes
from ``vllm.model_executor.layers.quantization.utils.w8a8_utils``'s
module-level ``CUTLASS_FP8_SUPPORTED = cutlass_fp8_supported()``, which
is reached transitively while ``vllm_omni.patch`` does:

    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptNvFp4LinearMethod as _OriginalModelOptNvFp4LinearMethod,
    )

to install the NVFP4 weight_scale NaN clamp. That import is guarded by
``except ImportError`` so a missing/older modelopt degrades gracefully
with a warning, but ``NVMLError_InvalidArgument`` is a ``RuntimeError``
subclass, not an ``ImportError`` subclass — so the guard didn't catch it
and the whole ``vllm_omni`` import crashed instead of degrading.

This test does not require a real ``vllm`` install (unavailable in this
environment, and CUDA/NVML-only): it extracts the actual try/except/else
block via ``ast`` from the current ``vllm_omni/patch.py`` source, then
execs it in a subprocess with a fake import chain that raises a
non-ImportError exception where the real modelopt import chain would
raise ``NVMLError_InvalidArgument``. This exercises the real, current
source of the guard rather than a hand-copied duplicate that could drift
out of sync.
"""

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

PATCH_PY = Path(__file__).resolve().parents[3] / "vllm_omni" / "patch.py"


def _extract_nvfp4_install_block() -> str:
    source = PATCH_PY.read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Try) and any(
            isinstance(handler, ast.ExceptHandler) and handler.name == "_nan_clamp_import_err"
            for handler in node.handlers
        ):
            segment = ast.get_source_segment(source, node)
            assert segment is not None
            return segment
    raise AssertionError("NVFP4 NaN-clamp try/except block not found in vllm_omni/patch.py")


# Fakes the exact failure mode from the issue: importing
# vllm.model_executor.layers.quantization.modelopt raises a non-ImportError
# exception (standing in for NVMLError_InvalidArgument, a RuntimeError
# subclass) instead of failing to import. Runs in a subprocess so the fake
# `vllm.*` namespace packages and sys.meta_path finder never leak into other
# tests in this process.
_NVFP4_IMPORT_GUARD_PROBE = """
import importlib.abc
import importlib.machinery
import logging
import os
import sys
import types


class _FakeNVMLError(RuntimeError):
    pass  # stand-in for vllm.third_party.pynvml.NVMLError_InvalidArgument


class _RaisingLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return None

    def exec_module(self, module):
        raise _FakeNVMLError("Invalid Argument")


class _RaisingFinder(importlib.abc.MetaPathFinder):
    TARGET = "vllm.model_executor.layers.quantization.modelopt"

    def find_spec(self, fullname, path, target=None):
        if fullname == self.TARGET:
            return importlib.machinery.ModuleSpec(fullname, _RaisingLoader())
        return None


for _pkg in (
    "vllm",
    "vllm.model_executor",
    "vllm.model_executor.layers",
    "vllm.model_executor.layers.quantization",
):
    _mod = types.ModuleType(_pkg)
    _mod.__path__ = []
    sys.modules[_pkg] = _mod

sys.meta_path.insert(0, _RaisingFinder())

_PATCH_LOGGER = logging.getLogger("vllm_omni.patch.test")
_already_patched_upstream = False
_clamp_installed = False

__BLOCK__

print("SURVIVED")
"""


def test_nvfp4_install_guard_survives_non_import_error_from_modelopt_chain():
    """The install guard must not let a non-ImportError exception from the
    optional modelopt import chain crash the whole vllm_omni import."""
    block = _extract_nvfp4_install_block()
    script = _NVFP4_IMPORT_GUARD_PROBE.replace("__BLOCK__", block)
    env = os.environ.copy()
    env.pop("VLLM_OMNI_SKIP_NVFP4_NAN_CLAMP", None)
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=30, env=env)
    assert result.returncode == 0 and "SURVIVED" in result.stdout, (
        "NVFP4 NaN-clamp install guard must catch non-ImportError exceptions raised by "
        "the optional modelopt import chain (e.g. an NVML/driver error surfaced "
        "transitively via cutlass_fp8_supported()) instead of crashing the whole "
        f"vllm_omni import.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
