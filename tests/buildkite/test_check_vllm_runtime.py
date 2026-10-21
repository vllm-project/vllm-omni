# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""The CUDA runtime gate must verify release wheel build commits."""

import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

SCRIPT = Path(__file__).resolve().parents[2] / ".buildkite/cuda/scripts/check_vllm_runtime.py"


@pytest.mark.parametrize("commit", ["g5df63a720", "g82eb574d9", "gdeadbeef0", None, "g5df6"])
def test_release_wheel_commit_check(monkeypatch, commit):
    # All candidates advertise the same release version. Only the wheel built
    # from the pinned source can pass; absent/truncated build metadata fails.
    fake_vllm = ModuleType("vllm")
    vars(fake_vllm)["__version__"] = "0.32.0"
    fake_version = ModuleType("vllm._version")
    vars(fake_version)["__commit_id__"] = commit
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm._version", fake_version)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            __version__="2.13.0",
            version=SimpleNamespace(cuda="13.0"),
            cuda=SimpleNamespace(is_available=lambda: True, get_device_name=lambda: "test GPU"),
        ),
    )
    for name in ("flashinfer", "vllm._C_stable_libtorch"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    main = runpy.run_path(str(SCRIPT))["main"]
    if commit == "g5df63a720":
        main()
    else:
        with pytest.raises(AssertionError, match="Expected vLLM"):
            main()
