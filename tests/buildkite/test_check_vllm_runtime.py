# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""The CUDA runtime gate must verify release wheel build commits."""

import importlib
import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

SCRIPT = Path(__file__).resolve().parents[2] / ".buildkite/cuda/scripts/check_vllm_runtime.py"


@pytest.mark.parametrize(
    "commit", ["g4ba787ec9", "g61cbf5429", "gcbafc4690", "g4135590ff", "g82eb574d9", "gdeadbeef0", None, "g5df6"]
)
@pytest.mark.parametrize("lmcache_abi_error", [False, True])
def test_release_wheel_commit_check(monkeypatch, commit, lmcache_abi_error):
    # All candidates advertise the same release version. Only the wheel built
    # from the pinned source can pass; absent/truncated build metadata fails.
    fake_vllm = ModuleType("vllm")
    vars(fake_vllm)["__version__"] = "0.33.0rc3.dev3+g4ba787ec9"
    fake_version = ModuleType("vllm._version")
    vars(fake_version)["__commit_id__"] = commit
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm._version", fake_version)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            __version__="2.14.0",
            version=SimpleNamespace(cuda="13.0"),
            cuda=SimpleNamespace(is_available=lambda: True, get_device_name=lambda: "test GPU"),
        ),
    )
    for name in ("flashinfer", "vllm._C_stable_libtorch", "lmcache.cuda_ops"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    if lmcache_abi_error:
        original_import = importlib.import_module

        def import_module(name: str) -> ModuleType:
            if name == "lmcache.cuda_ops":
                raise ImportError("undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_jb")
            return original_import(name)

        monkeypatch.setattr(importlib, "import_module", import_module)
    main = runpy.run_path(str(SCRIPT))["main"]
    if commit == "g4ba787ec9":
        if lmcache_abi_error:
            with pytest.raises(ImportError, match="undefined symbol:.*c10_cuda_check_implementation"):
                main()
        else:
            main()
    else:
        with pytest.raises(AssertionError, match="Expected vLLM"):
            main()
