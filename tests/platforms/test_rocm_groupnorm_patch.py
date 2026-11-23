# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import builtins
import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch.nn as nn

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_PATCH_PATH = Path(__file__).parents[2] / "vllm_omni" / "platforms" / "rocm" / "patch" / "worker" / "patch_groupnorm.py"
_SPEC = importlib.util.spec_from_file_location("test_patch_groupnorm", _PATCH_PATH)
assert _SPEC is not None and _SPEC.loader is not None
patch_groupnorm = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = patch_groupnorm
_SPEC.loader.exec_module(patch_groupnorm)


def test_import_does_not_load_diffusion_registry():
    code = (
        "import importlib.util, sys; "
        f"spec = importlib.util.spec_from_file_location('groupnorm_patch', {str(_PATCH_PATH)!r}); "
        "assert spec is not None and spec.loader is not None; "
        "module = importlib.util.module_from_spec(spec); "
        "spec.loader.exec_module(module); "
        "assert 'vllm_omni.diffusion.registry' not in sys.modules"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.fixture(autouse=True)
def aiter_available(monkeypatch):
    # Exercise replacement on CPU so a missing Hunyuan guard cannot pass silently.
    aiter_ops = types.ModuleType("vllm._aiter_ops")
    setattr(aiter_ops, "is_aiter_found_and_supported", lambda: True)
    monkeypatch.setitem(sys.modules, "vllm._aiter_ops", aiter_ops)


def test_patch_import_does_not_load_runtime_registry(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "vllm_omni.diffusion.registry":
            pytest.fail("Importing the ROCm patch must not load the runtime registry")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert _SPEC is not None and _SPEC.loader is not None
    _SPEC.loader.exec_module(importlib.util.module_from_spec(_SPEC))


def test_hunyuan_image3_keeps_pytorch_groupnorm(monkeypatch):
    model = types.SimpleNamespace(vae=nn.Sequential(nn.GroupNorm(4, 8)))
    monkeypatch.setattr(
        patch_groupnorm,
        "_replace_groupnorm_with_aiter",
        lambda _: pytest.fail("Hunyuan Image 3.0 must keep PyTorch GroupNorm"),
    )

    patch_groupnorm.apply_aiter_groupnorm(model, "HunyuanImage3ForCausalMM")
    assert type(model.vae[0]) is nn.GroupNorm


def test_other_models_keep_aiter_groupnorm_patch(monkeypatch):
    model = types.SimpleNamespace(vae=nn.Sequential(nn.GroupNorm(4, 8)))

    replaced = False

    def replace_groupnorm(_vae):
        nonlocal replaced
        replaced = True
        return True

    monkeypatch.setattr(patch_groupnorm, "_replace_groupnorm_with_aiter", replace_groupnorm)
    patch_groupnorm.apply_aiter_groupnorm(model, "OtherDiffusionModel")
    assert replaced


def test_model_without_vae_needs_no_groupnorm_patch(monkeypatch):
    model = types.SimpleNamespace()
    monkeypatch.setattr(
        patch_groupnorm,
        "_replace_groupnorm_with_aiter",
        lambda _: pytest.fail("Models without a VAE need no GroupNorm replacement"),
    )
    patch_groupnorm.apply_aiter_groupnorm(model, "OtherDiffusionModel")
