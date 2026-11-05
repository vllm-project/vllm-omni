# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import importlib.util
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
patch_groupnorm._registry_mod.initialize_model = patch_groupnorm._original_initialize_model


@pytest.fixture(autouse=True)
def aiter_available(monkeypatch):
    # Exercise replacement on CPU so a missing Hunyuan guard cannot pass silently.
    aiter_ops = types.ModuleType("vllm._aiter_ops")
    setattr(aiter_ops, "is_aiter_found_and_supported", lambda: True)
    monkeypatch.setitem(sys.modules, "vllm._aiter_ops", aiter_ops)


def test_hunyuan_image3_keeps_pytorch_groupnorm(monkeypatch):
    model = types.SimpleNamespace(vae=nn.Sequential(nn.GroupNorm(4, 8)))
    monkeypatch.setattr(patch_groupnorm, "_original_initialize_model", lambda _, **kwargs: model)
    monkeypatch.setattr(
        patch_groupnorm,
        "_replace_groupnorm_with_aiter",
        lambda _: pytest.fail("Hunyuan Image 3.0 must keep PyTorch GroupNorm"),
    )

    od_config = types.SimpleNamespace(model_class_name="HunyuanImage3ForCausalMM")
    assert patch_groupnorm._patched_initialize_model(od_config) is model
    assert type(model.vae[0]) is nn.GroupNorm


def test_other_models_keep_aiter_groupnorm_patch(monkeypatch):
    model = types.SimpleNamespace(vae=nn.Sequential(nn.GroupNorm(4, 8)))
    monkeypatch.setattr(patch_groupnorm, "_original_initialize_model", lambda _, **kwargs: model)

    replaced = False

    def replace_groupnorm(_vae):
        nonlocal replaced
        replaced = True
        return True

    monkeypatch.setattr(patch_groupnorm, "_replace_groupnorm_with_aiter", replace_groupnorm)
    od_config = types.SimpleNamespace(model_class_name="OtherDiffusionModel")
    assert patch_groupnorm._patched_initialize_model(od_config) is model
    assert replaced


@pytest.mark.parametrize("model_init_kwargs", [None, {"backbone_meta": True}])
def test_initialize_model_forwards_kwargs(monkeypatch, model_init_kwargs):
    model = types.SimpleNamespace()
    od_config = types.SimpleNamespace(model_class_name="OtherDiffusionModel")

    def initialize_model(config, **kwargs):
        assert config is od_config
        assert kwargs == {"model_init_kwargs": model_init_kwargs}
        return model

    monkeypatch.setattr(patch_groupnorm, "_original_initialize_model", initialize_model)
    assert patch_groupnorm._patched_initialize_model(od_config, model_init_kwargs=model_init_kwargs) is model
