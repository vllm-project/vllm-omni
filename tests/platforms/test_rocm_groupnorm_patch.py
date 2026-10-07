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


def test_hunyuan_image3_keeps_pytorch_groupnorm(monkeypatch):
    model = types.SimpleNamespace(vae=nn.Sequential(nn.GroupNorm(4, 8)))
    monkeypatch.setattr(patch_groupnorm, "_original_initialize_model", lambda _: model)
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
    monkeypatch.setattr(patch_groupnorm, "_original_initialize_model", lambda _: model)

    replaced = False

    def replace_groupnorm(_vae):
        nonlocal replaced
        replaced = True
        return True

    monkeypatch.setattr(patch_groupnorm, "_replace_groupnorm_with_aiter", replace_groupnorm)
    aiter_ops = types.ModuleType("vllm._aiter_ops")
    setattr(aiter_ops, "is_aiter_found_and_supported", lambda: True)
    monkeypatch.setitem(sys.modules, "vllm._aiter_ops", aiter_ops)

    od_config = types.SimpleNamespace(model_class_name="OtherDiffusionModel")
    assert patch_groupnorm._patched_initialize_model(od_config) is model
    assert replaced


def test_a_stage_without_a_vae_is_left_alone(monkeypatch):
    """Cosmos3's disaggregated reasoner never decodes, so its ``vae`` is ``None``."""
    model = types.SimpleNamespace(vae=None)
    monkeypatch.setattr(patch_groupnorm, "_original_initialize_model", lambda _: model)
    monkeypatch.setattr(
        patch_groupnorm,
        "_replace_groupnorm_with_aiter",
        lambda _: pytest.fail("there is no VAE to patch"),
    )
    aiter_ops = types.ModuleType("vllm._aiter_ops")
    setattr(aiter_ops, "is_aiter_found_and_supported", lambda: True)
    monkeypatch.setitem(sys.modules, "vllm._aiter_ops", aiter_ops)
    warnings: list[str] = []
    monkeypatch.setattr(patch_groupnorm.logger, "warning", lambda msg, *a, **k: warnings.append(msg))

    od_config = types.SimpleNamespace(model_class_name="Cosmos3ReasonerPipeline")
    assert patch_groupnorm._patched_initialize_model(od_config) is model
    assert warnings == []
