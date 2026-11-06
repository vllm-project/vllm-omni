# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import builtins
import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch.nn as nn

from vllm_omni.diffusion import registry
from vllm_omni.platforms.interface import OmniPlatform

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

    assert patch_groupnorm.apply_aiter_groupnorm(model, "HunyuanImage3ForCausalMM") is None
    assert type(model.vae[0]) is nn.GroupNorm


def test_other_models_keep_aiter_groupnorm_patch(monkeypatch):
    model = types.SimpleNamespace(vae=nn.Sequential(nn.GroupNorm(4, 8)))

    replaced = False

    def replace_groupnorm(_vae):
        nonlocal replaced
        replaced = True
        return True

    monkeypatch.setattr(patch_groupnorm, "_replace_groupnorm_with_aiter", replace_groupnorm)
    assert patch_groupnorm.apply_aiter_groupnorm(model, "OtherDiffusionModel") is None
    assert replaced


def test_rocm_platform_initialization_keeps_model_discovery_lightweight(tmp_path):
    (tmp_path / "modular_model_index.json").write_text(json.dumps({"_class_name": "HyNomasModularPipeline"}))
    code = (
        "import sys, types\n"
        "import torch\n"
        "import vllm_omni\n"
        # NVIDIA/CPU test environments cannot import vLLM's ROCm base, which
        # queries a physical GPU at module scope. Exercise the real Omni class
        # initializer with only that hardware-specific base replaced.
        "if torch.version.hip is None:\n"
        "    rocm = types.ModuleType('vllm.platforms.rocm')\n"
        "    rocm.RocmPlatform = type('RocmPlatform', (), {})\n"
        "    sys.modules['vllm.platforms.rocm'] = rocm\n"
        "from vllm_omni.platforms.rocm.platform import RocmOmniPlatform\n"
        "RocmOmniPlatform()\n"
        "from vllm_omni.diffusion.utils.hf_utils import get_diffusion_model_index\n"
        f"assert get_diffusion_model_index({str(tmp_path)!r})['_class_name'] == 'HyNomasModularPipeline'\n"
        "assert 'vllm_omni.diffusion.registry' not in sys.modules\n"
        "assert 'hymm' not in sys.modules\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("is_rocm", [False, True])
@pytest.mark.parametrize("has_vae", [False, True])
def test_registry_applies_groupnorm_only_after_rocm_model_initialization(monkeypatch, is_rocm, has_vae):
    # Exercise upstream's real platform hook. Only the hardware-querying vLLM
    # base is substituted when this CPU regression runs without ROCm.
    import torch

    if torch.version.hip is None:
        rocm_base = types.ModuleType("vllm.platforms.rocm")
        setattr(rocm_base, "RocmPlatform", type("RocmPlatform", (), {}))
        monkeypatch.setitem(sys.modules, "vllm.platforms.rocm", rocm_base)
    platform_path = Path(__file__).parents[2] / "vllm_omni/platforms/rocm/platform.py"
    spec = importlib.util.spec_from_file_location("groupnorm_test_rocm_platform", platform_path)
    assert spec is not None and spec.loader is not None
    rocm_platform = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rocm_platform)

    model = nn.Module()
    if has_vae:
        model.vae = nn.Sequential(nn.GroupNorm(4, 8))
    config = types.SimpleNamespace(
        model_class_name="OtherDiffusionModel",
        quantization_config=None,
        parallel_config=types.SimpleNamespace(vae_patch_parallel_size=1, vae_parallel_mode="tile"),
        vae_use_slicing=False,
        vae_use_tiling=False,
    )
    events = []

    def construct_model(*, od_config):
        assert od_config is config
        events.append("construct")
        return model

    monkeypatch.setattr(registry.DiffusionModelRegistry, "_try_load_model_cls", lambda _: construct_model)
    monkeypatch.setattr(
        registry,
        "current_omni_platform",
        rocm_platform.RocmOmniPlatform if is_rocm else OmniPlatform,
    )
    monkeypatch.setattr(registry, "_apply_sequence_parallel_if_enabled", lambda *_: events.append("parallel"))
    monkeypatch.setattr(registry, "_apply_wan_vae_fastpath_if_enabled", lambda *_: events.append("fastpath"))
    monkeypatch.setitem(sys.modules, "vllm_omni.platforms.rocm.patch.worker.patch_groupnorm", patch_groupnorm)

    def replace_groupnorm(vae):
        assert vae is model.vae
        events.append("groupnorm")
        return True

    monkeypatch.setattr(patch_groupnorm, "_replace_groupnorm_with_aiter", replace_groupnorm)
    assert registry.initialize_model(config) is model
    assert events == ["construct", "parallel", "fastpath"] + (["groupnorm"] if is_rocm and has_vae else [])


def test_groupnorm_replacement_preserves_weights_and_non_affine_layers(monkeypatch):
    class AiterGroupNorm(nn.GroupNorm):
        pass

    module = types.ModuleType("aiter.ops.groupnorm")
    setattr(module, "GroupNorm", AiterGroupNorm)
    monkeypatch.setitem(sys.modules, "aiter.ops.groupnorm", module)
    vae = nn.Sequential(nn.GroupNorm(4, 8), nn.GroupNorm(4, 8, affine=False))
    weight, bias = vae[0].weight, vae[0].bias
    non_affine = vae[1]
    assert patch_groupnorm._replace_groupnorm_with_aiter(vae)
    assert isinstance(vae[0], AiterGroupNorm)
    assert vae[0].weight is weight and vae[0].bias is bias
    assert vae[1] is non_affine


def test_unavailable_aiter_keeps_pytorch_groupnorm(monkeypatch):
    monkeypatch.setattr(sys.modules["vllm._aiter_ops"], "is_aiter_found_and_supported", lambda: False)
    monkeypatch.setattr(patch_groupnorm, "_replace_groupnorm_with_aiter", lambda _: pytest.fail("AITER unavailable"))
    model = nn.Module()
    model.vae = nn.Sequential(nn.GroupNorm(4, 8))
    assert patch_groupnorm.apply_aiter_groupnorm(model, "Other") is None
    assert type(model.vae[0]) is nn.GroupNorm


def test_aiter_failure_keeps_model_initialization_successful(monkeypatch):
    def unavailable(_):
        raise RuntimeError("optional AITER extension failed")

    monkeypatch.setattr(patch_groupnorm, "_replace_groupnorm_with_aiter", unavailable)
    model = nn.Module()
    model.vae = nn.Sequential(nn.GroupNorm(4, 8))
    assert patch_groupnorm.apply_aiter_groupnorm(model, "Other") is None
    assert type(model.vae[0]) is nn.GroupNorm


def test_model_without_vae_needs_no_groupnorm_patch(monkeypatch):
    model = types.SimpleNamespace()
    monkeypatch.setattr(
        patch_groupnorm,
        "_replace_groupnorm_with_aiter",
        lambda _: pytest.fail("Models without a VAE need no GroupNorm replacement"),
    )
    patch_groupnorm.apply_aiter_groupnorm(model, "OtherDiffusionModel")
