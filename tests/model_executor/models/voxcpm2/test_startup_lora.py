# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Native VoxCPM2 adapter validation and merge parity, without model weights."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch import nn
from torch.nn import functional as F

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def merge_lora():
    # Keep these L1 checks independent of vLLM custom-op/platform imports.
    path = Path(__file__).resolve().parents[4] / "vllm_omni/model_executor/models/voxcpm2/lora.py"
    spec = importlib.util.spec_from_file_location("voxcpm2_startup_lora", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.merge_voxcpm2_lora


@pytest.fixture
def adapter(tmp_path):
    def attention():
        return nn.ModuleDict({name: nn.Linear(3, 4, bias=True) for name in ("q_proj", "k_proj", "v_proj", "o_proj")})

    base_lm = nn.ModuleDict({"layers": nn.ModuleList([attention()])})
    residual_lm = nn.ModuleDict({"layers": nn.ModuleList([attention()])})
    tts = nn.ModuleDict(
        {
            "feat_decoder": nn.ModuleDict({"estimator": nn.ModuleDict({"decoder": attention()})}),
            "fusion_concat_proj": nn.Linear(3, 4),
            "stop_head": nn.Linear(3, 2),
        }
    )
    roots = {"base_lm": base_lm, "residual_lm": residual_lm, "tts": tts}
    native = nn.ModuleDict({"base_lm": base_lm, "residual_lm": residual_lm, **dict(tts.items())})
    targets = {name: module for name, module in native.named_modules() if isinstance(module, nn.Linear)}
    targets.pop("stop_head")
    generator = torch.Generator().manual_seed(42)
    tensors = {
        f"{name}.lora_{suffix}": torch.randn(shape, generator=generator)
        for name in targets
        for suffix, shape in (("A", (2, 3)), ("B", (4, 2)))
    }
    config = {
        "r": 2,
        "alpha": 3,
        "enable_lm": True,
        "enable_dit": True,
        "enable_proj": True,
        "target_proj_modules": ["fusion_concat_proj"],
    }

    def save():
        (tmp_path / "lora_config.json").write_text(json.dumps({"lora_config": config}))
        save_file(tensors, str(tmp_path / "lora_weights.safetensors"))

    save()
    return tmp_path, roots, targets, tensors, config, save


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_merge_matches_native_linear_output(merge_lora, adapter, dtype):
    path, roots, targets, tensors, _, _ = adapter
    inputs = torch.tensor([[0.5, -0.2, 0.1]], dtype=dtype)
    expected = {}
    identities = {}
    for name, layer in targets.items():
        layer.to(dtype=dtype)
        identities[name] = id(layer.weight)
        # Native LoRALinear: linear(x, W, bias) + linear(linear(x, A), B) * alpha/r.
        expected[name] = F.linear(inputs.float(), layer.weight.float(), layer.bias.float()) + 1.5 * F.linear(
            F.linear(inputs.float(), tensors[f"{name}.lora_A"]), tensors[f"{name}.lora_B"]
        )
    untouched = roots["tts"].stop_head.weight.detach().clone()
    assert merge_lora(str(path), **roots) == 13
    for name, layer in targets.items():
        tolerance = 0.025 if dtype == torch.bfloat16 else 1e-6
        torch.testing.assert_close(layer(inputs).float(), expected[name], atol=tolerance, rtol=tolerance)
        assert id(layer.weight) == identities[name]
        assert layer.weight.dtype == dtype
        assert set(dict(layer.named_parameters())) == {"weight", "bias"}
    torch.testing.assert_close(roots["tts"].stop_head.weight, untouched, rtol=0, atol=0)


@pytest.mark.parametrize("group", ["lm", "dit", "proj"])
def test_independently_enabled_groups(merge_lora, adapter, group):
    path, roots, targets, tensors, config, save = adapter
    for key in ("lm", "dit", "proj"):
        config[f"enable_{key}"] = key == group
    prefixes = {"lm": ("base_lm.", "residual_lm."), "dit": ("feat_decoder.",), "proj": ("fusion_concat_proj.",)}
    for key in list(tensors):
        if not key.startswith(prefixes[group]):
            del tensors[key]
    original = {name: layer.weight.detach().clone() for name, layer in targets.items()}
    save()
    assert merge_lora(str(path), **roots) == {"lm": 8, "dit": 4, "proj": 1}[group]
    for name, layer in targets.items():
        if not (name + ".").startswith(prefixes[group]):
            torch.testing.assert_close(layer.weight, original[name], rtol=0, atol=0)


@pytest.mark.parametrize(
    "corruption", ["missing", "extra", "shape", "rank", "alpha", "nonfinite", "dtype", "target", "flag"]
)
def test_rejects_invalid_checkpoint_before_mutating_weights(merge_lora, adapter, corruption):
    path, roots, targets, tensors, config, save = adapter
    key = "fusion_concat_proj.lora_B"  # Validate even the last component before merging any layer.
    if corruption == "missing":
        del tensors[key]
    elif corruption == "extra":
        tensors["audio_vae.lora_A"] = torch.ones(2, 3)
    elif corruption == "shape":
        tensors[key] = torch.ones(5, 2)
    elif corruption == "rank":
        config["r"] = 0
    elif corruption == "alpha":
        config["alpha"] = float("nan")
    elif corruption == "nonfinite":
        tensors[key][0, 0] = float("nan")
    elif corruption == "dtype":
        tensors[key] = tensors[key].long()
    elif corruption == "target":
        config["target_modules_lm"] = ["does_not_exist"]
    elif corruption == "flag":
        config["enable_dit"] = "false"
    save()
    original = {name: layer.weight.detach().clone() for name, layer in targets.items()}
    with pytest.raises(ValueError, match="VoxCPM2"):
        merge_lora(str(path), **roots)
    for name, layer in targets.items():
        torch.testing.assert_close(layer.weight, original[name], rtol=0, atol=0)


def test_requires_native_metadata_and_safetensors(merge_lora, adapter):
    path, roots, _, _, _, save = adapter
    (path / "lora_config.json").write_text(json.dumps({"r": 2, "alpha": 3}))
    with pytest.raises(ValueError, match="lora_config object"):
        merge_lora(str(path), **roots)
    save()
    (path / "lora_weights.safetensors").unlink()
    with pytest.raises(FileNotFoundError):
        merge_lora(str(path), **roots)


def test_talker_loads_base_before_merge_and_rejects_reload(adapter):
    pytest.importorskip("vllm")
    from vllm_omni.model_executor.models.voxcpm2.runtime_config import _VoxCPM2RuntimeConfig
    from vllm_omni.model_executor.models.voxcpm2.voxcpm2_talker import VoxCPM2TalkerForConditionalGeneration

    path, roots, targets, tensors, _, _ = adapter
    talker = VoxCPM2TalkerForConditionalGeneration.__new__(VoxCPM2TalkerForConditionalGeneration)
    nn.Module.__init__(talker)
    talker.model = roots["base_lm"]
    talker.residual_model = roots["residual_lm"]
    talker._tts = roots["tts"]
    talker._runtime_config = _VoxCPM2RuntimeConfig(startup_lora_path=str(path))
    talker._startup_lora_applied = False
    talker._patch_size, talker._feat_dim, talker._side_dtype = 4, 64, torch.float32
    weights = [(f"base_lm.{name}", torch.ones_like(param)) for name, param in talker.model.named_parameters()]
    loaded = talker.load_weights(iter(weights))
    name = "base_lm.layers.0.q_proj"
    expected = 1 + 1.5 * tensors[f"{name}.lora_B"] @ tensors[f"{name}.lora_A"]
    torch.testing.assert_close(targets[name].weight, expected)
    assert "model.layers.0.q_proj.weight" in loaded
    assert talker._startup_lora_applied
    with pytest.raises(ValueError, match="restart the server"):
        talker.load_weights(iter(weights))
    torch.testing.assert_close(targets[name].weight, expected)
