# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for the Drama 3 quantization configs."""

import sys
import types
from collections.abc import Mapping
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import fp8, get_quantization_config

from vllm_omni.model_executor.models.fish_omni import quantization
from vllm_omni.model_executor.models.fish_omni.quantization import (
    DRAMA3_INT4_QUANTIZATION_METHOD,
    QUANTIZATION_METHOD,
    Drama3Int4Config,
    ExpertFp8Config,
)
from vllm_omni.platforms.npu.quant import fish_omni as npu_quantization

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

CHECKPOINT_QUANT_CONFIG = {
    "quant_method": "bsgemm-moe",
    "weight_block_size": [128, 128],
    "activation_scheme": "dynamic",
}


INT4_CHECKPOINT_QUANT_CONFIG = {
    "bits": 4,
    "group_size": 128,
    "sym": True,
    "desc_act": False,
    "quant_method": DRAMA3_INT4_QUANTIZATION_METHOD,
    "experts": {"quant_method": "gptq"},
    "dense": {
        "quant_method": "gptq_marlin",
        "targets": [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "linear_attn.in_proj_qkv",
            "linear_attn.in_proj_z",
            "linear_attn.out_proj",
        ],
    },
}


def _install_fake_ascend(monkeypatch, **modules: Mapping[str, object]) -> None:
    """Stub the vllm_ascend attributes ``get_fish_omni_quant_method`` imports."""
    packages = {name.rsplit(".", 1)[0] for name in modules}
    packages |= {parent for name in packages for parent in _ancestors(name)}
    for name in sorted(packages):
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = []  # mark it importable as a package
            monkeypatch.setitem(sys.modules, name, package)
    for module_name, attrs in modules.items():
        module = types.ModuleType(module_name)
        for attr, value in attrs.items():
            setattr(module, attr, value)
        monkeypatch.setitem(sys.modules, module_name, module)


def _ancestors(dotted: str) -> list[str]:
    parts = dotted.split(".")
    return [".".join(parts[:i]) for i in range(1, len(parts))]


def test_registered_with_vllm():
    assert get_quantization_config(QUANTIZATION_METHOD) is ExpertFp8Config


def test_from_checkpoint_config():
    config = ExpertFp8Config.from_config(CHECKPOINT_QUANT_CONFIG)
    assert config.get_name() == QUANTIZATION_METHOD
    assert config.is_checkpoint_fp8_serialized
    assert config.activation_scheme == "dynamic"
    assert config.weight_block_size == [128, 128]


@pytest.mark.parametrize(
    "override",
    [
        {"activation_scheme": "static"},
        {"weight_block_size": None},
        {"weight_block_size": [128]},
        {"weight_block_size": [0, 128]},
    ],
)
def test_rejects_unsupported_checkpoint_config(override):
    with pytest.raises(ValueError, match=QUANTIZATION_METHOD):
        ExpertFp8Config.from_config({**CHECKPOINT_QUANT_CONFIG, **override})


def test_linear_layers_stay_unquantized(monkeypatch):
    monkeypatch.setattr(quantization, "current_platform", SimpleNamespace(device_type="cuda"))
    config = ExpertFp8Config.from_config(CHECKPOINT_QUANT_CONFIG)
    method = config.get_quant_method(MagicMock(spec=LinearBase), prefix="model.layers.0.self_attn.qkv_proj")
    assert isinstance(method, UnquantizedLinearMethod)


def test_npu_linear_layers_use_the_ascend_unquantized_method(monkeypatch):
    """The generic method skips Ascend's FRACTAL_NZ conversion and GEMM op."""
    monkeypatch.setattr(quantization, "current_platform", SimpleNamespace(device_type="npu"))

    class _AscendUnquantizedLinearMethod(UnquantizedLinearMethod):
        pass

    _install_fake_ascend(
        monkeypatch,
        **{"vllm_ascend.ops.linear": {"AscendUnquantizedLinearMethod": _AscendUnquantizedLinearMethod}},
    )
    config = ExpertFp8Config.from_config(CHECKPOINT_QUANT_CONFIG)
    method = config.get_quant_method(MagicMock(spec=LinearBase), prefix="model.layers.0.self_attn.qkv_proj")
    assert isinstance(method, _AscendUnquantizedLinearMethod)


def test_routed_experts_use_fp8_moe(monkeypatch):
    monkeypatch.setattr(quantization, "current_platform", SimpleNamespace(device_type="cuda"))
    sentinel = object()
    created = []

    def fake_moe_method(quant_config, layer):
        created.append((quant_config, layer))
        return sentinel

    monkeypatch.setattr(fp8, "Fp8MoEMethod", fake_moe_method)
    config = ExpertFp8Config.from_config(CHECKPOINT_QUANT_CONFIG)
    experts = MagicMock(spec=fp8.RoutedExperts)
    assert config.get_quant_method(experts, prefix="model.layers.0.mlp.experts") is sentinel
    assert created == [(config, experts)]


def test_npu_routed_experts_use_ascend_fp8(monkeypatch):
    monkeypatch.setattr(quantization, "current_platform", SimpleNamespace(device_type="npu"))
    monkeypatch.setattr(npu_quantization, "_dequantize_npu_block_fp8_on_cpu", lambda method: None)
    sentinel = object()
    seen = {}

    class _AscendFp8Config:
        def __init__(self, **kwargs):
            seen["kwargs"] = kwargs

        def get_quant_method(self, layer, prefix, tid2eid=None):
            seen["prefix"] = prefix
            seen["tid2eid"] = tid2eid
            return sentinel

    _install_fake_ascend(
        monkeypatch,
        **{
            "vllm_ascend.ops.linear": {"AscendUnquantizedLinearMethod": UnquantizedLinearMethod},
            "vllm_ascend.quantization.configs.fp8_config": {"AscendFp8Config": _AscendFp8Config},
        },
    )

    config = ExpertFp8Config.from_config(CHECKPOINT_QUANT_CONFIG)
    experts = MagicMock(spec=fp8.RoutedExperts)
    tid2eid = object()
    assert config.get_quant_method(experts, prefix="model.layers.0.mlp.experts", tid2eid=tid2eid) is sentinel
    assert seen["prefix"] == "model.layers.0.mlp.experts"
    assert seen["tid2eid"] is tid2eid
    assert seen["kwargs"]["is_checkpoint_fp8_serialized"] is True
    assert seen["kwargs"]["weight_block_size"] == [128, 128]


def _expert_layer() -> torch.nn.Module:
    layer = torch.nn.Module()
    layer.register_parameter("weight", torch.nn.Parameter(torch.ones(1, 2, 4), requires_grad=False))
    layer.register_parameter("scale", torch.nn.Parameter(torch.tensor([[[2.0, 3.0]]]), requires_grad=False))
    return layer


def test_cpu_dequant_override_is_scoped_to_selected_scheme(monkeypatch):
    """One scheme instance delegates to upstream on the host; nothing else changes.

    The device round trip itself needs an Ascend card; this covers the scoping
    and that the upstream expansion runs with host-resident parameters.
    """
    expanded = torch.zeros(1, 2, 4, dtype=torch.bfloat16)
    seen: dict[str, tuple[object, ...]] = {}

    class Scheme:
        def _resolve_experts(self, layer, weight_name, scale_name):
            seen["args"] = (layer, weight_name, scale_name)
            seen["devices"] = (getattr(layer, weight_name).device.type, getattr(layer, scale_name).device.type)
            return expanded

    selected, other = Scheme(), Scheme()
    original = Scheme._resolve_experts

    class AscendConfig:
        def __init__(self, **kwargs):
            pass

        def get_quant_method(self, layer, prefix, tid2eid=None):
            return SimpleNamespace(quant_method=selected)

    _install_fake_ascend(
        monkeypatch,
        **{
            "vllm_ascend.ops.linear": {"AscendUnquantizedLinearMethod": UnquantizedLinearMethod},
            "vllm_ascend.quantization.configs.fp8_config": {"AscendFp8Config": AscendConfig},
            "vllm_ascend.quantization.methods.w8a8.fp8_block": {"AscendFp8BlockFusedMoEMethod": Scheme},
        },
    )
    monkeypatch.setattr(quantization, "current_platform", SimpleNamespace(device_type="npu"))
    config = ExpertFp8Config.from_config(CHECKPOINT_QUANT_CONFIG)
    config.get_quant_method(torch.nn.Module(), "model.layers.0.mlp.experts")

    assert Scheme._resolve_experts is original
    assert other._resolve_experts(_expert_layer(), "weight", "scale") is expanded

    layer = _expert_layer()
    assert torch.equal(selected._resolve_experts(layer, "weight", "scale"), expanded)
    assert seen["args"] == (layer, "weight", "scale")
    assert seen["devices"] == ("cpu", "cpu")
    npu_quantization._dequantize_npu_block_fp8_on_cpu(None)


def test_int4_quantization_is_registered() -> None:
    assert get_quantization_config(DRAMA3_INT4_QUANTIZATION_METHOD) is Drama3Int4Config


@pytest.mark.parametrize(
    "prefix,quantized",
    [
        ("model.layers.0.self_attn.qkv_proj", True),
        ("model.layers.0.self_attn.o_proj", True),
        ("model.layers.0.linear_attn.in_proj_qkvz", True),
        ("model.layers.0.linear_attn.out_proj", True),
        ("model.layers.0.linear_attn.in_proj_ba", False),
        ("model.layers.0.mlp.shared_expert.gate_up_proj", False),
        ("model.layers.0.mlp.gate", False),
    ],
)
def test_only_checkpoint_dense_targets_are_quantized(prefix: str, quantized: bool) -> None:
    config = Drama3Int4Config.from_config(INT4_CHECKPOINT_QUANT_CONFIG)
    sentinel = MagicMock()
    with patch("vllm.model_executor.layers.quantization.auto_gptq.AutoGPTQLinearMethod", return_value=sentinel):
        method = config.get_quant_method(MagicMock(spec=LinearBase), prefix)
    if quantized:
        assert method is sentinel
    else:
        assert isinstance(method, UnquantizedLinearMethod)


@pytest.mark.parametrize(
    "section,field,value,message",
    [
        (None, "bits", 8, "symmetric 4-bit weights without activation ordering"),
        (None, "sym", False, "symmetric 4-bit weights without activation ordering"),
        (None, "desc_act", True, "symmetric 4-bit weights without activation ordering"),
        (None, "group_size", None, "requires group_size"),
        ("dense", "targets", None, r"requires dense\.targets"),
        (None, "quant_method", "gptq", "expected 'drama3-int4'"),
        ("experts", "quant_method", "awq", r"requires experts\.quant_method='gptq'"),
        ("dense", "quant_method", "gptq", r"requires dense\.quant_method='gptq_marlin'"),
    ],
)
def test_int4_other_configs_are_rejected(
    section: str | None, field: str, value: int | bool | str | None, message: str
) -> None:
    config = deepcopy(INT4_CHECKPOINT_QUANT_CONFIG)
    target = config if section is None else config[section]
    if value is None:
        del target[field]
    else:
        target[field] = value

    with pytest.raises(ValueError, match=message):
        Drama3Int4Config.from_config(config)
