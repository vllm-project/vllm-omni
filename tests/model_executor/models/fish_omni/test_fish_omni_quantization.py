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

    class _WeightFormat:
        FP8 = "fp8"

    class _Profile:
        def supports_weight_format(self, fmt):
            assert fmt == _WeightFormat.FP8
            return True

    class _AscendFp8Config:
        def __init__(self, **kwargs):
            seen["kwargs"] = kwargs

        def get_quant_method(self, layer, prefix):
            seen["layer"] = layer
            seen["prefix"] = prefix
            return sentinel

    _install_fake_ascend(
        monkeypatch,
        **{
            "vllm_ascend.ops.linear": {"AscendUnquantizedLinearMethod": UnquantizedLinearMethod},
            "vllm_ascend.device.hardware_profile": {
                "WeightFormat": _WeightFormat,
                "get_current_hardware_profile": lambda: _Profile(),
            },
            "vllm_ascend.quantization.configs.fp8_config": {"AscendFp8Config": _AscendFp8Config},
        },
    )

    config = ExpertFp8Config.from_config(CHECKPOINT_QUANT_CONFIG)
    experts = MagicMock(spec=fp8.RoutedExperts)
    assert config.get_quant_method(experts, prefix="model.layers.0.mlp.experts") is sentinel
    assert seen["layer"] is experts
    assert seen["prefix"] == "model.layers.0.mlp.experts"
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

        def get_quant_method(self, layer, prefix):
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
    monkeypatch.setattr(npu_quantization, "_supports_native_fp8_matmul", lambda: True)
    config = ExpertFp8Config.from_config(CHECKPOINT_QUANT_CONFIG)
    config.get_quant_method(torch.nn.Module(), "model.layers.0.mlp.experts")

    assert Scheme._resolve_experts is original
    assert other._resolve_experts(_expert_layer(), "weight", "scale") is expanded

    layer = _expert_layer()
    assert torch.equal(selected._resolve_experts(layer, "weight", "scale"), expanded)
    assert seen["args"] == (layer, "weight", "scale")
    assert seen["devices"] == ("cpu", "cpu")
    npu_quantization._dequantize_npu_block_fp8_on_cpu(None)


def test_npu_routed_experts_dequantize_on_cpu_when_fp8_matmul_unsupported(monkeypatch):
    """Devices without native FP8 matmul still dequantize Drama 3 experts on CPU."""
    monkeypatch.setattr(quantization, "current_platform", SimpleNamespace(device_type="npu"))
    bf16_processed: list[torch.nn.Module] = []
    resolved_calls: list[tuple[str, str, tuple[str, str]]] = []
    expanded = {
        "w13_weight": torch.full((1, 2, 4), 2.0, dtype=torch.bfloat16),
        "w2_weight": torch.full((1, 2, 4), 3.0, dtype=torch.bfloat16),
    }

    class _WeightFormat:
        FP8 = "fp8"

    class _Profile:
        def supports_weight_format(self, fmt):
            assert fmt == _WeightFormat.FP8
            return False

    class _Bf16Method:
        def process_weights_after_loading(self, layer):
            bf16_processed.append(layer)

    class _UpstreamFp8BlockMoE:
        def __init__(self, weight_block_size, moe_config):
            raise NotImplementedError("Native block-wise FP8 checkpoints run on Ascend 950 only")

        @property
        def bf16_method(self):
            if self._bf16_method is None:
                self._bf16_method = _Bf16Method()
            return self._bf16_method

        def _resolve_experts(self, layer, weight_name, scale_name):
            resolved_calls.append(
                (
                    weight_name,
                    scale_name,
                    (getattr(layer, weight_name).device.type, getattr(layer, scale_name).device.type),
                )
            )
            return expanded[weight_name]

    class _AscendFusedMoEMethod:
        def __init__(self, scheme, moe_config):
            self.quant_method = scheme
            self.moe_config = moe_config

        def process_weights_after_loading(self, layer):
            self.quant_method.process_weights_after_loading(layer)

    class _AscendFp8Config:
        def __init__(self, **kwargs):
            raise AssertionError("AscendFp8Config should not be used when native FP8 matmul is unsupported")

    moe_config = SimpleNamespace(tag="drama3-moe")
    layer = torch.nn.Module()
    layer.moe_config = moe_config
    for name in ("w13_weight", "w2_weight"):
        layer.register_parameter(name, torch.nn.Parameter(torch.ones(1, 2, 4), requires_grad=False))
        layer.register_parameter(f"{name}_scale_inv", torch.nn.Parameter(torch.ones(1, 1, 2), requires_grad=False))

    _install_fake_ascend(
        monkeypatch,
        **{
            "vllm_ascend.ops.linear": {"AscendUnquantizedLinearMethod": UnquantizedLinearMethod},
            "vllm_ascend.device.hardware_profile": {
                "WeightFormat": _WeightFormat,
                "get_current_hardware_profile": lambda: _Profile(),
            },
            "vllm_ascend.quantization.utils": {"is_fused_moe_layer": lambda mod: mod is layer},
            "vllm_ascend.quantization.configs.fp8_config": {"AscendFp8Config": _AscendFp8Config},
            "vllm_ascend.quantization.method_adapters": {"AscendFusedMoEMethod": _AscendFusedMoEMethod},
            "vllm_ascend.quantization.methods.w8a8.fp8_block": {"AscendFp8BlockFusedMoEMethod": _UpstreamFp8BlockMoE},
        },
    )

    config = ExpertFp8Config.from_config(CHECKPOINT_QUANT_CONFIG)
    assert config.get_quant_method(torch.nn.Module(), "model.embed_tokens") is None

    fake_vllm_cfg = SimpleNamespace(model_config=SimpleNamespace(dtype=torch.bfloat16))
    with patch("vllm.config.get_current_vllm_config", return_value=fake_vllm_cfg):
        method = config.get_quant_method(layer, "model.layers.0.mlp.experts")

    assert isinstance(method, _AscendFusedMoEMethod)
    assert method.moe_config is moe_config
    scheme = method.quant_method
    assert isinstance(scheme, _UpstreamFp8BlockMoE)
    assert scheme.use_mxfp8 is False
    assert scheme.mxfp8_method is None
    assert (scheme.block_n, scheme.block_k, scheme.group_size) == (128, 128, 128)
    assert scheme.model_dtype is torch.bfloat16

    method.process_weights_after_loading(layer)
    assert resolved_calls == [
        ("w13_weight", "w13_weight_scale_inv", ("cpu", "cpu")),
        ("w2_weight", "w2_weight_scale_inv", ("cpu", "cpu")),
    ]
    assert torch.equal(layer.w13_weight, expanded["w13_weight"])
    assert torch.equal(layer.w2_weight, expanded["w2_weight"])
    assert not hasattr(layer, "w13_weight_scale_inv")
    assert not hasattr(layer, "w2_weight_scale_inv")
    assert bf16_processed == [layer]


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
