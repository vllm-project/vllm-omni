# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Focused tests for Diffusers backend quantization conversion helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.models.diffusers_adapter import quantization_utils

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _FakePipelineQuantizationConfig:
    def __init__(self, *, quant_mapping=None, quant_backend=None, quant_kwargs=None, components_to_quantize=None):
        self.quant_mapping = quant_mapping
        self.quant_backend = quant_backend
        self.quant_kwargs = quant_kwargs
        self.components_to_quantize = components_to_quantize


class _FakeTorchAoConfig:
    def __init__(self, quant_type):
        self.quant_type = quant_type

    @classmethod
    def from_dict(cls, config):
        return cls(quant_type=config["quant_type"])


class _FakeFloat8DynamicActivationFloat8WeightConfig:
    pass


class _FakeInt8DynamicActivationInt8WeightConfig:
    pass


@pytest.fixture(autouse=True)
def patch_quantization_backends(monkeypatch):
    monkeypatch.setattr(
        quantization_utils,
        "PipelineQuantizationConfig",
        _FakePipelineQuantizationConfig,
    )
    monkeypatch.setattr(quantization_utils, "TorchAoConfig", _FakeTorchAoConfig)

    def fake_get_torchao_quant_type_cls(class_name: str):
        return {
            "Float8DynamicActivationFloat8WeightConfig": _FakeFloat8DynamicActivationFloat8WeightConfig,
            "Int8DynamicActivationInt8WeightConfig": _FakeInt8DynamicActivationInt8WeightConfig,
        }[class_name]

    monkeypatch.setattr(quantization_utils, "_get_torchao_quant_type_cls", fake_get_torchao_quant_type_cls)
    monkeypatch.setattr(quantization_utils, "_get_diffusers_quantization_config_cls", lambda _: _FakeTorchAoConfig)


def _quant_config(method: str, **kwargs):
    return SimpleNamespace(get_name=lambda: method, **kwargs)


@pytest.mark.parametrize(
    ("method", "expected_type"),
    [
        ("fp8", _FakeFloat8DynamicActivationFloat8WeightConfig),
        ("int8", _FakeInt8DynamicActivationInt8WeightConfig),
    ],
)
def test_apply_injects_converted_quantization_config(method, expected_type):
    od_config = SimpleNamespace(quantization_config=_quant_config(method))
    load_kwargs = {}

    injected = quantization_utils.apply_diffusers_quantization_config(
        od_config,
        load_kwargs,
        {"transformer": ["diffusers", "Transformer2DModel"]},
    )

    assert injected is True
    torchao_config = load_kwargs["quantization_config"].quant_mapping["transformer"]
    assert isinstance(torchao_config.quant_type, expected_type)


def test_apply_injects_converted_quantization_config_for_transformer_2():
    od_config = SimpleNamespace(quantization_config=_quant_config("int8"))
    load_kwargs = {}

    quantization_utils.apply_diffusers_quantization_config(
        od_config,
        load_kwargs,
        {
            "transformer": ["diffusers", "Transformer2DModel"],
            "transformer_2": ["diffusers", "Transformer2DModel"],
        },
    )

    assert sorted(load_kwargs["quantization_config"].quant_mapping) == ["transformer", "transformer_2"]


@pytest.mark.parametrize("method", ["gguf", "modelopt", "mxfp4", "mxfp8", "inc"])
def test_unsupported_methods_fail_explicitly(method):
    with pytest.raises(NotImplementedError, match=method):
        quantization_utils.ensure_supported_diffusers_quantization(_quant_config(method))


@pytest.mark.parametrize(
    "quant_config",
    [
        _quant_config("fp8", activation_scheme="static"),
        _quant_config("int8", is_checkpoint_int8_serialized=True),
        _quant_config("fp8", weight_block_size=[128, 128]),
        _quant_config("int8", ignored_layers=["transformer.proj_out"]),
    ],
)
def test_ambiguous_mappings_fail_explicitly(quant_config):
    with pytest.raises(NotImplementedError):
        quantization_utils.ensure_supported_diffusers_quantization(quant_config)


def test_apply_preserves_diffusers_load_kwargs_quantization_config(mocker):
    od_config = SimpleNamespace(quantization_config=_quant_config("fp8"))
    existing = object()
    load_kwargs = {"quantization_config": existing}
    mock_warning = mocker.patch.object(quantization_utils.logger, "warning")

    injected = quantization_utils.apply_diffusers_quantization_config(od_config, load_kwargs, {})

    assert injected is False
    assert load_kwargs["quantization_config"] is existing
    assert "Using the Diffusers-native quantization_config" in mock_warning.call_args.args[0]


def test_convert_diffusers_quantization_config_from_dict():
    load_kwargs = {
        "quantization_config": {
            "quant_mapping": {
                "transformer": {
                    "quant_method": "torchao",
                    "quant_type": "fake-int8-config",
                }
            }
        }
    }

    quantization_utils.convert_diffusers_quantization_config(load_kwargs)

    quant_config = load_kwargs["quantization_config"]
    assert isinstance(quant_config, _FakePipelineQuantizationConfig)
    assert isinstance(quant_config.quant_mapping["transformer"], _FakeTorchAoConfig)
    assert quant_config.quant_mapping["transformer"].quant_type == "fake-int8-config"
