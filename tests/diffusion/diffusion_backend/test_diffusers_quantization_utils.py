# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Diffusers backend quantization conversion helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.models.diffusers_adapter import quantization_utils

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _FakePipelineQuantizationConfig:
    def __init__(self, *, quant_mapping):
        self.quant_mapping = quant_mapping


class _FakeTorchAoConfig:
    def __init__(self, quant_type, modules_to_not_convert=None):
        self.quant_type = quant_type
        self.modules_to_not_convert = modules_to_not_convert


class _FakeFloat8DynamicActivationFloat8WeightConfig:
    pass


class _FakeInt8DynamicActivationInt8WeightConfig:
    pass


@pytest.fixture(autouse=True)
def patch_quantization_backends(monkeypatch):
    monkeypatch.setattr(
        quantization_utils,
        "_get_diffusers_quantization_classes",
        lambda: (_FakePipelineQuantizationConfig, _FakeTorchAoConfig),
    )

    def fake_get_torchao_quant_type_cls(class_name: str):
        return {
            "Float8DynamicActivationFloat8WeightConfig": _FakeFloat8DynamicActivationFloat8WeightConfig,
            "Int8DynamicActivationInt8WeightConfig": _FakeInt8DynamicActivationInt8WeightConfig,
        }[class_name]

    monkeypatch.setattr(quantization_utils, "_get_torchao_quant_type_cls", fake_get_torchao_quant_type_cls)


def _quant_config(method: str, **kwargs):
    return SimpleNamespace(get_name=lambda: method, **kwargs)


def test_builds_fp8_pipeline_quantization_config():
    pipeline_config = quantization_utils.build_diffusers_quantization_config(
        _quant_config(
            "fp8",
            activation_scheme="dynamic",
            is_checkpoint_fp8_serialized=False,
            weight_block_size=None,
        )
    )

    torchao_config = pipeline_config.quant_mapping["transformer"]
    assert isinstance(torchao_config, _FakeTorchAoConfig)
    assert isinstance(torchao_config.quant_type, _FakeFloat8DynamicActivationFloat8WeightConfig)
    assert torchao_config.modules_to_not_convert is None


def test_builds_int8_pipeline_quantization_config():
    pipeline_config = quantization_utils.build_diffusers_quantization_config(
        _quant_config(
            "int8",
            activation_scheme="dynamic",
            is_checkpoint_int8_serialized=False,
        )
    )

    torchao_config = pipeline_config.quant_mapping["transformer"]
    assert isinstance(torchao_config, _FakeTorchAoConfig)
    assert isinstance(torchao_config.quant_type, _FakeInt8DynamicActivationInt8WeightConfig)
    assert torchao_config.modules_to_not_convert is None


def test_builds_from_real_vllm_fp8_quantization_config():
    fp8_module = pytest.importorskip("vllm.model_executor.layers.quantization.fp8")
    quant_config = fp8_module.Fp8Config(
        is_checkpoint_fp8_serialized=False,
        activation_scheme="dynamic",
    )

    pipeline_config = quantization_utils.build_diffusers_quantization_config(quant_config)

    torchao_config = pipeline_config.quant_mapping["transformer"]
    assert isinstance(torchao_config.quant_type, _FakeFloat8DynamicActivationFloat8WeightConfig)


def test_builds_from_real_omni_int8_quantization_config():
    from vllm_omni.quantization.int8_config import DiffusionInt8Config

    quant_config = DiffusionInt8Config(
        is_checkpoint_int8_serialized=False,
        activation_scheme="dynamic",
    )

    pipeline_config = quantization_utils.build_diffusers_quantization_config(quant_config)

    torchao_config = pipeline_config.quant_mapping["transformer"]
    assert isinstance(torchao_config.quant_type, _FakeInt8DynamicActivationInt8WeightConfig)


def test_builds_real_diffusers_torchao_config_when_available(monkeypatch):
    pytest.importorskip("diffusers.quantizers")
    pytest.importorskip("torchao.quantization")

    def real_get_diffusers_quantization_classes():
        from diffusers.quantizers import PipelineQuantizationConfig
        from diffusers.quantizers.quantization_config import TorchAoConfig

        return PipelineQuantizationConfig, TorchAoConfig

    def real_get_torchao_quant_type_cls(class_name: str):
        import torchao.quantization as torchao_quantization

        return getattr(torchao_quantization, class_name)

    monkeypatch.setattr(
        quantization_utils,
        "_get_diffusers_quantization_classes",
        real_get_diffusers_quantization_classes,
    )
    monkeypatch.setattr(quantization_utils, "_get_torchao_quant_type_cls", real_get_torchao_quant_type_cls)

    pipeline_config = quantization_utils.build_diffusers_quantization_config(
        _quant_config(
            "int8",
            activation_scheme="dynamic",
            is_checkpoint_int8_serialized=False,
        )
    )

    from diffusers.quantizers import PipelineQuantizationConfig
    from diffusers.quantizers.quantization_config import TorchAoConfig
    from torchao.quantization import Int8DynamicActivationInt8WeightConfig

    torchao_config = pipeline_config.quant_mapping["transformer"]
    assert isinstance(pipeline_config, PipelineQuantizationConfig)
    assert isinstance(torchao_config, TorchAoConfig)
    assert isinstance(torchao_config.quant_type, Int8DynamicActivationInt8WeightConfig)


@pytest.mark.parametrize(
    "method",
    [
        "gguf",
        "modelopt",
        "mxfp4",
        "mxfp8",
        "mxfp4_dualscale",
        "inc",
        "component",
    ],
)
def test_unsupported_methods_fail_explicitly(method):
    with pytest.raises(NotImplementedError, match=method):
        quantization_utils.ensure_supported_diffusers_quantization(_quant_config(method))


def test_rejects_static_fp8_mapping():
    with pytest.raises(NotImplementedError, match="activation_scheme='dynamic'"):
        quantization_utils.ensure_supported_diffusers_quantization(
            _quant_config(
                "fp8",
                activation_scheme="static",
            )
        )


def test_rejects_serialized_int8_mapping():
    with pytest.raises(NotImplementedError, match="serialized vLLM int8"):
        quantization_utils.ensure_supported_diffusers_quantization(
            _quant_config(
                "int8",
                is_checkpoint_int8_serialized=True,
            )
        )


def test_rejects_fp8_weight_block_size_mapping():
    with pytest.raises(NotImplementedError, match="weight_block_size"):
        quantization_utils.ensure_supported_diffusers_quantization(
            _quant_config(
                "fp8",
                weight_block_size=[128, 128],
            )
        )


def test_rejects_ignored_layer_name_mapping():
    with pytest.raises(NotImplementedError, match="ignored_layers"):
        quantization_utils.ensure_supported_diffusers_quantization(
            _quant_config(
                "int8",
                ignored_layers=["transformer.proj_out"],
            )
        )


def test_rejects_modules_to_not_convert_name_mapping():
    with pytest.raises(NotImplementedError, match="modules_to_not_convert"):
        quantization_utils.ensure_supported_diffusers_quantization(
            _quant_config(
                "int8",
                modules_to_not_convert=["transformer.proj_out"],
            )
        )


def test_apply_preserves_diffusers_load_kwargs_quantization_config(caplog):
    od_config = SimpleNamespace(quantization_config=_quant_config("fp8"))
    existing = object()
    load_kwargs = {"quantization_config": existing}

    with caplog.at_level("WARNING"):
        quantization_utils.apply_diffusers_quantization_config(od_config, load_kwargs)

    assert load_kwargs["quantization_config"] is existing
    assert "Using the Diffusers-native quantization_config" in caplog.text


def test_apply_skips_when_no_vllm_quantization_config():
    od_config = SimpleNamespace(quantization_config=None)
    load_kwargs = {}

    quantization_utils.apply_diffusers_quantization_config(od_config, load_kwargs)

    assert load_kwargs == {}


def test_apply_injects_converted_quantization_config():
    od_config = SimpleNamespace(quantization_config=_quant_config("int8"))
    load_kwargs = {}

    quantization_utils.apply_diffusers_quantization_config(od_config, load_kwargs)

    torchao_config = load_kwargs["quantization_config"].quant_mapping["transformer"]
    assert isinstance(torchao_config.quant_type, _FakeInt8DynamicActivationInt8WeightConfig)
