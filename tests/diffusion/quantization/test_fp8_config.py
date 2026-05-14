# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the unified quantization framework."""

import pytest
from torch import nn

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_build_quant_config_fp8():
    from vllm_omni.quantization import build_quant_config

    config = build_quant_config("fp8")
    assert config is not None
    assert config.get_name() == "fp8"
    assert config.activation_scheme == "dynamic"


def test_build_quant_config_none():
    from vllm_omni.quantization import build_quant_config

    assert build_quant_config(None) is None


def test_build_quant_config_none_string():
    from vllm_omni.quantization import build_quant_config

    assert build_quant_config("none") is None


def test_build_quant_config_invalid():
    from vllm_omni.quantization import build_quant_config

    with pytest.raises(ValueError, match="Unknown quantization method"):
        build_quant_config("invalid_method")


def test_build_quant_config_dict():
    from vllm_omni.quantization import build_quant_config

    config = build_quant_config({"method": "fp8", "activation_scheme": "static"})
    assert config is not None
    assert config.get_name() == "fp8"
    assert config.activation_scheme == "static"


def test_build_quant_config_dict_not_mutated():
    from vllm_omni.quantization import build_quant_config

    original = {"method": "fp8", "activation_scheme": "static"}
    copy = original.copy()
    build_quant_config(original)
    assert original == copy


def test_build_quant_config_modelopt_fp8_config_json():
    from vllm.model_executor.layers.quantization.modelopt import ModelOptFp8Config

    from vllm_omni.quantization import build_quant_config

    config = build_quant_config(
        {
            "quant_method": "modelopt",
            "quant_algo": "FP8",
            "ignore": ["proj_out"],
            "producer": {"name": "modelopt"},
        }
    )

    assert isinstance(config, ModelOptFp8Config)
    assert config.get_name() == "modelopt"
    assert config.is_checkpoint_fp8_serialized


def test_build_quant_config_per_component():
    from vllm_omni.quantization import ComponentQuantizationConfig, build_quant_config

    config = build_quant_config(
        {
            "transformer": {"method": "fp8"},
            "vae": None,
        }
    )
    assert isinstance(config, ComponentQuantizationConfig)
    assert config.component_configs["transformer"].get_name() == "fp8"
    assert config.component_configs["vae"] is None


def test_build_quant_config_per_component_string():
    from vllm_omni.quantization import ComponentQuantizationConfig, build_quant_config

    config = build_quant_config({"transformer": "fp8", "vae": None})
    assert isinstance(config, ComponentQuantizationConfig)
    assert config.component_configs["transformer"].get_name() == "fp8"


def test_build_quant_config_per_component_inner_dict_not_mutated():
    """Inner component dicts should not be mutated by build_quant_config."""
    from vllm_omni.quantization import build_quant_config

    inner = {"method": "fp8", "activation_scheme": "static"}
    original = inner.copy()
    build_quant_config({"transformer": inner, "vae": None})
    assert inner == original


def test_flat_dict_not_misdetected_as_per_component():
    """A flat config like {"activation_scheme": "static"} must NOT be treated
    as a per-component dict — it should raise ValueError for missing 'method'."""
    from vllm_omni.quantization import build_quant_config

    with pytest.raises(ValueError, match="must have a 'method' or 'quant_method' key"):
        build_quant_config({"activation_scheme": "static"})


def test_build_quant_config_passthrough():
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    from vllm_omni.quantization import build_quant_config

    fp8 = Fp8Config(is_checkpoint_fp8_serialized=False, activation_scheme="dynamic")
    assert build_quant_config(fp8) is fp8


def test_component_config_routing():
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    from vllm_omni.quantization import ComponentQuantizationConfig

    fp8 = Fp8Config(is_checkpoint_fp8_serialized=False, activation_scheme="dynamic")
    config = ComponentQuantizationConfig(
        component_configs={"transformer": fp8, "vae": None},
    )

    assert config.get_name() == "component"
    assert config.resolve("transformer.blocks.0.attn") is fp8
    assert config.resolve("vae.encoder.conv_in") is None
    assert config.resolve("unknown.layer") is None


def test_component_config_with_default():
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    from vllm_omni.quantization import ComponentQuantizationConfig

    fp8 = Fp8Config(is_checkpoint_fp8_serialized=False, activation_scheme="dynamic")
    config = ComponentQuantizationConfig(
        component_configs={"vae": None},
        default_config=fp8,
    )

    assert config.resolve("transformer.blocks.0") is fp8
    assert config.resolve("vae.encoder") is None


def test_gguf_config():
    from vllm_omni.quantization import build_quant_config
    from vllm_omni.quantization.gguf_config import DiffusionGGUFConfig

    config = build_quant_config(
        {
            "method": "gguf",
            "gguf_model": "path/to/model.gguf",
        }
    )
    assert isinstance(config, DiffusionGGUFConfig)
    assert config.gguf_model == "path/to/model.gguf"
    assert config.get_name() == "gguf"


def test_integration_string():
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    config = OmniDiffusionConfig(model="test", quantization_config="fp8")
    assert config.quantization_config is not None
    assert config.quantization_config.get_name() == "fp8"


def test_integration_dict():
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    config = OmniDiffusionConfig(
        model="test",
        quantization_config={"method": "fp8", "activation_scheme": "static"},
    )
    assert config.quantization_config is not None
    assert config.quantization_config.get_name() == "fp8"
    assert config.quantization_config.activation_scheme == "static"


def test_integration_no_quant():
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    config = OmniDiffusionConfig(model="test")
    assert config.quantization_config is None


def test_integration_per_component():
    """OmniDiffusionConfig with per-component quantization dict."""
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.quantization import ComponentQuantizationConfig

    config = OmniDiffusionConfig(
        model="test",
        quantization_config={
            "transformer": {"method": "fp8"},
            "vae": None,
        },
    )
    assert isinstance(config.quantization_config, ComponentQuantizationConfig)
    assert config.quantization_config.component_configs["transformer"].get_name() == "fp8"
    assert config.quantization_config.component_configs["vae"] is None


def test_transformer_config_auto_detects_modelopt_fp8():
    from vllm.model_executor.layers.quantization.modelopt import ModelOptFp8Config

    from vllm_omni.diffusion.data import TransformerConfig

    config = TransformerConfig.from_dict(
        {
            "_class_name": "FluxTransformer2DModel",
            "quantization_config": {
                "quant_method": "modelopt",
                "quant_algo": "FP8",
                "ignore": ["proj_out"],
            },
        }
    )

    assert isinstance(config.quant_config, ModelOptFp8Config)
    assert config.quant_method == "modelopt"


def test_supported_methods_includes_vllm():
    from vllm_omni.quantization import SUPPORTED_QUANTIZATION_METHODS

    for method in ["fp8", "gguf", "awq", "gptq", "bitsandbytes", "modelopt"]:
        assert method in SUPPORTED_QUANTIZATION_METHODS, f"{method} missing"


def test_supported_methods_count():
    from vllm_omni.quantization import SUPPORTED_QUANTIZATION_METHODS

    assert len(SUPPORTED_QUANTIZATION_METHODS) >= 20


def test_per_component_routing_with_resolve():
    """Verify resolve() routes correctly by prefix."""
    from vllm_omni.quantization import ComponentQuantizationConfig, build_quant_config

    config = build_quant_config(
        {
            "transformer": {"method": "fp8"},
            "vae": None,
        }
    )
    assert isinstance(config, ComponentQuantizationConfig)

    assert config.resolve("transformer.blocks.0.attn.to_q") is not None
    assert config.resolve("transformer.blocks.0.attn.to_q").get_name() == "fp8"
    assert config.resolve("vae.encoder.conv_in") is None
    assert config.resolve("unknown.layer.0.weight") is None


def test_per_component_routing_with_default():
    """Verify default config applies to unmatched prefixes."""
    from vllm_omni.quantization import ComponentQuantizationConfig, build_quant_config

    config = build_quant_config(
        {
            "vae": None,
            "default": "fp8",
        }
    )
    assert isinstance(config, ComponentQuantizationConfig)

    assert config.resolve("vae.decoder.conv") is None
    resolved = config.resolve("transformer.blocks.0.attn")
    assert resolved is not None
    assert resolved.get_name() == "fp8"


@pytest.mark.parametrize("quant_algo", ["FP8"], ids=["modelopt_fp8"])
def test_omni_convertor_thinker_finds_text_config_quant(quant_algo):
    """Thinker stage should discover quantization_config from
    thinker_config.text_config for verified modelopt FP8 checkpoints."""
    from types import SimpleNamespace

    from vllm_omni.config.model import OmniModelArchConfigConvertor

    text_config = SimpleNamespace(
        quantization_config={
            "quant_method": "modelopt",
            "quant_algo": quant_algo,
            "ignore": ["lm_head", "model.layers.0.mlp.gate"],
        },
        model_type="qwen3_moe",
    )
    thinker_config = SimpleNamespace(text_config=text_config)
    hf_config = SimpleNamespace(
        thinker_config=thinker_config,
        talker_config=SimpleNamespace(text_config=SimpleNamespace()),
        model_type="qwen3_omni_moe",
    )

    convertor = OmniModelArchConfigConvertor(hf_config, text_config, stage_config_name="thinker_config")
    quant_cfg = convertor.get_quantization_config()

    assert quant_cfg is not None
    assert quant_cfg["quant_method"] == "modelopt"
    assert "lm_head" in quant_cfg["ignore"]


def test_omni_convertor_talker_returns_none():
    """Talker stage should get no quantization config when its text_config
    has no quantization_config (talker weights are BF16)."""
    from types import SimpleNamespace

    from vllm_omni.config.model import OmniModelArchConfigConvertor

    talker_text_config = SimpleNamespace(model_type="qwen3_omni_moe_talker")
    talker_config = SimpleNamespace(text_config=talker_text_config)
    hf_config = SimpleNamespace(
        talker_config=talker_config,
        model_type="qwen3_omni_moe",
    )

    convertor = OmniModelArchConfigConvertor(hf_config, talker_text_config, stage_config_name="talker_config")
    quant_cfg = convertor.get_quantization_config()

    assert quant_cfg is None


def test_omni_convertor_no_stage_name_falls_back():
    """Without stage_config_name, should fall back to base behavior."""
    from types import SimpleNamespace

    from vllm_omni.config.model import OmniModelArchConfigConvertor

    hf_config = SimpleNamespace(model_type="qwen3_omni_moe")
    text_config = SimpleNamespace()

    convertor = OmniModelArchConfigConvertor(hf_config, text_config)
    quant_cfg = convertor.get_quantization_config()

    assert quant_cfg is None


def test_multi_component_model_routing():
    """Integration test: walk a multi-component model and verify per-component
    quantization routes resolve() correctly for every linear layer."""
    from vllm_omni.quantization import ComponentQuantizationConfig, build_quant_config

    # Build a mock multi-stage model mimicking Bagel/Qwen3-Omni layout
    class MockTransformerBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn_q = nn.Linear(64, 64)
            self.attn_k = nn.Linear(64, 64)
            self.mlp = nn.Linear(64, 256)

    class MockVAEBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Linear(64, 64)

    class MockMultiStageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = nn.ModuleDict(
                {
                    "block_0": MockTransformerBlock(),
                    "block_1": MockTransformerBlock(),
                }
            )
            self.vae = nn.ModuleDict(
                {
                    "encoder": MockVAEBlock(),
                    "decoder": MockVAEBlock(),
                }
            )

    model = MockMultiStageModel()
    config = build_quant_config(
        {
            "transformer": {"method": "fp8", "activation_scheme": "dynamic"},
            "vae": None,
        }
    )
    assert isinstance(config, ComponentQuantizationConfig)

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            resolved = config.resolve(name)
            if name.startswith("transformer"):
                assert resolved is not None, f"{name} should be quantized"
                assert resolved.get_name() == "fp8"
            elif name.startswith("vae"):
                assert resolved is None, f"{name} should NOT be quantized"
