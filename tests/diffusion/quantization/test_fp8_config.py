# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for the unified quantization framework."""

from types import SimpleNamespace

import pytest
import torch
from pytest_mock import MockerFixture
from torch import nn
from vllm.config import set_current_vllm_config
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.fp8 import Fp8KVCacheMethod
from vllm.model_executor.layers.quantization.modelopt import ModelOptFp8Config, ModelOptNvFp4Config
from vllm.model_executor.layers.quantization.online.base import OnlineQuantizationConfig
from vllm.model_executor.layers.quantization.online.fp8 import Fp8PerTensorOnlineLinearMethod
from vllm.model_executor.layers.quantization.utils.quant_utils import kFp8StaticTensorSym

from vllm_omni.config.model import OmniModelArchConfigConvertor
from vllm_omni.diffusion.data import OmniDiffusionConfig, TransformerConfig
from vllm_omni.diffusion.vllm_config import create_diffusion_vllm_config
from vllm_omni.quantization import (
    SUPPORTED_QUANTIZATION_METHODS,
    ComponentQuantizationConfig,
    build_quantization_config,
)
from vllm_omni.quantization.factory import resolve_quantization_config_from_disk
from vllm_omni.quantization.fp8_config import OmniFp8Config

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

LAYER_PREFIX = "blocks.0.attn"
GLOB_IGNORED_LAYER = "blocks.*.ff"
GLOB_IGNORED_PREFIX = "blocks.3.ff"
GLOB_KEPT_PREFIX = "blocks.3.attn"
FUSED_QKV_PREFIX = "blocks.0.attn.qkv_proj"
PACKED_QKV_MAPPING = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
IGNORED_QKV_SHARDS = ["blocks.0.attn.q_proj", "blocks.0.attn.k_proj", "blocks.0.attn.v_proj"]


@pytest.fixture
def bf16_vllm_config():
    """Set the model dtype that vLLM's online fp8 methods read on construction."""
    od_config = OmniDiffusionConfig(model=None, dtype=torch.bfloat16)
    with set_current_vllm_config(create_diffusion_vllm_config(torch.device("cpu"), od_config)):
        yield


@pytest.mark.parametrize("is_diffusion", [True, False])
def test_build_quantization_config_fp8(is_diffusion):
    config = build_quantization_config("fp8", is_diffusion=is_diffusion)
    assert isinstance(config, OmniFp8Config)
    assert config.get_name() == "fp8"
    assert config.activation_scheme == "dynamic"
    assert not config.is_checkpoint_fp8_serialized


def test_online_fp8_uses_upstream_online_linear_method(bf16_vllm_config, mocker: MockerFixture):
    """Ensure online fp8 quantizes linears with vLLM's online method."""
    config = build_quantization_config("fp8")
    linear = mocker.Mock(spec=LinearBase)

    method = config.get_quant_method(linear, LAYER_PREFIX)

    assert isinstance(method, Fp8PerTensorOnlineLinearMethod)


def test_online_fp8_ignored_layers_match_globs(bf16_vllm_config, mocker: MockerFixture):
    """Ensure online fp8 ignored_layers match glob patterns like vLLM's online config."""
    config = build_quantization_config({"method": "fp8", "ignored_layers": [GLOB_IGNORED_LAYER]})
    linear = mocker.Mock(spec=LinearBase)

    ignored_method = config.get_quant_method(linear, GLOB_IGNORED_PREFIX)
    kept_method = config.get_quant_method(linear, GLOB_KEPT_PREFIX)

    assert isinstance(ignored_method, UnquantizedLinearMethod)
    assert isinstance(kept_method, Fp8PerTensorOnlineLinearMethod)


def test_online_fp8_forwards_packed_modules_mapping(bf16_vllm_config, mocker: MockerFixture):
    """Ensure online fp8 forwards packed_modules_mapping so ignoring q/k/v_proj also skips the fused qkv_proj."""
    config = build_quantization_config({"method": "fp8", "ignored_layers": IGNORED_QKV_SHARDS})
    config.packed_modules_mapping = PACKED_QKV_MAPPING
    linear = mocker.Mock(spec=LinearBase)

    method = config.get_quant_method(linear, FUSED_QKV_PREFIX)

    assert isinstance(method, UnquantizedLinearMethod)


def test_only_serialized_fp8_loads_kv_cache_scales(mocker: MockerFixture):
    """Ensure only serialized fp8 checkpoints load KV-cache scales for attention."""
    online = build_quantization_config("fp8")
    serialized = build_quantization_config({"method": "fp8", "is_checkpoint_fp8_serialized": True})
    attention = mocker.Mock(spec=Attention)

    online_method = online.get_quant_method(attention, LAYER_PREFIX)
    serialized_method = serialized.get_quant_method(attention, LAYER_PREFIX)

    assert online_method is None
    assert isinstance(serialized_method, Fp8KVCacheMethod)


def test_build_quantization_config_upstream_online_fp8_shorthand():
    config = build_quantization_config({"method": "fp8_per_tensor", "ignore": ["proj_out"]})
    assert isinstance(config, OnlineQuantizationConfig)
    assert config.args.linear.weight == kFp8StaticTensorSym
    assert config.ignored_layers == ["proj_out"]


def test_online_fp8_rejects_static_activation():
    with pytest.raises(ValueError, match="activation_scheme='dynamic'"):
        build_quantization_config({"method": "fp8", "activation_scheme": "static"})


def test_build_online_quantization_config_from_fields():
    """Ensure "online" builds vLLM's online config from plain QuantizationConfigArgs fields in vLLM."""
    config = build_quantization_config({"method": "online", "linear": "fp8_per_tensor_static", "ignore": ["proj_out"]})
    assert isinstance(config, OnlineQuantizationConfig)
    assert config.args.linear.weight == kFp8StaticTensorSym
    assert config.ignored_layers == ["proj_out"]


def test_build_online_quantization_config_requires_args():
    with pytest.raises(ValueError, match="requires quantization config arguments"):
        build_quantization_config("online")


def test_serialized_checkpoint_replaces_online_fp8_config():
    config = resolve_quantization_config_from_disk(
        build_quantization_config("fp8"),
        {"quant_method": "fp8", "is_checkpoint_fp8_serialized": True, "activation_scheme": "static"},
    )
    assert isinstance(config, OmniFp8Config)
    assert config.is_checkpoint_fp8_serialized
    assert config.activation_scheme == "static"


def test_build_quantization_config_none():
    assert build_quantization_config(None) is None


def test_build_quantization_config_none_string():
    assert build_quantization_config("none") is None


def test_build_quantization_config_invalid():
    with pytest.raises(ValueError, match="Unknown quantization method"):
        build_quantization_config("invalid_method")


def test_build_quantization_config_dict():
    config = build_quantization_config(
        {"method": "fp8", "is_checkpoint_fp8_serialized": True, "activation_scheme": "static"}
    )
    assert config is not None
    assert config.get_name() == "fp8"
    assert config.activation_scheme == "static"


def test_build_quantization_config_dict_not_mutated():
    original = {"method": "fp8", "is_checkpoint_fp8_serialized": True, "activation_scheme": "static"}
    copy = original.copy()
    build_quantization_config(original)
    assert original == copy


def test_build_quantization_config_checkpoint_metadata_not_mutated():
    """Ensure checkpoint metadata remains unchanged during config construction."""
    metadata = {
        "quant_method": "fp8",
        "is_checkpoint_fp8_serialized": True,
        "activation_scheme": "static",
    }
    original = metadata.copy()

    config = build_quantization_config(None, metadata)

    assert isinstance(config, OmniFp8Config)
    assert config.is_checkpoint_fp8_serialized
    assert metadata == original


def test_build_quantization_config_modelopt_fp8_config_json():
    config = build_quantization_config(
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


def test_build_quantization_config_modelopt_nvfp4_from_str_and_dict():
    """Ensure (str method, chkpt dict) disambiguates NVFP4 like the dict-only form."""
    config = build_quantization_config(
        "modelopt",
        {"quant_method": "modelopt", "quant_algo": "NVFP4", "producer": {"name": "modelopt"}},
    )
    assert isinstance(config, ModelOptNvFp4Config)


def test_build_quantization_config_per_component():
    config = build_quantization_config({"transformer": {"method": "fp8"}, "vae": None})
    assert isinstance(config, ComponentQuantizationConfig)
    assert config.component_configs["transformer"].get_name() == "fp8"
    assert config.component_configs["vae"] is None


def test_build_quantization_config_per_component_string():
    config = build_quantization_config({"transformer": "fp8", "vae": None})
    assert isinstance(config, ComponentQuantizationConfig)
    assert config.component_configs["transformer"].get_name() == "fp8"


def test_build_quantization_config_per_component_inner_dict_not_mutated():
    """Inner component dicts should not be mutated by build_quantization_config."""
    inner = {"method": "fp8", "is_checkpoint_fp8_serialized": True, "activation_scheme": "static"}
    original = inner.copy()
    build_quantization_config({"transformer": inner, "vae": None})
    assert inner == original


def test_flat_dict_not_misdetected_as_per_component():
    """A flat config like {"activation_scheme": "static"} must NOT be treated as
    a per-component dict — it should raise ValueError for missing 'method'."""
    with pytest.raises(ValueError, match="must have a 'method' or 'quant_method' key"):
        build_quantization_config({"activation_scheme": "static"})


def test_build_quantization_config_conflicting_method_keys_raise():
    """Ensure a dict declaring both aliases with different values raises."""
    with pytest.raises(ValueError):
        build_quantization_config({"method": "int8", "quant_method": "fp8"})


def test_build_quantization_config_passthrough():
    fp8 = OmniFp8Config()
    assert build_quantization_config(fp8) is fp8


def test_component_config_routing():
    fp8 = OmniFp8Config()
    config = ComponentQuantizationConfig(component_configs={"transformer": fp8, "vae": None})

    assert config.get_name() == "component"
    assert config.resolve("transformer.blocks.0.attn") is fp8
    assert config.resolve("vae.encoder.conv_in") is None
    assert config.resolve("unknown.layer") is None


def test_component_config_with_default():
    fp8 = OmniFp8Config()
    config = ComponentQuantizationConfig(component_configs={"vae": None}, default_config=fp8)

    assert config.resolve("transformer.blocks.0") is fp8
    assert config.resolve("vae.encoder") is None


def test_integration_no_quant():
    config = OmniDiffusionConfig(model="test")
    assert config.quantization_config is None


def test_transformer_config_auto_detects_modelopt_fp8():
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
    for method in ["fp8", "awq", "gptq", "bitsandbytes", "modelopt"]:
        assert method in SUPPORTED_QUANTIZATION_METHODS, f"{method} missing"


def test_supported_methods_count():
    assert len(SUPPORTED_QUANTIZATION_METHODS) >= 20


def test_per_component_routing_with_resolve():
    """Verify resolve() routes correctly by prefix."""
    config = build_quantization_config({"transformer": {"method": "fp8"}, "vae": None})
    assert isinstance(config, ComponentQuantizationConfig)

    assert config.resolve("transformer.blocks.0.attn.to_q") is not None
    assert config.resolve("transformer.blocks.0.attn.to_q").get_name() == "fp8"
    assert config.resolve("vae.encoder.conv_in") is None
    assert config.resolve("unknown.layer.0.weight") is None


def test_per_component_routing_with_default():
    """Verify default config applies to unmatched prefixes."""
    config = build_quantization_config({"vae": None, "default": "fp8"})
    assert isinstance(config, ComponentQuantizationConfig)

    assert config.resolve("vae.decoder.conv") is None
    resolved = config.resolve("transformer.blocks.0.attn")
    assert resolved is not None
    assert resolved.get_name() == "fp8"


@pytest.mark.parametrize("quant_algo", ["FP8"], ids=["modelopt_fp8"])
def test_omni_convertor_thinker_finds_text_config_quant(quant_algo):
    """Thinker stage discovers quantization_config from thinker_config.text_config."""
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
    """Talker stage gets no quantization config (talker weights are BF16)."""
    talker_text_config = SimpleNamespace(model_type="qwen3_omni_moe_talker")
    talker_config = SimpleNamespace(text_config=talker_text_config)
    hf_config = SimpleNamespace(talker_config=talker_config, model_type="qwen3_omni_moe")

    convertor = OmniModelArchConfigConvertor(hf_config, talker_text_config, stage_config_name="talker_config")
    assert convertor.get_quantization_config() is None


def test_omni_convertor_no_stage_name_falls_back():
    """Without stage_config_name, should fall back to base behavior."""
    hf_config = SimpleNamespace(model_type="qwen3_omni_moe")
    convertor = OmniModelArchConfigConvertor(hf_config, SimpleNamespace())
    assert convertor.get_quantization_config() is None


def test_multi_component_model_routing():
    """Walk a multi-component model; verify per-component resolve() for each linear layer."""

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
            self.transformer = nn.ModuleDict({"block_0": MockTransformerBlock(), "block_1": MockTransformerBlock()})
            self.vae = nn.ModuleDict({"encoder": MockVAEBlock(), "decoder": MockVAEBlock()})

    model = MockMultiStageModel()
    config = build_quantization_config({"transformer": {"method": "fp8", "activation_scheme": "dynamic"}, "vae": None})
    assert isinstance(config, ComponentQuantizationConfig)

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            resolved = config.resolve(name)
            if name.startswith("transformer"):
                assert resolved is not None, f"{name} should be quantized"
                assert resolved.get_name() == "fp8"
            elif name.startswith("vae"):
                assert resolved is None, f"{name} should NOT be quantized"
