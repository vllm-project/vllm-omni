# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Configurable online FP8 block scaling uses current vLLM online APIs."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.config import set_current_vllm_config
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    create_fp8_quant_key,
    kFp8StaticTensorSym,
)

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.vllm_config import create_diffusion_vllm_config
from vllm_omni.quantization import build_quant_config, fp8_config
from vllm_omni.quantization.factory import resolve_quant_config_from_disk

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture
def rocm_platform(monkeypatch):
    monkeypatch.setattr(fp8_config, "current_platform", SimpleNamespace(is_rocm=lambda: True))


def _linear_stub() -> LinearBase:
    layer = LinearBase.__new__(LinearBase)
    nn.Module.__init__(layer)
    return layer


@pytest.mark.parametrize("block_size", [32, 64, 128])
def test_online_block_fp8_builds_matching_quant_specs(rocm_platform, block_size):
    spec = {"method": "fp8", "online_block_size": block_size, "ignored_layers": ["*proj_out*"]}
    config = build_quant_config(spec)

    expected_weight = create_fp8_quant_key(
        static=True,
        group_shape=GroupShape(block_size, block_size),
    )
    expected_activation = create_fp8_quant_key(
        static=False,
        group_shape=GroupShape(1, block_size),
    )
    assert isinstance(config, fp8_config.DiffusionFp8Config)
    assert config.args.linear.weight == expected_weight
    assert config.args.linear.activation == expected_activation
    assert config.args.moe.weight == kFp8StaticTensorSym
    assert config.online_block_size == block_size
    assert config.weight_block_size is None
    assert config.ignored_layers == ["*proj_out*"]
    assert spec == {"method": "fp8", "online_block_size": block_size, "ignored_layers": ["*proj_out*"]}


def test_online_block_fp8_routes_real_linear_and_preserves_skips(rocm_platform):
    config = build_quant_config({"method": "fp8", "online_block_size": 64, "ignored_layers": ["*proj_out*"]})
    layer = _linear_stub()
    od_config = OmniDiffusionConfig(model="test", dtype=torch.bfloat16, quantization_config=config)
    vllm_config = create_diffusion_vllm_config(torch.device("cpu"), od_config)

    with set_current_vllm_config(vllm_config):
        method = config.get_quant_method(layer, "transformer.blocks.0.attn.to_q")
        assert isinstance(method, fp8_config.OnlineBlockFp8LinearMethod)
        assert method.weight_block_size == [64, 64]
        assert method.activation_quant_key == create_fp8_quant_key(
            static=False,
            group_shape=GroupShape(1, 64),
        )
        assert method.weight_quant_key == create_fp8_quant_key(
            static=True,
            group_shape=GroupShape(64, 64),
        )
        assert "transformer.blocks.0.attn.to_q" in config.quantized_layers

        skipped = config.get_quant_method(layer, "transformer.proj_out")
        assert isinstance(skipped, UnquantizedLinearMethod)
        assert config.get_quant_method(nn.Module(), "transformer.norm") is None


@pytest.mark.parametrize("invalid", [True, 16, 48, 64.0, [64, 64]])
def test_online_block_fp8_rejects_invalid_groups(rocm_platform, invalid):
    with pytest.raises(ValueError, match="must be 32, 64 or 128"):
        build_quant_config({"method": "fp8", "online_block_size": invalid})


@pytest.mark.parametrize(
    ("settings", "match"),
    [
        ({"is_checkpoint_fp8_serialized": True}, "Serialized FP8"),
        ({"weight_block_size": [128, 128]}, "serialized checkpoint or fp8_per_block"),
        ({"activation_scheme": "static"}, "activation_scheme='dynamic'"),
        ({"store_dtype": "auto"}, "does not support store_dtype"),
        ({"weight_scheme": "per_channel"}, "ordinary BF16/FP16"),
    ],
)
def test_online_block_fp8_rejects_incompatible_modes(rocm_platform, settings, match):
    with pytest.raises(ValueError, match=match):
        build_quant_config({"method": "fp8", "online_block_size": 64, **settings})


def test_online_block_fp8_requires_rocm(monkeypatch):
    monkeypatch.setattr(fp8_config, "current_platform", SimpleNamespace(is_rocm=lambda: False))
    with pytest.raises(ValueError, match="supported on ROCm only"):
        build_quant_config({"method": "fp8", "online_block_size": 64})


def test_default_fp8_remains_per_tensor():
    config = build_quant_config("fp8")
    assert isinstance(config, fp8_config.DiffusionFp8Config)
    assert config.args.linear.weight == kFp8StaticTensorSym
    assert config.args.linear.activation is None
    assert config.online_block_size is None


def test_disk_rebuild_preserves_online_block_size(rocm_platform):
    active = build_quant_config({"method": "fp8", "online_block_size": 64, "ignored_layers": ["*proj_out*"]})
    rebuilt = resolve_quant_config_from_disk(
        active,
        {"quant_method": "fp8", "ignored_layers": ["*final_layer*"]},
    )
    assert isinstance(rebuilt, fp8_config.DiffusionFp8Config)
    assert rebuilt.online_block_size == 64
    assert rebuilt.ignored_layers == ["*final_layer*"]


def test_serialized_disk_metadata_replaces_online_block_mode(rocm_platform):
    active = build_quant_config({"method": "fp8", "online_block_size": 64})
    rebuilt = resolve_quant_config_from_disk(
        active,
        {
            "quant_method": "fp8",
            "is_checkpoint_fp8_serialized": True,
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
        },
    )
    assert isinstance(rebuilt, Fp8Config)
    assert rebuilt.is_checkpoint_fp8_serialized
    assert rebuilt.activation_scheme == "dynamic"
    assert rebuilt.weight_block_size == [128, 128]
