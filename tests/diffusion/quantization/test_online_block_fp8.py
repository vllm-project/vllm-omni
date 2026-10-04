# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Explicit FP8 block configuration preserves scope and serialized defaults."""

from types import SimpleNamespace

import pytest
from torch import nn

from vllm_omni.quantization import build_quant_config, online_block_fp8

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture
def rocm_config_platform(monkeypatch):
    monkeypatch.setattr(online_block_fp8, "current_platform", SimpleNamespace(is_rocm=lambda: True))


@pytest.mark.parametrize("block_size", [32, 64, 128])
def test_build_online_block_fp8_retains_checkpoint_precision(rocm_config_platform, block_size):
    spec = {"method": "fp8", "online_block_size": block_size, "ignored_layers": ["proj_out"]}
    config = build_quant_config(spec)

    assert isinstance(config, online_block_fp8.OnlineBlockFp8Config)
    assert not config.is_checkpoint_fp8_serialized
    assert config.weight_block_size is None
    assert config.activation_scheme == "dynamic"
    assert config.online_block_size == block_size
    assert config.ignored_layers == ["proj_out"]
    assert spec == {"method": "fp8", "online_block_size": block_size, "ignored_layers": ["proj_out"]}


@pytest.mark.parametrize("invalid", [None, True, 16, 48, 64.0, [64, 64]])
def test_online_block_fp8_rejects_invalid_groups(rocm_config_platform, invalid):
    with pytest.raises(ValueError, match="must be 32, 64 or 128"):
        build_quant_config({"method": "fp8", "online_block_size": invalid})


@pytest.mark.parametrize(
    "settings,match",
    [
        ({"is_checkpoint_fp8_serialized": True}, "non-serialized"),
        ({"weight_block_size": [128, 128]}, "non-serialized"),
        ({"activation_scheme": "static"}, "dynamic"),
    ],
)
def test_online_block_fp8_rejects_incompatible_checkpoint_modes(rocm_config_platform, settings, match):
    with pytest.raises(ValueError, match=match):
        build_quant_config({"method": "fp8", "online_block_size": 64, **settings})


def test_online_block_fp8_requires_supported_platform(monkeypatch):
    monkeypatch.setattr(online_block_fp8, "current_platform", SimpleNamespace(is_rocm=lambda: False))
    with pytest.raises(ValueError, match="supported on ROCm"):
        build_quant_config({"method": "fp8", "online_block_size": 64})


def test_default_and_serialized_fp8_keep_original_config():
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    assert type(build_quant_config("fp8")) is Fp8Config
    serialized = build_quant_config(
        {"method": "fp8", "is_checkpoint_fp8_serialized": True, "weight_block_size": [128, 128]}
    )
    assert type(serialized) is Fp8Config
    assert serialized.is_checkpoint_fp8_serialized
    assert serialized.weight_block_size == [128, 128]


def test_online_block_routing_preserves_original_non_linear_and_skipped_methods(rocm_config_platform, monkeypatch):
    config = online_block_fp8.OnlineBlockFp8Config(online_block_size=64)
    sentinel = object()
    monkeypatch.setattr(online_block_fp8.Fp8Config, "get_quant_method", lambda self, layer, prefix: sentinel)
    assert config.get_quant_method(nn.Module(), "ignored.layer") is sentinel


def test_online_block_method_matches_weight_and_activation_groups(rocm_config_platform, monkeypatch):
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape, create_fp8_quant_key

    # Kernel creation needs the live vLLM model context. Its constructor is
    # exercised by the full quality lane; this contract checks our group policy.
    monkeypatch.setattr(online_block_fp8.Fp8PerBlockOnlineLinearMethod, "__init__", lambda self: None)
    method = online_block_fp8.OnlineBlockFp8LinearMethod(64)
    assert method.weight_block_size == [64, 64]
    assert method.activation_quant_key == create_fp8_quant_key(static=False, group_shape=GroupShape(1, 64))
    assert method.weight_quant_key == create_fp8_quant_key(static=True, group_shape=GroupShape(64, 64))
