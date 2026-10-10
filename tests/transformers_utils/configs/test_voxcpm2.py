# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for VoxCPM2Config."""

import math

import pytest
from transformers.configuration_utils import PretrainedConfig

from vllm_omni.transformers_utils.configs.voxcpm2 import VoxCPM2Config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_is_pretrained_config_subclass():
    assert issubclass(VoxCPM2Config, PretrainedConfig)
    assert VoxCPM2Config.model_type == "voxcpm2"


def test_defaults_when_no_lm_config():
    cfg = VoxCPM2Config()
    assert cfg.vocab_size == 73448
    assert cfg.hidden_size == 2048
    assert cfg.num_hidden_layers == 28
    assert cfg.num_attention_heads == 16
    assert cfg.num_key_value_heads == 2
    assert cfg.hidden_act == "silu"
    assert cfg.tie_word_embeddings is False
    assert cfg.num_experts == 0
    # empty nested dicts
    assert cfg.lm_config == {}
    assert cfg.dit_config == {}


def test_head_dim_default_is_hidden_over_heads():
    cfg = VoxCPM2Config()
    assert cfg.head_dim == cfg.hidden_size // cfg.num_attention_heads


def test_kv_channels_overrides_head_dim():
    cfg = VoxCPM2Config(lm_config={"kv_channels": 128})
    assert cfg.head_dim == 128


def test_lm_config_hoisted_to_top_level():
    cfg = VoxCPM2Config(lm_config={"hidden_size": 512, "num_hidden_layers": 4, "vocab_size": 100})
    assert cfg.hidden_size == 512
    assert cfg.num_hidden_layers == 4
    assert cfg.vocab_size == 100
    # unspecified fields fall back to signature defaults
    assert cfg.num_attention_heads == 16


def test_mup_neutralized_when_use_mup_false():
    # use_mup defaults to False -> scale_emb=1.0, scale_depth=sqrt(N),
    # dim_model_base=hidden_size (so vllm's always-on muP cancels out).
    cfg = VoxCPM2Config(lm_config={"hidden_size": 256, "num_hidden_layers": 9})
    assert cfg.scale_emb == 1.0
    assert cfg.scale_depth == pytest.approx(math.sqrt(9))
    assert cfg.dim_model_base == cfg.hidden_size


def test_mup_honored_when_use_mup_true():
    cfg = VoxCPM2Config(lm_config={"use_mup": True, "scale_depth": 1.4, "dim_model_base": 64, "scale_emb": 12.0})
    assert cfg.scale_emb == 12.0
    assert cfg.scale_depth == 1.4
    assert cfg.dim_model_base == 64


def test_no_rope_scaling_by_default():
    cfg = VoxCPM2Config()
    assert cfg.rope_scaling is None


def test_rope_scaling_type_key_renamed_and_factor_filled():
    # "type" -> "rope_type", and a missing "factor" is defaulted to 1.0.
    cfg = VoxCPM2Config(lm_config={"rope_scaling": {"type": "longrope", "long_factor": [1.0], "short_factor": [1.0]}})
    assert cfg.rope_scaling["rope_type"] == "longrope"
    assert "type" not in cfg.rope_scaling
    assert cfg.rope_scaling["factor"] == 1.0
    # rope_parameters is built for vllm's MiniCPMAttention with theta injected.
    assert cfg.rope_parameters["rope_theta"] == cfg.rope_theta
    assert cfg.rope_parameters["rope_type"] == "longrope"


def test_get_text_config_returns_self():
    cfg = VoxCPM2Config()
    assert cfg.get_text_config() is cfg


def test_keys_to_ignore_at_inference():
    assert "past_key_values" in VoxCPM2Config.keys_to_ignore_at_inference


def test_to_dict_from_dict_roundtrip():
    cfg = VoxCPM2Config(lm_config={"hidden_size": 512, "num_hidden_layers": 4})
    restored = VoxCPM2Config.from_dict(cfg.to_dict())
    assert restored.hidden_size == 512
    assert restored.num_hidden_layers == 4
    assert restored.scale_depth == pytest.approx(math.sqrt(4))
