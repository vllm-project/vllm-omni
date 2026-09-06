# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for OmniVoiceConfig."""

import pytest
from transformers.configuration_utils import PretrainedConfig

from vllm_omni.transformers_utils.configs.omnivoice import OmniVoiceConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_is_pretrained_config_subclass():
    assert issubclass(OmniVoiceConfig, PretrainedConfig)
    assert OmniVoiceConfig.model_type == "omnivoice"


def test_audio_codec_defaults():
    cfg = OmniVoiceConfig()
    assert cfg.audio_vocab_size == 1025
    assert cfg.audio_mask_id == 1024
    assert cfg.num_audio_codebook == 8
    assert cfg.audio_codebook_weights == [8, 8, 6, 6, 4, 4, 2, 2]


def test_llm_defaults_hoisted_to_top_level():
    cfg = OmniVoiceConfig()
    assert cfg.hidden_size == cfg.llm_hidden_size == 1024
    assert cfg.num_attention_heads == cfg.llm_num_attention_heads == 16
    assert cfg.num_key_value_heads == cfg.llm_num_key_value_heads == 8
    assert cfg.num_hidden_layers == cfg.llm_num_hidden_layers == 28
    # head_dim default = hidden // heads
    assert cfg.head_dim == 1024 // 16


def test_llm_config_dict_overrides_defaults():
    cfg = OmniVoiceConfig(llm_config={"hidden_size": 512, "num_attention_heads": 8})
    assert cfg.llm_hidden_size == 512
    assert cfg.hidden_size == 512
    assert cfg.llm_num_attention_heads == 8
    assert cfg.head_dim == 512 // 8


def test_llm_config_accepts_pretrained_config_instance():
    inner = PretrainedConfig(hidden_size=256, num_attention_heads=4)
    cfg = OmniVoiceConfig(llm_config=inner)
    assert cfg.llm_hidden_size == 256
    assert cfg.hidden_size == 256


def test_llm_config_invalid_type_falls_back_to_defaults():
    # A non-dict, non-PretrainedConfig llm_config is ignored (defensive fallback).
    cfg = OmniVoiceConfig(llm_config=12345)
    assert cfg.llm_hidden_size == 1024
    assert cfg.hidden_size == 1024


def test_generation_config_nested_hyperparams_are_hoisted():
    # A nested generation_config dict is flattened via setdefault.
    cfg = OmniVoiceConfig(generation_config={"num_step": 50, "guidance_scale": 3.5})
    assert cfg.num_step == 50
    assert cfg.guidance_scale == 3.5


def test_generation_defaults():
    cfg = OmniVoiceConfig()
    assert cfg.num_step == 32
    assert cfg.guidance_scale == 2.0
    assert cfg.sample_rate == 24000
    assert cfg.frame_rate == 25


def test_cuda_graph_default_enabled(monkeypatch):
    monkeypatch.delenv("OMNIVOICE_CUDA_GRAPH", raising=False)
    cfg = OmniVoiceConfig()
    assert cfg.enable_cuda_graph is True


def test_cuda_graph_disabled_via_env(monkeypatch):
    monkeypatch.setenv("OMNIVOICE_CUDA_GRAPH", "0")
    cfg = OmniVoiceConfig()
    assert cfg.enable_cuda_graph is False


def test_explicit_cuda_graph_beats_env(monkeypatch):
    # config.json value takes precedence over the env var.
    monkeypatch.setenv("OMNIVOICE_CUDA_GRAPH", "0")
    cfg = OmniVoiceConfig(enable_cuda_graph=True)
    assert cfg.enable_cuda_graph is True


def test_get_text_config_returns_self():
    cfg = OmniVoiceConfig()
    assert cfg.get_text_config() is cfg


def test_to_dict_from_dict_roundtrip():
    cfg = OmniVoiceConfig(llm_config={"hidden_size": 512})
    restored = OmniVoiceConfig.from_dict(cfg.to_dict())
    assert restored.hidden_size == 512
    assert restored.num_step == 32
