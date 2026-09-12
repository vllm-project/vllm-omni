# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for CosyVoice3Config."""

import pytest
from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig

from vllm_omni.transformers_utils.configs.cosyvoice3 import CosyVoice3Config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_is_pretrained_config_subclass():
    assert issubclass(CosyVoice3Config, PretrainedConfig)
    assert CosyVoice3Config.model_type == "cosyvoice3"


def test_default_scalar_fields():
    cfg = CosyVoice3Config()
    assert cfg.sample_rate == 24000
    assert cfg.target_sr == 24000
    assert cfg.hidden_size == cfg.llm_output_size == 896
    assert cfg.num_attention_heads == 14
    assert cfg.num_hidden_layers == 24
    assert cfg.vocab_size == 151923
    assert cfg.token_frame_rate == 25
    assert cfg.token_mel_ratio == 2
    assert cfg.version == "cosyvoice3"


def test_default_eos_token_id_is_speech_stop():
    # Speech EOS defaults to speech_token_size + 1 (6561 + 1) so vLLM stops
    # generation at the right token.
    cfg = CosyVoice3Config()
    assert cfg.eos_token_id == 6562


def test_explicit_eos_token_id_is_respected():
    # setdefault must not override a caller-provided eos_token_id.
    cfg = CosyVoice3Config(eos_token_id=999)
    assert cfg.eos_token_id == 999


def test_feat_extractor_derived_from_sample_rate():
    cfg = CosyVoice3Config()
    assert cfg.feat_extractor["sampling_rate"] == cfg.sample_rate
    assert cfg.feat_extractor["num_mels"] == 80
    assert cfg.feat_extractor["n_fft"] == 1920
    assert cfg.feat_extractor["center"] is False


def test_llm_subconfig_consistency():
    cfg = CosyVoice3Config()
    assert cfg.llm["llm_input_size"] == cfg.llm_input_size
    assert cfg.llm["llm_output_size"] == cfg.llm_output_size
    assert cfg.llm["speech_token_size"] == 6561
    # The nested LLM eos is speech_token_size + 1, matching the top-level default.
    assert cfg.llm["eos_token_id"] == 6562
    assert cfg.llm["spk_embed_dim"] == cfg.spk_embed_dim


def test_flow_subconfig_derived_fields():
    cfg = CosyVoice3Config()
    assert cfg.flow["spk_embed_dim"] == cfg.spk_embed_dim
    assert cfg.flow["input_frame_rate"] == cfg.token_frame_rate
    assert cfg.flow["token_mel_ratio"] == cfg.token_mel_ratio
    # static_chunk_size = token_frame_rate * token_mel_ratio = 25 * 2 = 50
    assert cfg.flow["decoder"]["estimator"]["static_chunk_size"] == 50


def test_hift_sampling_rate_matches():
    cfg = CosyVoice3Config()
    assert cfg.hift["sampling_rate"] == cfg.sample_rate


def test_to_dict_from_dict_roundtrip():
    cfg = CosyVoice3Config()
    d = cfg.to_dict()
    assert d["model_type"] == "cosyvoice3"
    restored = CosyVoice3Config.from_dict(d)
    assert restored.sample_rate == cfg.sample_rate
    assert restored.vocab_size == cfg.vocab_size
    assert restored.eos_token_id == cfg.eos_token_id


def test_registered_with_autoconfig():
    # Import-time AutoConfig.register("cosyvoice3", ...) must have taken effect.
    assert AutoConfig.for_model("cosyvoice3").__class__ is CosyVoice3Config().__class__ or isinstance(
        AutoConfig.for_model("cosyvoice3"), CosyVoice3Config
    )
