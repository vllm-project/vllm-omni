# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
from transformers import AutoConfig, Qwen2Config

from vllm_omni.transformers_utils.configs.llama_omni2 import LlamaOmni2Config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _real_checkpoint_config() -> dict:
    return {
        "architectures": ["Omni2Speech2SQwen2ForCausalLM"],
        "attention_dropout": 0.0,
        "bos_token_id": 151643,
        "eos_token_id": 151645,
        "hidden_act": "silu",
        "hidden_size": 896,
        "initializer_range": 0.02,
        "intermediate_size": 4864,
        "max_position_embeddings": 32768,
        "max_window_layers": 21,
        "num_attention_heads": 14,
        "num_hidden_layers": 24,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1_000_000.0,
        "sliding_window": None,
        "speech_encoder": "models/speech_encoder/large-v3.pt",
        "speech_encoder_ds_rate": 5,
        "speech_encoder_hidden_size": 1280,
        "speech_encoder_type": "whisper",
        "speech_generator": {
            "architectures": ["Qwen2ForCausalLM"],
            "attention_dropout": 0.0,
            "bos_token_id": 151643,
            "eos_token_id": 151643,
            "hidden_act": "silu",
            "hidden_size": 896,
            "initializer_range": 0.02,
            "intermediate_size": 4864,
            "max_position_embeddings": 32768,
            "max_window_layers": 24,
            "model_type": "qwen2",
            "num_attention_heads": 14,
            "num_hidden_layers": 24,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1_000_000.0,
            "sliding_window": None,
            "tie_word_embeddings": True,
            "use_cache": True,
            "use_mrope": False,
            "use_sliding_window": False,
            "vocab_size": 158227,
        },
        "speech_projector_type": "linear",
        "stream_params": "(3,10)",
        "tie_word_embeddings": True,
        "tokenizer_model_max_length": 4096,
        "tokenizer_padding_side": "right",
        "unit_vocab_size": 6561,
        "use_cache": True,
        "use_sliding_window": False,
        "vocab_size": 151936,
    }


def test_real_checkpoint_fields_are_normalized_for_both_qwen2_stages():
    config = LlamaOmni2Config(**_real_checkpoint_config())

    assert config.model_type == "omni2_speech2s_qwen2"
    assert config.architectures == ["Omni2Speech2SQwen2ForCausalLM"]
    assert isinstance(config.thinker_config, Qwen2Config)
    assert isinstance(config.talker_config, Qwen2Config)
    assert config.thinker_config.hidden_size == 896
    assert config.thinker_config.vocab_size == 151936
    assert config.talker_config.hidden_size == 896
    assert config.talker_config.vocab_size == 158227
    assert config.stream_text_tokens == 3
    assert config.stream_unit_tokens == 10
    assert config.unit_vocab_size == 6561


def test_auto_config_registration_builds_llama_omni2_config():
    config = AutoConfig.for_model("omni2_speech2s_qwen2", **_real_checkpoint_config())

    assert isinstance(config, LlamaOmni2Config)


@pytest.mark.parametrize(
    "stream_params",
    [
        "",
        "(3,)",
        "(3, 10, 12)",
        "(0, 10)",
        "(3, 0)",
        "(-3, 10)",
        "(3.0, 10)",
        "['3', 10]",
    ],
)
def test_stream_params_reject_malformed_or_non_positive_values(stream_params):
    raw = _real_checkpoint_config()
    raw["stream_params"] = stream_params

    with pytest.raises(ValueError, match="stream_params"):
        LlamaOmni2Config(**raw)


def test_stream_params_reject_code_execution(monkeypatch):
    raw = _real_checkpoint_config()
    raw["stream_params"] = "__import__('os').system('false')"
    called = False

    def fail_system(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("stream_params executed code")

    monkeypatch.setattr("os.system", fail_system)

    with pytest.raises(ValueError, match="stream_params"):
        LlamaOmni2Config(**raw)

    assert called is False
