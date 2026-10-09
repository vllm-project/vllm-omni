# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import pytest

from vllm_omni.transformers_utils.configs.chatterbox import ChatterboxConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_token_space_001() -> None:
    cfg = ChatterboxConfig()
    assert cfg.model_type == "chatterbox"
    # No tokenizer is loaded, so the sampler's vocabulary is the speech head's.
    assert cfg.vocab_size == cfg.speech_vocab_size == 6563
    assert cfg.text_vocab_size == 50276
    assert cfg.speech_token_limit == cfg.start_speech_token == 6561
    assert cfg.eos_token_id == cfg.stop_speech_token == 6562


def test_turbo_defaults_match_upstream_001() -> None:
    cfg = ChatterboxConfig()
    assert cfg.variant == "turbo"
    assert (cfg.hidden_size, cfg.num_hidden_layers, cfg.num_attention_heads) == (1024, 24, 16)
    assert cfg.max_position_embeddings == 8196
    assert cfg.cond_prompt_len == 375
    assert (cfg.sample_rate, cfg.token_rate, cfg.token_mel_ratio) == (24000, 25, 2)
    assert (cfg.n_cfm_timesteps, cfg.meanflow) == (2, True)
    assert cfg.t3_weights == "t3_turbo_v1.safetensors"
    assert cfg.s3gen_weights == "s3gen_meanflow.safetensors"
    assert cfg.mel == {
        "n_fft": 1920,
        "num_mels": 80,
        "sampling_rate": 24000,
        "hop_size": 480,
        "win_size": 1920,
        "fmin": 0,
        "fmax": 8000,
        "center": False,
    }


def test_survives_a_config_json_round_trip_001() -> None:
    """The engine builds the config from a two-key config.json it writes itself."""
    cfg = ChatterboxConfig.from_dict(
        {"model_type": "chatterbox", "architectures": ["ChatterboxForConditionalGeneration"]}
    )
    assert (cfg.vocab_size, cfg.text_vocab_size) == (6563, 50276)
    assert cfg.architectures == ["ChatterboxForConditionalGeneration"]
