# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The config's constants, and the variant switch that every deploy file relies on."""

import pytest

from vllm_omni.transformers_utils.configs.chatterbox import VARIANT_FIELDS, ChatterboxConfig, validate_flow_settings

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_token_space() -> None:
    cfg = ChatterboxConfig()
    assert cfg.model_type == "chatterbox"
    # No tokenizer is loaded, so the sampler's vocabulary is the speech head's.
    assert cfg.vocab_size == cfg.speech_vocab_size == 6563
    assert cfg.text_vocab_size == 50276
    assert cfg.speech_token_limit == cfg.start_speech_token == 6561
    assert cfg.eos_token_id == cfg.stop_speech_token == 6562


def test_turbo_defaults_match_upstream() -> None:
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


def test_original_defaults_match_upstream() -> None:
    cfg = ChatterboxConfig("original")
    assert (cfg.vocab_size, cfg.speech_vocab_size, cfg.text_vocab_size) == (8194, 8194, 704)
    assert (cfg.num_hidden_layers, cfg.max_position_embeddings, cfg.activation_function) == (30, 131072, "silu")
    assert (cfg.cond_prompt_len, cfg.enc_cond_seconds, cfg.min_ref_seconds) == (150, 6, 0.0)
    assert cfg.loudness_target_lufs is None
    assert (cfg.n_silence_tokens, cfg.meanflow, cfg.n_cfm_timesteps) == (0, False, 10)
    assert (cfg.t3_weights, cfg.s3gen_weights) == ("t3_cfg.safetensors", "s3gen.safetensors")


def test_survives_a_config_json_round_trip() -> None:
    """The engine builds the config from a two-key config.json it writes itself."""
    cfg = ChatterboxConfig.from_dict(
        {"model_type": "chatterbox", "architectures": ["ChatterboxForConditionalGeneration"]}
    )
    assert (cfg.vocab_size, cfg.text_vocab_size) == (6563, 50276)
    assert cfg.architectures == ["ChatterboxForConditionalGeneration"]
    restored = ChatterboxConfig.from_dict(ChatterboxConfig("original", flow_dtype="bfloat16").to_dict())
    assert (restored.variant, restored.flow_dtype, restored.num_hidden_layers) == ("original", "bfloat16", 30)


@pytest.mark.parametrize("variant", ["turbo", "original"])
def test_variant_override_after_construction_switches_every_checkpoint_field(variant: str) -> None:
    """vLLM applies a deploy file's hf_overrides through update, after construction."""
    config = ChatterboxConfig("original" if variant == "turbo" else "turbo")
    config.update({"variant": variant})
    expected = ChatterboxConfig(variant)
    assert {name: getattr(config, name) for name in VARIANT_FIELDS[variant]} == {
        name: getattr(expected, name) for name in VARIANT_FIELDS[variant]
    }
    assert config.to_dict() == expected.to_dict()


def test_unknown_variant_fails_on_first_use() -> None:
    with pytest.raises(ValueError, match="variant"):
        ChatterboxConfig("multilingual")
    config = ChatterboxConfig()
    config.update({"variant": "multilingual"})
    with pytest.raises(ValueError, match="multilingual"):
        _ = config.num_hidden_layers


def test_checkpoint_fields_cannot_be_overridden_one_by_one() -> None:
    """A deploy file switches the checkpoint with variant, never a field of it."""
    config = ChatterboxConfig()
    with pytest.raises(AttributeError):
        config.update({"num_hidden_layers": 30})


@pytest.mark.parametrize("settings", [{"flow_dtype": "float16"}, {"flow_attention_backend": "flash"}])
def test_unknown_flow_settings_are_rejected_at_construction_and_on_override(settings: dict) -> None:
    with pytest.raises(ValueError, match="flow"):
        ChatterboxConfig(**settings)
    config = ChatterboxConfig()
    with pytest.raises(ValueError, match="flow"):
        config.update(settings)


def test_cudnn_attention_needs_bf16_flow() -> None:
    with pytest.raises(ValueError, match="cuDNN"):
        ChatterboxConfig(flow_attention_backend="cudnn")
    with pytest.raises(ValueError, match="cuDNN"):
        validate_flow_settings("float32", "cudnn")
    validate_flow_settings("bfloat16", "cudnn")
