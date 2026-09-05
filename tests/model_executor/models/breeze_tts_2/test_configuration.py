import pytest
from transformers.models.qwen3 import Qwen3Config

from vllm_omni.model_executor.models.breeze_tts_2.configuration_breeze_tts_2 import (
    BreezeTTS2Config,
    BreezeTTS2DepthDecoderConfig,
    BreezeTTS2TextEncoderConfig,
)


def _config() -> BreezeTTS2Config:
    return BreezeTTS2Config(
        backbone_config={
            "model_type": "qwen3",
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "vocab_size": 128,
            "max_position_embeddings": 512,
        },
        text_encoder_config={
            "model_type": "t5gemma2_text",
            "hidden_size": 24,
            "intermediate_size": 48,
            "num_hidden_layers": 6,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "vocab_size": 256,
        },
        depth_decoder_config={
            "model_type": "breeze_depth_decoder_model",
            "hidden_size": 32,
            "backbone_hidden_size": 32,
            "audio_embed_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "num_codebooks": 16,
            "vocab_size": 2051,
            "rope_scaling": {
                "rope_type": "llama3",
                "factor": 32.0,
                "low_freq_factor": 0.001953125,
                "high_freq_factor": 0.0078125,
                "original_max_position_embeddings": 16,
            },
        },
        codec_config={"model_type": "mimi", "codebook_size": 2048, "num_quantizers": 32},
        num_codebooks=16,
        vocab_size=2051,
        text_vocab_size=262158,
    )


def test_nested_configs_and_text_config_use_qwen_backbone():
    config = _config()

    assert isinstance(config.backbone_config, Qwen3Config)
    assert config.get_text_config() is config.backbone_config
    assert config.get_text_config().vocab_size == 128
    assert isinstance(config.text_encoder_config, BreezeTTS2TextEncoderConfig)
    assert isinstance(config.depth_decoder_config, BreezeTTS2DepthDecoderConfig)
    assert config.num_codebooks == config.depth_decoder_config.num_codebooks == 16
    assert config.codec_config["codebook_size"] == 2048


def test_transformers_five_roundtrip_preserves_nested_configs_and_rope():
    config = _config()

    roundtrip = BreezeTTS2Config(**dict(config.to_dict()))

    assert isinstance(roundtrip.backbone_config, Qwen3Config)
    assert isinstance(roundtrip.text_encoder_config, BreezeTTS2TextEncoderConfig)
    assert isinstance(roundtrip.depth_decoder_config, BreezeTTS2DepthDecoderConfig)
    assert roundtrip.depth_decoder_config.rope_parameters["rope_theta"] == 500_000.0


def test_nested_autoconfig_registration_is_idempotent():
    import importlib

    module = importlib.import_module(
        "vllm_omni.model_executor.models.breeze_tts_2.configuration_breeze_tts_2"
    )
    importlib.reload(module)

    assert module.BreezeTTS2Config.model_type == "breeze"


def test_text_encoder_defaults_to_upstream_layer_pattern():
    config = BreezeTTS2TextEncoderConfig(num_hidden_layers=6)

    assert config.layer_types == [
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ]


def test_invalid_codebook_layout_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        BreezeTTS2DepthDecoderConfig(num_codebooks=0)
