# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for VibeVoice TTS configuration parsing.

Verifies that VibeVoiceTTSConfig correctly parses the config.json
from microsoft/VibeVoice-1.5B and exposes the expected sub-configs.
"""

import pytest

from vllm_omni.model_executor.models.vibevoice_tts.configuration_vibevoice_tts import (
    VibeVoiceAcousticTokenizerConfig,
    VibeVoiceDiffusionHeadConfig,
    VibeVoiceSemanticTokenizerConfig,
    VibeVoiceTTSConfig,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# Mirrors the real microsoft/VibeVoice-1.5B config.json
SAMPLE_CONFIG_DICT = {
    "model_type": "vibevoice",
    "acoustic_vae_dim": 64,
    "semantic_vae_dim": 128,
    "acoustic_tokenizer_config": {
        "causal": True,
        "channels": 1,
        "vae_dim": 64,
        "fix_std": 0.5,
        "std_dist_type": "gaussian",
        "encoder_depths": "3-3-3-3-3-3-8",
        "encoder_n_filters": 32,
        "encoder_ratios": [8, 5, 5, 4, 2, 2],
        "decoder_n_filters": 32,
        "decoder_ratios": [8, 5, 5, 4, 2, 2],
        "layernorm": "RMSNorm",
        "mixer_layer": "depthwise_conv",
    },
    "semantic_tokenizer_config": {
        "causal": True,
        "channels": 1,
        "vae_dim": 128,
        "fix_std": 0,
        "std_dist_type": "none",
        "encoder_depths": "3-3-3-3-3-3-8",
        "encoder_n_filters": 32,
        "encoder_ratios": [8, 5, 5, 4, 2, 2],
    },
    "decoder_config": {
        "model_type": "qwen2",
        "hidden_size": 1536,
        "num_attention_heads": 12,
        "num_hidden_layers": 28,
        "num_key_value_heads": 2,
        "intermediate_size": 8960,
        "vocab_size": 151936,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
    },
    "diffusion_head_config": {
        "hidden_size": 1536,
        "head_layers": 4,
        "head_ffn_ratio": 3.0,
        "rms_norm_eps": 1e-5,
        "latent_size": 64,
        "prediction_type": "v_prediction",
        "ddpm_num_steps": 1000,
        "ddpm_num_inference_steps": 20,
        "ddpm_beta_schedule": "cosine",
    },
}


class TestVibeVoiceTTSConfig:
    def test_from_dict(self):
        """Config can be created from nested dict (as HF AutoConfig does)."""
        cfg = VibeVoiceTTSConfig(**SAMPLE_CONFIG_DICT)
        assert cfg.model_type == "vibevoice"

    def test_decoder_config_is_qwen2(self):
        """decoder_config is parsed as Qwen2Config."""
        cfg = VibeVoiceTTSConfig(**SAMPLE_CONFIG_DICT)
        assert cfg.decoder_config.hidden_size == 1536
        assert cfg.decoder_config.num_attention_heads == 12
        assert cfg.decoder_config.num_hidden_layers == 28
        assert cfg.decoder_config.vocab_size == 151936

    def test_acoustic_tokenizer_config(self):
        """acoustic_tokenizer_config is parsed correctly."""
        cfg = VibeVoiceTTSConfig(**SAMPLE_CONFIG_DICT)
        ac = cfg.acoustic_tokenizer_config
        assert isinstance(ac, VibeVoiceAcousticTokenizerConfig)
        assert ac.vae_dim == 64
        assert ac.causal is True
        assert ac.encoder_ratios == [8, 5, 5, 4, 2, 2]
        assert ac.fix_std == 0.5
        assert ac.layernorm == "RMSNorm"

    def test_semantic_tokenizer_config(self):
        """semantic_tokenizer_config is parsed correctly."""
        cfg = VibeVoiceTTSConfig(**SAMPLE_CONFIG_DICT)
        sc = cfg.semantic_tokenizer_config
        assert isinstance(sc, VibeVoiceSemanticTokenizerConfig)
        assert sc.vae_dim == 128
        assert sc.fix_std == 0
        assert sc.std_dist_type == "none"

    def test_diffusion_head_config(self):
        """diffusion_head_config is parsed correctly."""
        cfg = VibeVoiceTTSConfig(**SAMPLE_CONFIG_DICT)
        dh = cfg.diffusion_head_config
        assert isinstance(dh, VibeVoiceDiffusionHeadConfig)
        assert dh.hidden_size == 1536
        assert dh.head_layers == 4
        assert dh.latent_size == 64
        assert dh.prediction_type == "v_prediction"
        assert dh.ddpm_num_inference_steps == 20
        assert dh.ddpm_beta_schedule == "cosine"

    def test_get_text_config(self):
        """get_text_config() returns decoder_config for vLLM profiling."""
        cfg = VibeVoiceTTSConfig(**SAMPLE_CONFIG_DICT)
        tc = cfg.get_text_config()
        assert tc.hidden_size == 1536
        assert tc.num_attention_heads == 12

    def test_vae_dims(self):
        """Top-level VAE dims are set correctly."""
        cfg = VibeVoiceTTSConfig(**SAMPLE_CONFIG_DICT)
        assert cfg.acoustic_vae_dim == 64
        assert cfg.semantic_vae_dim == 128

    def test_sampling_rate_and_frame_rate(self):
        """Properties return expected values."""
        cfg = VibeVoiceTTSConfig(**SAMPLE_CONFIG_DICT)
        assert cfg.sampling_rate == 24000
        assert cfg.frame_rate == 7.5

    def test_defaults_when_none(self):
        """Config with all None sub-configs uses defaults."""
        cfg = VibeVoiceTTSConfig()
        assert cfg.acoustic_vae_dim == 64
        assert cfg.semantic_vae_dim == 128
        assert cfg.decoder_config is not None
        assert cfg.diffusion_head_config is not None
        assert cfg.acoustic_tokenizer_config is not None
        assert cfg.semantic_tokenizer_config is not None

    def test_decoder_ratios_default_to_encoder(self):
        """When decoder_ratios is None, it defaults to encoder_ratios."""
        ac = VibeVoiceAcousticTokenizerConfig(
            encoder_ratios=[8, 5, 5, 4, 2, 2],
            decoder_ratios=None,
        )
        assert ac.decoder_ratios == [8, 5, 5, 4, 2, 2]

    def test_decoder_ratios_override(self):
        """Explicit decoder_ratios overrides the default."""
        ac = VibeVoiceAcousticTokenizerConfig(
            encoder_ratios=[8, 5, 5, 4, 2, 2],
            decoder_ratios=[2, 2, 4, 5, 5, 8],
        )
        assert ac.decoder_ratios == [2, 2, 4, 5, 5, 8]
