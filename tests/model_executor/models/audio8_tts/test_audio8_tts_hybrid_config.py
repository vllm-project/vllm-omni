# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Config-layer tests for the hybrid-Mamba2 Audio8 checkpoint variant (0.1b).

The two checkpoints share ``model_type = "arktts"`` but not their slow branch:
the 0.6b is dense/Qwen2-shaped, the 0.1b declares ``slow_backbone = "falcon_h1"``
and is muP-scaled. These tests pin the flat-field -> config mapping for both, so
a future edit cannot silently route the 0.1b through the dense class.
"""

import pytest
from transformers import FalconH1Config

from vllm_omni.model_executor.models.audio8_tts.configuration_audio8_tts import (
    Audio8TTSConfig,
    Audio8TTSHybridSlowARConfig,
    Audio8TTSSlowARConfig,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

#: Flat fields from Audio8-TTS-Preview-0.1b ``config.json`` (trimmed to the ones
#: this module maps; the full file also carries codec/fast/RAS fields).
CONFIG_0P1B = {
    "architectures": ["ArkttsModel"],
    "model_type": "arktts",
    "slow_backbone": "falcon_h1",
    "vocab_size": 69633,
    "dim": 512,
    "hidden_size": 512,
    "n_layer": 24,
    "n_head": 8,
    "n_local_heads": 2,
    "head_dim": 64,
    "intermediate_size": 768,
    "max_seq_len": 2048,
    "rope_base": 100000000000.0,
    "norm_eps": 1e-05,
    "attention_qkv_bias": False,
    "attention_qk_norm": False,
    "embedding_multiplier": 0.10888671875,
    "lm_head_multiplier": 0.078125,
    "mlp_multipliers": [1.0, 1.0],
    "ssm_multipliers": [1.0, 1.0, 1.0, 1.0, 1.0],
    "mamba_d_ssm": 768,
    "mamba_n_heads": 24,
    "mamba_d_state": 64,
    "mamba_d_conv": 4,
    "mamba_chunk_size": 128,
    "mamba_expand": 2,
    "mamba_n_groups": 1,
    "codebook_size": 4096,
    "num_codebooks": 10,
    "semantic_begin_id": 65537,
    "semantic_end_id": 69632,
    "eos_token_id": 228,
    "pad_token_id": 0,
    "n_fast_layer": 4,
    "fast_dim": 512,
    "fast_n_head": 8,
    "fast_n_local_heads": 2,
    "fast_head_dim": 64,
    "fast_intermediate_size": 4864,
}

#: Minimal 0.6b-shaped config: no ``slow_backbone``, no muP scalars.
CONFIG_0P6B_BACKBONE_FIELDS = {
    "model_type": "arktts",
    "vocab_size": 155776,
    "dim": 896,
    "n_layer": 24,
    "n_head": 14,
    "n_local_heads": 2,
    "head_dim": 64,
    "intermediate_size": 4864,
    "max_seq_len": 2048,
    "rope_base": 1000000.0,
    "eos_token_id": 151645,
    "pad_token_id": 151643,
    "semantic_begin_id": 151678,
    "semantic_end_id": 155773,
}


def test_0p1b_selects_hybrid_slow_ar_config():
    cfg = Audio8TTSConfig(**CONFIG_0P1B)
    text_config = cfg.get_text_config()

    assert isinstance(text_config, Audio8TTSHybridSlowARConfig)
    # The wrap in ``audio8_tts_hybrid_slow_ar`` hands this object straight to
    # vLLM's native FalconH1ForCausalLM, so the isinstance check is load-bearing.
    assert isinstance(text_config, FalconH1Config)
    assert cfg.slow_backbone == "falcon_h1"


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("vocab_size", 69633),
        ("hidden_size", 512),
        ("num_hidden_layers", 24),
        ("num_attention_heads", 8),
        ("num_key_value_heads", 2),
        ("head_dim", 64),
        ("intermediate_size", 768),
        ("max_position_embeddings", 2048),
        ("mamba_d_ssm", 768),
        ("mamba_n_heads", 24),
        ("mamba_d_state", 64),
        ("mamba_d_conv", 4),
        ("mamba_chunk_size", 128),
        ("mamba_expand", 2),
        ("mamba_n_groups", 1),
        ("semantic_begin_id", 65537),
        ("semantic_end_id", 69632),
        ("eos_token_id", 228),
        ("pad_token_id", 0),
        ("codebook_size", 4096),
        ("num_codebooks", 10),
    ],
)
def test_0p1b_flat_fields_map_onto_hybrid_config(field, expected):
    text_config = Audio8TTSConfig(**CONFIG_0P1B).get_text_config()
    assert getattr(text_config, field) == expected


def test_0p1b_mup_scalars_survive():
    text_config = Audio8TTSConfig(**CONFIG_0P1B).get_text_config()

    # Both are absent from the 0.6b config and would be 1.0 defaults if dropped;
    # the slow-AR stage must apply them (see issue #6495 design note).
    assert text_config.embedding_multiplier == pytest.approx(0.10888671875)
    assert text_config.lm_head_multiplier == pytest.approx(0.078125)


def test_0p1b_rope_theta_uses_new_style_parameters():
    text_config = Audio8TTSConfig(**CONFIG_0P1B).get_text_config()

    assert text_config.rope_parameters["rope_theta"] == pytest.approx(1e11)
    assert text_config.rope_parameters["rope_type"] == "default"


def test_0p1b_fast_ar_config_is_config_driven():
    fast_ar_config = Audio8TTSConfig(**CONFIG_0P1B).fast_ar_config

    assert fast_ar_config.hidden_size == 512
    assert fast_ar_config.num_attention_heads == 8
    assert fast_ar_config.num_key_value_heads == 2
    assert fast_ar_config.num_hidden_layers == 4
    assert fast_ar_config.intermediate_size == 4864


def test_0p6b_backbone_fields_still_select_the_dense_path():
    """Regression guard: no ``slow_backbone`` must keep the Qwen2-shaped class."""
    cfg = Audio8TTSConfig(**CONFIG_0P6B_BACKBONE_FIELDS)
    text_config = cfg.get_text_config()

    assert isinstance(text_config, Audio8TTSSlowARConfig)
    assert not isinstance(text_config, FalconH1Config)
    assert cfg.slow_backbone is None
    assert text_config.hidden_size == 896
    assert text_config.rope_parameters["rope_theta"] == pytest.approx(1e6)


class TestAudio8PipelineResolver:
    def test_routes_by_slow_backbone(self):
        from vllm_omni.model_executor.models.audio8_tts.pipeline import (
            AUDIO8_TTS_HYBRID_PIPELINE,
            AUDIO8_TTS_PIPELINE,
            resolve_audio8_pipeline,
        )

        assert resolve_audio8_pipeline(Audio8TTSConfig(**CONFIG_0P1B)) is AUDIO8_TTS_HYBRID_PIPELINE
        assert resolve_audio8_pipeline(Audio8TTSConfig(**CONFIG_0P6B_BACKBONE_FIELDS)) is AUDIO8_TTS_PIPELINE

    def test_rejects_foreign_and_missing_config(self):
        from transformers import PretrainedConfig

        from vllm_omni.model_executor.models.audio8_tts.pipeline import resolve_audio8_pipeline

        assert resolve_audio8_pipeline(PretrainedConfig()) is None
        assert resolve_audio8_pipeline(None) is None

    def test_serialized_config_round_trips_to_the_hybrid_class(self):
        """Regression guard: ``sub_configs`` must not drag a 0.1b config back to dense."""
        restored = Audio8TTSConfig.from_dict(Audio8TTSConfig(**CONFIG_0P1B).to_dict())

        assert restored.slow_backbone == "falcon_h1"
        assert isinstance(restored.get_text_config(), Audio8TTSHybridSlowARConfig)
