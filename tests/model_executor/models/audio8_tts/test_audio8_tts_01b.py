# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Audio8 TTS Preview 0.1b (Falcon-H1 Slow AR) integration invariants.

Two checkpoints share ``model_type = "arktts"`` (0.6b Qwen2, 0.1b Falcon-H1).
These CPU tests pin the pieces that route between them -- the backbone-aware
config, the ``slow_backbone`` pipeline resolver, the registry entry, the shared
serving adapter, and the Slow AR weight remap -- without needing a checkpoint.
"""

import pytest
import torch

from vllm_omni.entrypoints.openai.tts_adapters import detect_tts_model_type, resolve_adapter
from vllm_omni.entrypoints.openai.tts_adapters.audio8_tts import Audio8TTSAdapter
from vllm_omni.model_executor.models.audio8_tts.audio8_tts_slow_ar import _remap_audio8_tts_weights
from vllm_omni.model_executor.models.audio8_tts.configuration_audio8_tts import Audio8TTSConfig
from vllm_omni.model_executor.models.audio8_tts.pipeline import (
    AUDIO8_TTS_01B_PIPELINE,
    AUDIO8_TTS_PIPELINE,
    resolve_arktts_pipeline,
)
from vllm_omni.model_executor.models.registry import _OMNI_MODELS

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# The released 0.1b checkpoint's Falcon-H1 backbone fields (config.json).
_01B_FIELDS = dict(
    slow_backbone="falcon_h1",
    vocab_size=69633,
    dim=512,
    n_head=8,
    n_local_heads=2,
    head_dim=64,
    n_layer=24,
    intermediate_size=768,
    max_seq_len=2048,
    rope_base=1e11,
    semantic_begin_id=65537,
    semantic_end_id=69632,
    eos_token_id=228,
    pad_token_id=0,
    fast_dim=512,
    fast_n_head=8,
    fast_n_local_heads=2,
    fast_head_dim=64,
    mamba_d_ssm=768,
    mamba_n_heads=24,
    mamba_d_state=64,
    mamba_n_groups=1,
    mamba_d_conv=4,
    mamba_expand=2,
    embedding_multiplier=0.10888671875,
    lm_head_multiplier=0.078125,
)


def _config_01b() -> Audio8TTSConfig:
    return Audio8TTSConfig(**_01B_FIELDS)


def test_config_01b_builds_falcon_backbone():
    """slow_backbone=falcon_h1 must yield a FalconH1Config whose dims match the
    checkpoint tensors (in_proj = 2*d_ssm + 2*n_groups*d_state + n_heads)."""
    from transformers.models.falcon_h1.configuration_falcon_h1 import FalconH1Config

    tc = _config_01b().get_text_config()
    assert isinstance(tc, FalconH1Config)
    assert (tc.hidden_size, tc.num_hidden_layers, tc.num_attention_heads, tc.num_key_value_heads) == (512, 24, 8, 2)
    assert (tc.mamba_d_ssm, tc.mamba_n_heads, tc.mamba_d_state, tc.mamba_n_groups) == (768, 24, 64, 1)
    # Falcon-H1 mamba in_proj width the loader must line up against.
    assert 2 * tc.mamba_d_ssm + 2 * tc.mamba_n_groups * tc.mamba_d_state + tc.mamba_n_heads == 1688
    # rope_parameters is what vLLM's get_rope reads; must carry the 1e11 base.
    assert tc.rope_parameters == {"rope_type": "default", "rope_theta": 1e11}
    # DualAR fields the Slow AR / Fast AR read off the text config.
    assert (tc.semantic_begin_id, tc.semantic_end_id, tc.num_codebooks, tc.codebook_size) == (65537, 69632, 10, 4096)
    assert tc.embedding_multiplier == pytest.approx(0.10888671875)


def test_config_06b_defaults_unchanged():
    """Without slow_backbone the config must stay the Qwen2-shaped 0.6b view --
    the backbone-aware branch must not regress the existing checkpoint."""
    cfg = Audio8TTSConfig()
    assert cfg.slow_backbone is None
    tc = cfg.get_text_config()
    assert type(tc).__name__ == "Audio8TTSSlowARConfig"
    assert (tc.hidden_size, tc.num_attention_heads, tc.vocab_size) == (896, 14, 155776)
    assert tc.semantic_begin_id == 151678


def test_pipeline_resolver_routes_by_slow_backbone():
    """The arktts resolver picks the Slow AR class by slow_backbone; the codec
    (Stage 1) and its stage key stay shared across both variants."""
    p01 = resolve_arktts_pipeline(_config_01b())
    p06 = resolve_arktts_pipeline(Audio8TTSConfig())
    assert p01 is AUDIO8_TTS_01B_PIPELINE
    assert p06 is AUDIO8_TTS_PIPELINE
    assert p01.model_arch == "Audio8TTS01BSlowARForConditionalGeneration"
    assert p01.stages[0].model_stage == "audio8_tts_01b_slow_ar"
    # 0.1b tokenizer maps <|im_end|> to 228, not the 0.6b's 151645.
    assert p01.stages[0].sampling_constraints["stop_token_ids"] == [228]
    # Both variants decode through the identical codec stage.
    assert p01.stages[1].model_stage == p06.stages[1].model_stage == "audio8_tts_codec_decoder"
    # Missing / non-arktts config falls back to the 0.6b default rather than raising.
    assert resolve_arktts_pipeline(None) is AUDIO8_TTS_PIPELINE


def test_registry_maps_01b_arch_to_falcon_module():
    """The 0.1b Slow AR arch must resolve to its own module (not the 0.6b one)."""
    entry = _OMNI_MODELS["Audio8TTS01BSlowARForConditionalGeneration"]
    assert entry == (
        "audio8_tts",
        "audio8_tts_falcon_slow_ar",
        "Audio8TTS01BSlowARForConditionalGeneration",
    )


def test_adapter_detects_01b_stage_key():
    """The 0.1b stage key must resolve to the shared audio8_tts adapter; the
    codec stage must not (only the entry Slow AR stage identifies a TTS model)."""
    assert "audio8_tts_01b_slow_ar" in Audio8TTSAdapter.stage_keys
    assert detect_tts_model_type("audio8_tts_01b_slow_ar", "Audio8TTS01BSlowARForConditionalGeneration") == "audio8_tts"
    assert resolve_adapter("audio8_tts") is Audio8TTSAdapter
    assert detect_tts_model_type("audio8_tts_codec_decoder", "Audio8TTSCodecDecoder") != "audio8_tts"


def test_fast_ar_remap_splits_wqkv_for_01b_dims():
    """Fast AR remap is shared with the 0.6b: wqkv (768x512 at 0.1b dims) must
    split into q(512) / k(128) / v(128) under the fast_ar.* prefix."""
    fast_dim, kv = 512, 128
    weights = [
        ("fast_layers.0.attention.wqkv.weight", torch.zeros(fast_dim + 2 * kv, fast_dim)),
        ("fast_layers.0.attention.wo.weight", torch.zeros(fast_dim, fast_dim)),
        ("fast_layers.0.feed_forward.w1.weight", torch.zeros(4864, fast_dim)),
        ("fast_embeddings.weight", torch.zeros(4096, fast_dim)),
    ]
    out = dict(_remap_audio8_tts_weights(weights, q_size=0, kv_size=0, fast_q_size=fast_dim, fast_kv_size=kv))
    assert all(name.startswith("fast_ar.") for name in out)
    assert out["fast_ar.layers.0.self_attn.q_proj.weight"].shape == (fast_dim, fast_dim)
    assert out["fast_ar.layers.0.self_attn.k_proj.weight"].shape == (kv, fast_dim)
    assert out["fast_ar.layers.0.self_attn.v_proj.weight"].shape == (kv, fast_dim)
    assert out["fast_ar.layers.0.self_attn.o_proj.weight"].shape == (fast_dim, fast_dim)
    assert out["fast_ar.layers.0.mlp.gate_proj.weight"].shape == (4864, fast_dim)
    assert "fast_ar.fast_embeddings.weight" in out
