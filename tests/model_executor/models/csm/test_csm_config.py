# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU correctness tests for CSM-1B config hoisting + backbone LlamaConfig synthesis.

Covers the two fragile spots:
  * ``CsmConfig`` hoists the nested ``backbone_config`` fields to the top level
    (so vLLM reads them) while keeping depth / codec params nested.
  * ``build_backbone_llama_config`` recovers ``rope_theta`` across the
    transformers >= 5.12 RoPE migration (where ``rope_theta`` is folded INTO
    ``rope_scaling`` and no longer a top-level attribute) and clamps
    ``original_max_position_embeddings`` strictly below ``max_position_embeddings``
    so transformers' rope-parameter validation passes.
"""

import pytest
from transformers import LlamaConfig

from vllm_omni.model_executor.models.csm.configuration_csm import (
    CsmConfig,
    build_backbone_llama_config,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_default_config_hoists_backbone_facts():
    cfg = CsmConfig()
    assert cfg.model_type == "csm"
    assert cfg.hidden_size == 2048
    assert cfg.num_hidden_layers == 16
    assert cfg.num_attention_heads == 32
    assert cfg.num_key_value_heads == 8
    assert cfg.head_dim == 64
    assert cfg.intermediate_size == 8192
    # cb0 logits surface == per-codebook vocab (2051: 2048 codes + 3 reserved).
    assert cfg.vocab_size == 2051
    assert cfg.num_codebooks == 32
    assert cfg.reserved_codebook_ids == (2048, 2049, 2050)
    # vLLM requires this to be absent / None.
    assert cfg.speculative_config is None
    # get_text_config returns self so vLLM reads the hoisted backbone fields.
    assert cfg.get_text_config() is cfg


def test_default_config_keeps_depth_and_codec_facts_nested():
    cfg = CsmConfig()
    assert cfg.depth_hidden_size == 1024
    assert cfg.depth_num_hidden_layers == 4
    assert cfg.depth_head_dim == 128
    assert cfg.depth_num_positions == 33
    assert cfg.codec_sample_rate == 24000
    assert cfg.codec_samples_per_frame == 1920


def test_backbone_llama_config_is_a_real_llama_config():
    llama = build_backbone_llama_config(CsmConfig())
    assert isinstance(llama, LlamaConfig)
    assert llama.hidden_size == 2048
    assert llama.num_hidden_layers == 16
    assert llama.num_attention_heads == 32
    assert llama.num_key_value_heads == 8
    assert llama.head_dim == 64
    assert llama.vocab_size == 2051
    assert llama.tie_word_embeddings is True


def test_rope_theta_falls_back_to_csm_default_when_absent():
    # Default config carries no rope_theta inside rope_scaling -> recover from the
    # top-level attribute / CSM-1B default (500000.0) and fold it back in. Under
    # transformers >= 5.12 LlamaConfig has NO top-level ``rope_theta`` attribute,
    # so vLLM's get_rope reads it from rope_scaling/rope_parameters -- that is
    # exactly where build_backbone_llama_config must leave it.
    llama = build_backbone_llama_config(CsmConfig())
    assert llama.rope_scaling["rope_theta"] == 500000.0
    assert llama.rope_parameters["rope_theta"] == 500000.0
    assert not hasattr(llama, "rope_theta")  # confirms the migration shape


def test_rope_theta_recovered_from_nested_rope_scaling():
    # transformers >= 5.12 shape: rope_theta lives INSIDE rope_scaling, no
    # top-level attribute. build_backbone_llama_config must read it from there
    # and keep it there (not drop it back onto a dead top-level attribute).
    cfg = CsmConfig(
        backbone_config={
            "rope_scaling": {
                "rope_type": "llama3",
                "factor": 32.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8192,
                "rope_theta": 123456.0,
            }
        }
    )
    llama = build_backbone_llama_config(cfg)
    assert llama.rope_scaling["rope_theta"] == 123456.0
    assert llama.rope_parameters["rope_theta"] == 123456.0


def test_original_max_position_clamped_below_max_position():
    # rope-parameter validation requires original_max_position_embeddings <
    # max_position_embeddings (2048). The default 8192 must be clamped to 2047.
    llama = build_backbone_llama_config(CsmConfig())
    assert llama.rope_scaling["original_max_position_embeddings"] == llama.max_position_embeddings - 1
    assert llama.rope_scaling["original_max_position_embeddings"] < llama.max_position_embeddings


def test_explicit_backbone_overrides_are_honored():
    cfg = CsmConfig(backbone_config={"hidden_size": 1536, "num_hidden_layers": 12, "vocab_size": 4096})
    assert cfg.hidden_size == 1536
    assert cfg.num_hidden_layers == 12
    assert cfg.vocab_size == 4096
    llama = build_backbone_llama_config(cfg)
    assert llama.hidden_size == 1536
    assert llama.num_hidden_layers == 12
    assert llama.vocab_size == 4096
