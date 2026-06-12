# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Alpamayo config loading + registration."""

from __future__ import annotations

import os

import pytest
from transformers import AutoConfig

# Importing this module triggers AutoConfig.register side-effects.
import vllm_omni.transformers_utils.configs.alpamayo  # noqa: F401
from vllm_omni.model_executor.models.alpamayo.configuration_alpamayo import (
    Alpamayo15Config,
    AlpamayoR1Config,
)

_MODEL_15 = "/data/models/Alpamayo-1.5-10B"
_VLM_BASE = "/data/models/Qwen3-VL-8B-Instruct"

_requires_weights = pytest.mark.skipif(
    not (os.path.isdir(_MODEL_15) and os.path.isdir(_VLM_BASE)),
    reason="Alpamayo-1.5 / Qwen3-VL base weights not present locally",
)


def test_model_types_registered():
    assert Alpamayo15Config.model_type == "alpamayo1_5"
    assert AlpamayoR1Config.model_type == "alpamayo_r1"


@_requires_weights
def test_load_flat_15_config_materializes_subconfigs():
    cfg = AutoConfig.from_pretrained(_MODEL_15, trust_remote_code=True)
    assert isinstance(cfg, Alpamayo15Config)

    # Base Qwen3-VL sub-configs must be materialized from the flat config.
    assert cfg.text_config is not None
    assert cfg.vision_config is not None
    assert hasattr(cfg.text_config, "num_attention_heads")
    assert cfg.text_config.num_attention_heads > 0

    # Alpamayo-specific fields preserved.
    assert cfg.expert_cfg["hidden_size"] == 2048
    assert cfg.traj_tokenizer_cfg["action_space_cfg"]["n_waypoints"] == 64
    assert cfg.traj_vocab_size == 4000
    assert cfg.traj_token_start_idx == 151669
    assert cfg.traj_token_ids["future_start"] == 155681

    # Extended vocab propagated into text_config so embeddings size correctly.
    assert cfg.text_config.vocab_size == cfg.vocab_size == 155697


@_requires_weights
def test_15_config_roundtrip_serialization():
    cfg = AutoConfig.from_pretrained(_MODEL_15, trust_remote_code=True)
    d = cfg.to_dict()
    # Round-trip: sub-configs now present, so no base re-fetch needed.
    cfg2 = Alpamayo15Config.from_dict(d)
    assert cfg2.text_config.vocab_size == cfg.text_config.vocab_size
    assert cfg2.expert_cfg["hidden_size"] == cfg.expert_cfg["hidden_size"]
    assert cfg2.traj_token_ids == cfg.traj_token_ids
