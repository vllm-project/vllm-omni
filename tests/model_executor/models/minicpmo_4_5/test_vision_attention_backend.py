# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MiniCPM-o 4.5 vision attention backend selection."""

from types import SimpleNamespace

import pytest
import torch
from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import (
    SiglipEncoderLayer,
    SiglipVisionTransformer,
    _resolve_vision_attention_implementation,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_siglip_declares_transformers_5_flash_attention_support() -> None:
    assert SiglipVisionTransformer._supports_flash_attn is True


def _vision_config(attention_implementation: str) -> SimpleNamespace:
    return SimpleNamespace(
        _attn_implementation=attention_implementation,
        attention_dropout=0.0,
        hidden_act="gelu",
        hidden_size=32,
        intermediate_size=64,
        layer_norm_eps=1e-6,
        num_attention_heads=4,
    )


def test_siglip_declares_sdpa_support() -> None:
    assert SiglipVisionTransformer._supports_sdpa is True


def test_siglip_encoder_layer_supports_regional_compile() -> None:
    assert TorchCompileWithNoGuardsWrapper in SiglipEncoderLayer.__bases__


def test_siglip_encoder_selects_sdpa_attention() -> None:
    layer = SiglipEncoderLayer(_vision_config("sdpa"))

    assert type(layer.self_attn).__name__ == "SiglipSdpaAttention"


def test_siglip_sdpa_matches_eager_attention() -> None:
    torch.manual_seed(7)
    eager = SiglipEncoderLayer(_vision_config("eager")).eval()
    sdpa = SiglipEncoderLayer(_vision_config("sdpa")).eval()
    sdpa.load_state_dict(eager.state_dict())
    hidden_states = torch.randn(2, 5, 32)
    attention_mask = torch.zeros(2, 1, 5, 5)
    attention_mask[1, :, :, -1] = torch.finfo(torch.float32).min

    eager_output = eager(hidden_states, attention_mask)[0]
    sdpa_output = sdpa(hidden_states, attention_mask)[0]

    torch.testing.assert_close(sdpa_output, eager_output, rtol=1e-5, atol=1e-6)


def test_explicit_vision_flash_attention_override_is_preserved() -> None:
    vision_config = SimpleNamespace(_attn_implementation="flash_attention_2")
    model_config = SimpleNamespace(_attn_implementation="eager")

    assert _resolve_vision_attention_implementation(vision_config, model_config) == "flash_attention_2"


def test_legacy_model_attention_override_is_used_without_vision_override() -> None:
    vision_config = SimpleNamespace(_attn_implementation="eager")
    model_config = SimpleNamespace(_attn_implementation="flash_attention_2")

    assert _resolve_vision_attention_implementation(vision_config, model_config) == "flash_attention_2"


def test_vision_attention_defaults_to_eager() -> None:
    assert _resolve_vision_attention_implementation(SimpleNamespace(), SimpleNamespace()) == "eager"
