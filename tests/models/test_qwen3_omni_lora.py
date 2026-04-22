# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Qwen3-Omni Thinker LoRA support declarations."""

from __future__ import annotations

import pytest
from vllm.model_executor.models.interfaces import supports_lora

from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
    Qwen3OmniMoeThinkerForConditionalGeneration,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TestQwen3OmniThinkerLoRADeclaration:
    """Verify Qwen3-Omni Thinker declares LoRA support correctly."""

    def test_supports_lora_interface(self):
        """Thinker class must be recognized as SupportsLoRA."""
        assert supports_lora(Qwen3OmniMoeThinkerForConditionalGeneration)

    def test_packed_modules_mapping_has_qkv(self):
        """Language model's fused QKV must be declared."""
        mapping = Qwen3OmniMoeThinkerForConditionalGeneration.packed_modules_mapping
        assert "qkv_proj" in mapping
        assert mapping["qkv_proj"] == ["q_proj", "k_proj", "v_proj"]

    def test_packed_modules_mapping_has_attn_qkv(self):
        """Vision encoder's fused QKV must be declared for LoRA compatibility."""
        mapping = Qwen3OmniMoeThinkerForConditionalGeneration.packed_modules_mapping
        assert "attn.qkv" in mapping
        assert mapping["attn.qkv"] == ["attn.q", "attn.k", "attn.v"]

    def test_packed_modules_mapping_has_gate_up(self):
        """MoE MLP's fused gate+up projection must be declared."""
        mapping = Qwen3OmniMoeThinkerForConditionalGeneration.packed_modules_mapping
        assert "gate_up_proj" in mapping
        assert mapping["gate_up_proj"] == ["gate_proj", "up_proj"]
