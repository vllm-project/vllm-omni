# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression tests for the generic PEFT rank guard.

The generic PEFT fallback in ``DiffusionLoRAManager._load_adapter`` derives a
single global rank/scale (``peft_helper.r`` / ``lora_alpha``) from
``adapter_config.json`` and applies it to every module without re-checking the
per-module rank of the actual A/B tensors. ``vllm.lora.peft_helper.PEFTHelper``
additionally drops any ``rank_pattern``/``alpha_pattern``. These tests pin the
behavior of ``validate_generic_peft_ranks`` so that mixed-rank / fused /
pattern-bearing adapters are rejected with an actionable error instead of being
silently scaled incorrectly.

All fixtures are tiny synthetic tensors built on CPU; no real adapter is used.
"""

from __future__ import annotations

import json
import types

import pytest
import torch
from vllm.lora.lora_weights import LoRALayerWeights

from vllm_omni.diffusion.lora.utils import (
    read_peft_rank_alpha_patterns,
    validate_generic_peft_ranks,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_IN = 16
_OUT = 32


def _lora(module_name: str, rank: int, *, b_rank: int | None = None) -> LoRALayerWeights:
    """Build a LoRA module with lora_A [rank, in] and lora_B [out, b_rank].

    ``b_rank`` defaults to ``rank`` (a well-formed module). Passing a different
    ``b_rank`` models a corrupt / packed layout whose A and B disagree on rank.
    """
    b_rank = rank if b_rank is None else b_rank
    return LoRALayerWeights(
        module_name=module_name,
        rank=rank,
        lora_alpha=rank,
        lora_a=torch.zeros(rank, _IN),
        lora_b=torch.zeros(_OUT, b_rank),
    )


def _packed(module_name: str, ranks: list[int]) -> types.SimpleNamespace:
    """A PackedLoRALayerWeights-like object exposing list-valued lora_a/lora_b."""
    return types.SimpleNamespace(
        module_name=module_name,
        lora_a=[torch.zeros(r, _IN) for r in ranks],
        lora_b=[torch.zeros(_OUT, r) for r in ranks],
    )


# Test A: homogeneous rank matching the global r -> accepted.
def test_uniform_rank_matching_global_is_accepted():
    loras = {"blocks.0.attn.q": _lora("q", 8), "blocks.0.mlp.fc1": _lora("fc1", 8)}
    # Must not raise.
    validate_generic_peft_ranks(loras, global_rank=8)


# Test B: an explicit rank_pattern / alpha_pattern -> rejected (vLLM cannot honor it).
def test_declared_rank_pattern_is_rejected():
    loras = {"blocks.0.attn.q": _lora("q", 8)}
    with pytest.raises(ValueError, match="rank_pattern"):
        validate_generic_peft_ranks(loras, global_rank=8, rank_pattern={"blocks.0.attn.q": 16})


def test_declared_alpha_pattern_is_rejected():
    loras = {"blocks.0.attn.q": _lora("q", 8)}
    with pytest.raises(ValueError, match="alpha_pattern"):
        validate_generic_peft_ranks(loras, global_rank=8, alpha_pattern={"blocks.0.attn.q": 16})


# Test C: mixed physical rank with no pattern -> deterministic actionable error.
def test_mixed_rank_without_pattern_raises_actionable_error():
    loras = {
        "blocks.0.attn.q": _lora("q", 8),
        "blocks.0.attn.qkv": _lora("qkv", 24),  # fused Q/K/V at 3x rank
        "blocks.0.mlp.fc1": _lora("fc1", 8),
    }
    with pytest.raises(ValueError) as exc:
        validate_generic_peft_ranks(loras, global_rank=8)
    msg = str(exc.value)
    assert "r=8" in msg
    assert "blocks.0.attn.qkv" in msg
    assert "24" in msg
    assert "_load_diffusion_lora_adapter" in msg  # points to the model-specific escape hatch


# Test D: A/B internal rank inconsistency (packed/fused shape) -> rejected, no silent guess.
def test_inconsistent_a_b_rank_raises():
    loras = {"blocks.0.attn.qkv": _lora("qkv", 8, b_rank=24)}
    with pytest.raises(ValueError, match="inconsistent"):
        validate_generic_peft_ranks(loras, global_rank=8)


# Packed weights: per-slice ranks are each validated.
def test_packed_weights_all_slices_matching_is_accepted():
    loras = {"blocks.0.mlp.fc1": _packed("fc1", [8, 8])}
    validate_generic_peft_ranks(loras, global_rank=8)


def test_packed_weights_offrank_slice_is_rejected():
    loras = {"blocks.0.mlp.fc1": _packed("fc1", [8, 16])}
    with pytest.raises(ValueError, match="r=8"):
        validate_generic_peft_ranks(loras, global_rank=8)


# read_peft_rank_alpha_patterns: detects declarations, tolerant of missing config.
def test_read_patterns_from_config(tmp_path):
    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"r": 8, "lora_alpha": 8, "rank_pattern": {"qkv": 24}, "alpha_pattern": {}})
    )
    rank_pattern, alpha_pattern = read_peft_rank_alpha_patterns(str(tmp_path))
    assert rank_pattern == {"qkv": 24}
    assert alpha_pattern == {}


def test_read_patterns_missing_config_is_empty(tmp_path):
    rank_pattern, alpha_pattern = read_peft_rank_alpha_patterns(str(tmp_path))
    assert rank_pattern == {}
    assert alpha_pattern == {}
