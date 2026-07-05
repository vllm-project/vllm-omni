# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from vllm.model_executor.models.qwen3_asr import Qwen3ASRForConditionalGeneration

from vllm_omni.model_executor.models.aura_omni.qwen3_asr import (
    AuraQwen3ASRForConditionalGeneration,
)


def _bind_aura_asr_helpers(model: AuraQwen3ASRForConditionalGeneration) -> None:
    model._resolve_eos_token_id = AuraQwen3ASRForConditionalGeneration._resolve_eos_token_id.__get__(
        model, AuraQwen3ASRForConditionalGeneration
    )
    model._noop_row_mask = AuraQwen3ASRForConditionalGeneration._noop_row_mask.__get__(
        model, AuraQwen3ASRForConditionalGeneration
    )


def test_aura_qwen3_asr_noop_forces_eos_on_decode(monkeypatch):
    model = AuraQwen3ASRForConditionalGeneration.__new__(AuraQwen3ASRForConditionalGeneration)
    model._runtime_info = [{"omni_skip_stages": [0]}]
    model._cached_eos_token_id = 2
    _bind_aura_asr_helpers(model)

    base_logits = torch.zeros(1, 4)
    monkeypatch.setattr(
        Qwen3ASRForConditionalGeneration,
        "compute_logits",
        lambda _self, _hidden: base_logits.clone(),
    )

    logits = AuraQwen3ASRForConditionalGeneration.compute_logits(model, torch.zeros(1, 8))
    assert logits is not None
    assert torch.isneginf(logits[0, 0])
    assert torch.isneginf(logits[0, 1])
    assert logits[0, 2] == 0.0
    assert torch.isneginf(logits[0, 3])


def test_aura_qwen3_asr_noop_skips_prefill_rows(monkeypatch):
    model = AuraQwen3ASRForConditionalGeneration.__new__(AuraQwen3ASRForConditionalGeneration)
    model._runtime_info = [{"omni_skip_stages": [0]}]
    _bind_aura_asr_helpers(model)

    base_logits = torch.ones(8, 4)
    monkeypatch.setattr(
        Qwen3ASRForConditionalGeneration,
        "compute_logits",
        lambda _self, _hidden: base_logits.clone(),
    )

    logits = AuraQwen3ASRForConditionalGeneration.compute_logits(model, torch.zeros(8, 8))
    assert logits is not None
    assert torch.all(logits == 1.0)
