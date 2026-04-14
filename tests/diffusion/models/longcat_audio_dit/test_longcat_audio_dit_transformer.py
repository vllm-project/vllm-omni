# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_cross_attention_passes_cond_mask_to_attention_metadata():
    from vllm_omni.diffusion.models.longcat_audio_dit.longcat_audio_dit_transformer import (
        AudioDiTCrossAttention,
    )

    attn = AudioDiTCrossAttention(q_dim=8, kv_dim=8, heads=2, dim_head=4, bias=True)

    captured = {}

    def fake_attn(query, key, value, attn_metadata=None):
        captured["attn_metadata"] = attn_metadata
        return torch.zeros_like(query)

    attn.attn.forward = fake_attn

    x = torch.randn(2, 3, 8)
    cond = torch.randn(2, 5, 8)
    cond_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)

    attn(x, cond, cond_mask=cond_mask)

    assert captured["attn_metadata"] is not None
    assert torch.equal(captured["attn_metadata"].attn_mask, cond_mask)


def test_approx_batch_duration_uses_longest_prompt():
    from vllm_omni.diffusion.models.longcat_audio_dit.pipeline_longcat_audio_dit import (
        _approx_batch_duration_from_prompts,
        _approx_duration_from_text,
    )

    prompts = ["short", "this is a much longer prompt"]

    assert _approx_batch_duration_from_prompts(prompts) == max(_approx_duration_from_text(p) for p in prompts)


def test_approx_batch_duration_empty_prompts_returns_zero():
    from vllm_omni.diffusion.models.longcat_audio_dit.pipeline_longcat_audio_dit import (
        _approx_batch_duration_from_prompts,
    )

    assert _approx_batch_duration_from_prompts([]) == 0.0
