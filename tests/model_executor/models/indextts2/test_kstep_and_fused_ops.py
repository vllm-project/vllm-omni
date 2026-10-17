# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for IndexTTS K-step sampling and fused ops integration."""

import torch
import torch.nn as nn

from vllm_omni.model_executor.models.indextts2.indextts2_kstep_sample import (
    IndexTTSKStepResult,
    sample_token_tensor,
    talker_kstep_decode,
)
from vllm_omni.model_executor.models.indextts2.ops.fused_qkv_layer_norm import (
    qkv_head_layer_norm,
)
from vllm_omni.model_executor.models.indextts2.ops.fused_residual_layer_norm import (
    residual_layer_norm,
)


def test_sample_token_tensor_1d_and_2d():
    logits_1d = torch.randn(8194)
    past = torch.tensor([1, 5, 10])
    token_1d = sample_token_tensor(logits_1d, past_tokens=past)
    assert token_1d.numel() == 1
    assert 0 <= token_1d.item() < 8194

    logits_2d = torch.randn(1, 8194)
    token_2d = sample_token_tensor(logits_2d, past_tokens=past)
    assert token_2d.numel() == 1
    assert 0 <= token_2d.item() < 8194


def test_sample_token_tensor_empty_history():
    logits = torch.randn(8194)
    token = sample_token_tensor(logits, past_tokens=None)
    assert token.numel() == 1


class StubTalker(nn.Module):
    def __init__(self, model_dim=64, number_mel_codes=8194):
        super().__init__()
        self.model_dim = model_dim
        self.mel_embedding = nn.Embedding(number_mel_codes, model_dim)
        self.mel_pos_embedding = nn.Embedding(2048, model_dim)
        self.mel_pos_embedding.emb = self.mel_pos_embedding
        self.h = nn.ModuleList([nn.Linear(model_dim, model_dim)])
        self.start_layer = 0
        self.end_layer = 1
        self.final_norm = nn.LayerNorm(model_dim)
        self.ln_f = nn.Identity()
        self.head = nn.Linear(model_dim, number_mel_codes)

    def compute_logits(self, hidden_states):
        return self.head(hidden_states)


def test_talker_kstep_decode_multi_step():
    talker = StubTalker()
    init_hidden = torch.randn(1, 64)
    past_history = torch.tensor([42, 99])

    res = talker_kstep_decode(
        talker=talker,
        initial_hidden_states=init_hidden,
        start_pos=10,
        k_steps=3,
        past_tokens_tensor=past_history,
        repetition_penalty=10.0,
    )

    assert isinstance(res, IndexTTSKStepResult)
    assert len(res.sampled_tokens) >= 1
    assert res.sampled_tensor.numel() == len(past_history) + len(res.sampled_tokens)


def test_talker_kstep_decode_buffer_overflow():
    talker = StubTalker()
    init_hidden = torch.randn(1, 64)
    # Simulate past history filling the buffer up to max_decode_tokens
    max_tokens = 8
    past_history = torch.arange(max_tokens)

    res = talker_kstep_decode(
        talker=talker,
        initial_hidden_states=init_hidden,
        start_pos=10,
        k_steps=3,
        past_tokens_tensor=past_history,
        repetition_penalty=10.0,
        max_decode_tokens=max_tokens,
    )

    assert isinstance(res, IndexTTSKStepResult)
    assert len(res.sampled_tokens) == 3
    # With sliding window, buffer size is capped at max_tokens
    assert res.sampled_tensor.numel() == max_tokens
    # Verify the most recent sampled tokens are at the end of the sliding window
    for i in range(3):
        assert res.sampled_tensor[-3 + i].item() == res.sampled_tokens[i]


def test_fused_residual_layer_norm_cpu_fallback():
    x = torch.randn(2, 16, 64)
    res = torch.randn(2, 16, 64)
    w = torch.ones(64)
    b = torch.zeros(64)
    out = residual_layer_norm(res, x, weight=w, bias=b)
    assert out.shape == (2, 16, 64)


def test_fused_qkv_layer_norm_cpu_fallback():
    qkv = torch.randn(2, 16, 64 * 3)
    kv = torch.empty(2, 4, 16, 2 * 16)
    q_norm = nn.LayerNorm(16)
    k_norm = nn.LayerNorm(16)
    q = qkv_head_layer_norm(qkv, kv, num_heads=4, head_dim=16, q_norm=q_norm, k_norm=k_norm)
    assert q.shape == (2, 4, 16, 16)


if __name__ == "__main__":
    print("Testing test_sample_token_tensor_1d_and_2d...")
    test_sample_token_tensor_1d_and_2d()
    print("Testing test_sample_token_tensor_empty_history...")
    test_sample_token_tensor_empty_history()
    print("Testing test_talker_kstep_decode_multi_step...")
    test_talker_kstep_decode_multi_step()
    print("Testing test_talker_kstep_decode_buffer_overflow...")
    test_talker_kstep_decode_buffer_overflow()
    print("Testing test_fused_residual_layer_norm_cpu_fallback...")
    test_fused_residual_layer_norm_cpu_fallback()
    print("Testing test_fused_qkv_layer_norm_cpu_fallback...")
    test_fused_qkv_layer_norm_cpu_fallback()
    print("ALL UNIT TESTS PASSED SUCCESSFULLY!")
