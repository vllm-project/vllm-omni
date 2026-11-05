# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for IndexTTS K-step sampling and fused ops integration."""

import pytest
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

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


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


class ScriptedTalker(StubTalker):
    """StubTalker that emits a scripted sequence of token IDs across steps."""

    def __init__(self, scripted_ids: list[int], model_dim: int = 64, number_mel_codes: int = 8194):
        super().__init__(model_dim=model_dim, number_mel_codes=number_mel_codes)
        self.scripted_ids = list(scripted_ids)
        self.call_idx = 0

    def compute_logits(self, hidden_states):
        logits = torch.full((1, self.head.out_features), -1e9, device=hidden_states.device)
        tok = self.scripted_ids[min(self.call_idx, len(self.scripted_ids) - 1)]
        self.call_idx += 1
        logits[0, tok] = 1e4
        return logits


def test_talker_kstep_decode_early_stop_truncates_tokens_and_latents():
    """Stop at step 1 (0-indexed) of k_steps=4 with use_latent=True must truncate
    sampled_tokens, accumulated_latents, and sampled_tensor at the stop token.
    """
    stop_id = 8193
    talker = ScriptedTalker([101, stop_id, 202, 303])
    init_hidden = torch.randn(1, 64)
    past_history = torch.tensor([10, 20, 30])

    res = talker_kstep_decode(
        talker=talker,
        initial_hidden_states=init_hidden,
        start_pos=3,
        k_steps=4,
        top_k=1,
        repetition_penalty=1.0,
        stop_token_ids=(stop_id,),
        use_latent=True,
        past_tokens_tensor=past_history,
        max_decode_tokens=16,
    )

    assert res.finished is True
    assert res.sampled_tokens == [101, stop_id]
    assert len(res.accumulated_latents) == 2
    assert res.sampled_tensor.tolist() == [10, 20, 30, 101, stop_id]


def test_talker_kstep_decode_early_stop_on_full_buffer_preserves_history():
    """When past_tokens_tensor fills max_decode_tokens, post-stop steps must not
    overwrite valid history slots or leak discarded tokens into sampled_tensor.
    """
    stop_id = 8193
    max_tokens = 6
    past_history = torch.tensor([11, 12, 13, 14, 15, 16])
    talker = ScriptedTalker([101, stop_id, 202, 303])
    init_hidden = torch.randn(1, 64)

    res = talker_kstep_decode(
        talker=talker,
        initial_hidden_states=init_hidden,
        start_pos=6,
        k_steps=4,
        top_k=1,
        repetition_penalty=1.0,
        stop_token_ids=(stop_id,),
        use_latent=True,
        past_tokens_tensor=past_history,
        max_decode_tokens=max_tokens,
    )

    assert res.finished is True
    assert res.sampled_tokens == [101, stop_id]
    assert len(res.accumulated_latents) == 2
    # Last 6 valid tokens: [13, 14, 15, 16, 101, 8193] — no post-stop 202/303 leaked!
    assert res.sampled_tensor.tolist() == [13, 14, 15, 16, 101, stop_id]


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
