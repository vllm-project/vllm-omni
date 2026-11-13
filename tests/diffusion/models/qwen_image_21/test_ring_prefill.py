# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ring SP prefill of Qwen-Image-2.1 is block-causal attention over [prefix, target]."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.utils.piecewise_attn import piecewise_attn
from vllm_omni.diffusion.models.qwen_image_21.qwen_image_21_transformer import QwenImage21Attention

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _sdpa(query, key, value, mask):
    """(B, S, H, D) attention with a boolean (Sq, Skv) mask."""
    q, k, v = (t.transpose(1, 2) for t in (query, key, value))
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask).transpose(1, 2)


def _piecewise_layer(query, key, value, attn_metadata):
    """Stand-in for the never-sharded prefix layer: piecewise attention on the local rows."""

    def segment(q, k, v, causal, softmax_scale):
        # Varlen-flash convention: a causal query segment is aligned to the end of its K/V.
        offset = k.shape[1] - q.shape[1]
        mask = torch.arange(k.shape[1]).view(1, -1) <= offset + torch.arange(q.shape[1]).view(-1, 1)
        return _sdpa(q, k, v, mask if causal else None)

    return piecewise_attn(query, key, value, attn_metadata.full_attn_spans, query.shape[-1] ** -0.5, segment)


def _ring_layer(query, key, value, attn_metadata):
    """Stand-in for ring attention on one rank holding every target shard: full attention over [joint K/V, K/V]."""
    assert attn_metadata.joint_query is None and attn_metadata.attn_mask is None
    key = torch.cat([attn_metadata.joint_key, key], dim=1)
    value = torch.cat([attn_metadata.joint_value, value], dim=1)
    return _sdpa(query, key, value, None)


@pytest.mark.parametrize(
    "prefix_spans",
    [[], [(3, 6)]],
    ids=["text_only_prefix", "prefix_with_condition_image"],
)
def test_ring_prefill_matches_block_causal_attention(prefix_spans):
    torch.manual_seed(0)
    prefix_len, target_len, heads, head_dim = 8, 8, 2, 16
    seq_len = prefix_len + target_len
    query, key, value = (torch.randn(1, seq_len, heads, head_dim) for _ in range(3))
    spans = [*prefix_spans, (prefix_len, seq_len)]  # the target image is the last full block

    # Reference: (q_idx >= kv_idx) or same image block, over the whole sequence.
    block_ids = torch.full((seq_len,), -1)
    for block, (start, end) in enumerate(spans):
        block_ids[start:end] = block
    same_block = (block_ids.view(-1, 1) == block_ids.view(1, -1)) & (block_ids.view(-1, 1) >= 0)
    causal = torch.arange(seq_len).view(-1, 1) >= torch.arange(seq_len).view(1, -1)
    reference = _sdpa(query, key, value, causal | same_block)

    fake_self = SimpleNamespace(prefix_attn=_piecewise_layer, attn=_ring_layer)
    metadata = AttentionMetadata(full_attn_spans=[spans], extra={"kv_cache_dtype": None})
    output = QwenImage21Attention._ring_prefill(fake_self, query, key, value, metadata, prefix_len)

    torch.testing.assert_close(output, reference, atol=1e-5, rtol=1e-5)


def test_ring_prefill_keeps_prefix_rows_off_the_target():
    """The bug this replaces: prefix rows that also attend to the target change the cached prefix K/V."""
    torch.manual_seed(0)
    prefix_len, seq_len = 8, 16
    query, key, value = (torch.randn(1, seq_len, 2, 16) for _ in range(3))
    metadata = AttentionMetadata(full_attn_spans=[[(prefix_len, seq_len)]])
    fake_self = SimpleNamespace(prefix_attn=_piecewise_layer, attn=_ring_layer)

    output = QwenImage21Attention._ring_prefill(fake_self, query, key, value, metadata, prefix_len)
    value_changed = value.clone()
    value_changed[:, prefix_len:] += 1.0  # perturb only the target tokens
    output_changed = QwenImage21Attention._ring_prefill(fake_self, query, key, value_changed, metadata, prefix_len)

    torch.testing.assert_close(output[:, :prefix_len], output_changed[:, :prefix_len])
    assert not torch.allclose(output[:, prefix_len:], output_changed[:, prefix_len:])
