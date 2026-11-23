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
    """Stand-in for the never-sharded prefix layer: piecewise attention on the local rows,
    or the dense 4D mask when the transformer took the mask path (SDPA, padding)."""
    if attn_metadata.full_attn_spans is None:
        return _sdpa(query, key, value, attn_metadata.attn_mask[0, 0])

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


@pytest.mark.parametrize("attn_path", ["piecewise", "mask"])
@pytest.mark.parametrize(
    "prefix_spans",
    [[], [(3, 6)]],
    ids=["text_only_prefix", "prefix_with_condition_image"],
)
def test_ring_prefill_matches_block_causal_attention(prefix_spans, attn_path):
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
    if attn_path == "piecewise":
        metadata = AttentionMetadata(full_attn_spans=[spans], extra={"kv_cache_dtype": None})
    else:
        mask = (causal | same_block).view(1, 1, seq_len, seq_len)
        metadata = AttentionMetadata(attn_mask=mask, extra={"kv_cache_dtype": None})
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


class _RecordingAttention(torch.nn.Module):
    """Stub for the `Attention` layer: records the metadata it is called with."""

    use_ring = True

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, query, key, value, attn_metadata):
        self.calls.append(attn_metadata)
        return torch.zeros_like(query)


def _build_ring_attention(monkeypatch):
    from vllm_omni.diffusion.models.qwen_image_21 import qwen_image_21_transformer as transformer

    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_tp_group", lambda: SimpleNamespace(rank_in_group=0, world_size=1)
    )
    monkeypatch.setattr(transformer, "Attention", lambda **kwargs: _RecordingAttention())
    return transformer.QwenImage21Attention(dim=64, heads=2, dim_head=32)


@pytest.mark.parametrize("attn_path", ["piecewise", "mask"])
def test_forward_routes_ring_prefill_on_both_attention_paths(monkeypatch, attn_path):
    """`forward` must take the local-prefix ring path for spans and for the dense mask fallback."""
    attention = _build_ring_attention(monkeypatch)
    assert attention.prefix_attn is not None
    prefix_len, seq_len = 6, 10
    if attn_path == "piecewise":
        metadata = AttentionMetadata(full_attn_spans=[[(prefix_len, seq_len)]])
    else:
        causal = torch.arange(seq_len).view(-1, 1) >= torch.arange(seq_len).view(1, -1)
        metadata = AttentionMetadata(attn_mask=causal.view(1, 1, seq_len, seq_len))
    freqs = torch.polar(torch.ones(seq_len, 16), torch.zeros(seq_len, 16))

    attention(torch.randn(1, seq_len, 64), freqs, attn_metadata=metadata, sp_prefix_len=prefix_len)

    (prefix_metadata,) = attention.prefix_attn.calls
    (ring_metadata,) = attention.attn.calls
    if attn_path == "piecewise":
        assert prefix_metadata.full_attn_spans == [[]]
    else:
        assert prefix_metadata.attn_mask.shape[-2:] == (prefix_len, prefix_len)
    assert ring_metadata.joint_query is None
    assert ring_metadata.joint_key.shape[1] == prefix_len


def test_forward_keeps_joint_query_without_block_causal_structure(monkeypatch):
    """`causal_block=False` (no spans, no 4D mask): prefix rows attend everything, via the joint path."""
    attention = _build_ring_attention(monkeypatch)
    prefix_len, seq_len = 6, 10
    freqs = torch.polar(torch.ones(seq_len, 16), torch.zeros(seq_len, 16))

    attention(torch.randn(1, seq_len, 64), freqs, attn_metadata=None, sp_prefix_len=prefix_len)

    assert attention.prefix_attn.calls == []
    (ring_metadata,) = attention.attn.calls
    assert ring_metadata.joint_query.shape[1] == prefix_len
