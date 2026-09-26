# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for OmniVoice CUDA Graph generator wrapper numerical equivalence.

Verifies that _OmniVoiceCUDAGraphForward produces results equivalent to
eager mode across three scenarios:
  - Exact-size inputs (no padding) → bit-identical
  - Padded inputs (padded to nearest bucket) → correct slicing and exact match
  - Oversized inputs (128-aligned lazy capture) → bit-identical

Uses a small randomly initialized OmniVoiceGenerator so every test exercises
the production embedding, transformer, varlen-attention, and logits paths
without loading checkpoint weights.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from vllm.utils.math_utils import round_up

from vllm_omni.model_executor.models.omnivoice.omnivoice_generator import (
    OmniVoiceGenerator,
    _OmniVoiceCUDAGraphForward,
)
from vllm_omni.transformers_utils.configs.omnivoice import OmniVoiceConfig

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

DEVICE = torch.device("cuda:0")
NUM_CB = 8
VOCAB = 1025
CAPTURE_SIZES = [32, 64, 128]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _eager(gen: OmniVoiceGenerator, ids: torch.Tensor, mask: torch.Tensor, cu_seqs) -> torch.Tensor:
    """Run the production varlen step outside CUDA Graph."""
    seq_len = ids.shape[0]
    rope_table = gen._rope_table_for(seq_len, ids.device, gen.model_dtype)
    return gen._step_forward(ids, mask, cu_seqs, rope_table)


def _bucketed_eager(gen, wrapper, ids, mask, cu_seqs, batch_size) -> torch.Tensor:
    """Run eager with the exact padded shape and metadata used by Graph."""
    seq_len = ids.shape[0]
    bucket = wrapper._find_bucket(batch_size, seq_len)
    if bucket is None:
        bucket = round_up(seq_len, wrapper._LAZY_CAPTURE_ALIGNMENT)
    padded_ids, padded_mask = wrapper._pad_inputs(ids, mask, bucket)
    padded_cu_seqs = cu_seqs.clone()
    padded_cu_seqs[-1] = bucket
    return _eager(gen, padded_ids, padded_mask, padded_cu_seqs)[:, :seq_len, :]


def _make_inputs(seq_len: int, device: torch.device = DEVICE):
    ids = torch.randint(0, 100, (seq_len, NUM_CB), dtype=torch.long, device=device)
    mask = torch.ones(seq_len, dtype=torch.bool, device=device)
    cond_len = (seq_len + 1) // 2
    uncond_len = seq_len - cond_len

    cu_seqs = torch.tensor(
        [0, cond_len, cond_len + uncond_len, cond_len + uncond_len],
        dtype=torch.int32,
        device=device,
    )
    return ids, mask, cu_seqs


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gen():
    torch.manual_seed(42)
    config = OmniVoiceConfig(
        audio_vocab_size=VOCAB,
        num_audio_codebook=NUM_CB,
        llm_config={
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 128,
            "vocab_size": 128,
            "max_position_embeddings": 4096,
            "head_dim": 16,
        },
        enable_cuda_graph=False,
    )
    return OmniVoiceGenerator(config, SimpleNamespace(max_num_seqs=2)).to(DEVICE).eval()


@pytest.fixture(scope="module")
def wrapper(gen):
    w = _OmniVoiceCUDAGraphForward(gen, capture_sizes=CAPTURE_SIZES)
    w.warmup(DEVICE)
    return w


# ---------------------------------------------------------------------------
# 1. Exact-size inputs → bit-identical
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seq_len", CAPTURE_SIZES)
def test_exact_size_bit_identical(gen, wrapper, seq_len):
    """When input exactly matches a captured bucket, output must be bit-identical to eager."""
    ids, mask, cu_seqs = _make_inputs(seq_len)
    with torch.no_grad():
        eager_out = _eager(gen, ids, mask, cu_seqs)
        graph_out = wrapper(ids, mask, cu_seqs, 1)
    torch.testing.assert_close(graph_out, eager_out, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# 2. Padded inputs → correct shape, values match at actual positions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seq_len", [1, 15, 33, 60, 100])
def test_padded_output_shape(wrapper, seq_len):
    """Graph output must be sliced back to actual seq_len, not the bucket size."""
    ids, mask, cu_seqs = _make_inputs(seq_len)
    with torch.no_grad():
        graph_out = wrapper(ids, mask, cu_seqs, 1)
    assert graph_out.shape == (NUM_CB, seq_len, VOCAB)


@pytest.mark.parametrize("seq_len", [15, 33, 60, 100])
def test_padded_output_matches_eager(gen, wrapper, seq_len):
    """Padded graph output must equal eager output at actual positions."""
    ids, mask, cu_seqs = _make_inputs(seq_len)
    with torch.no_grad():
        eager_out = _bucketed_eager(gen, wrapper, ids, mask, cu_seqs, 1)
        graph_out = wrapper(ids, mask, cu_seqs, 1)
    torch.testing.assert_close(graph_out, eager_out, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# 3. Oversized inputs → aligned lazy capture, bit-identical
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seq_len", [129, 200, 256])
def test_aligned_lazy_capture_bit_identical(gen, wrapper, seq_len):
    """Sequences beyond the static plan use aligned lazy capture and remain bit-identical."""
    ids, mask, cu_seqs = _make_inputs(seq_len)
    with torch.no_grad():
        eager_out = _bucketed_eager(gen, wrapper, ids, mask, cu_seqs, 1)
        graph_out = wrapper(ids, mask, cu_seqs, 1)
    torch.testing.assert_close(graph_out, eager_out, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# 4. Determinism
# ---------------------------------------------------------------------------


def test_deterministic_across_calls(wrapper):
    """Same input must produce identical output on repeated CUDA graph replays."""
    ids, mask, cu_seqs = _make_inputs(32)
    with torch.no_grad():
        out1 = wrapper(ids, mask, cu_seqs, 1).clone()
        out2 = wrapper(ids, mask, cu_seqs, 1).clone()
    torch.testing.assert_close(out1, out2, atol=0, rtol=0)


def test_replay_uses_updated_cu_seqs(gen, wrapper):
    """The same graph key must honor new sequence boundaries on replay."""
    ids, mask, _ = _make_inputs(48)
    cu_seqs_a = torch.tensor([0, 32, 48, 48], dtype=torch.int32, device=DEVICE)
    cu_seqs_b = torch.tensor([0, 17, 48, 48], dtype=torch.int32, device=DEVICE)

    with torch.no_grad():
        eager_a = _bucketed_eager(gen, wrapper, ids, mask, cu_seqs_a, 1)
        graph_a = wrapper(ids, mask, cu_seqs_a, 1).clone()
        eager_b = _bucketed_eager(gen, wrapper, ids, mask, cu_seqs_b, 1)
        graph_b = wrapper(ids, mask, cu_seqs_b, 1).clone()

    torch.testing.assert_close(graph_a, eager_a, atol=0, rtol=0)
    torch.testing.assert_close(graph_b, eager_b, atol=0, rtol=0)
    assert not torch.equal(graph_a, graph_b)


def test_batch_two_cu_seqs_matches_eager(gen, wrapper):
    """B=2 uses four real sequence segments plus the fixed tail slot."""
    ids, mask, _ = _make_inputs(48)
    cu_seqs = torch.tensor([0, 10, 18, 32, 48, 48], dtype=torch.int32, device=DEVICE)

    assert cu_seqs.numel() == 2 * 2 + 2
    with torch.no_grad():
        eager_out = _bucketed_eager(gen, wrapper, ids, mask, cu_seqs, 2)
        graph_out = wrapper(ids, mask, cu_seqs, 2).clone()

    torch.testing.assert_close(graph_out, eager_out, atol=0, rtol=0)


def test_lazy_graph_cache_uses_lru_eviction(wrapper):
    """A lazy-cache hit must protect that graph from the next eviction."""
    original_limit = wrapper._MAX_LAZY_GRAPHS
    wrapper._lazy_graphs.clear()
    wrapper._MAX_LAZY_GRAPHS = 2
    try:
        # Static B=1 coverage ends at 128. These lengths round to three
        # distinct 128-aligned lazy keys: 256, 384, and 512.
        for seq_len in (129, 257):
            ids, mask, cu_seqs = _make_inputs(seq_len)
            with torch.no_grad():
                wrapper(ids, mask, cu_seqs, 1)
        assert list(wrapper._lazy_graphs) == [(1, 256), (1, 384)]

        # Refresh key 256, making key 384 the least recently used entry.
        ids, mask, cu_seqs = _make_inputs(129)
        with torch.no_grad():
            wrapper(ids, mask, cu_seqs, 1)
        assert list(wrapper._lazy_graphs) == [(1, 384), (1, 256)]

        # Inserting key 512 must evict key 384, not the recently hit key 256.
        ids, mask, cu_seqs = _make_inputs(385)
        with torch.no_grad():
            wrapper(ids, mask, cu_seqs, 1)
        assert list(wrapper._lazy_graphs) == [(1, 256), (1, 512)]
    finally:
        wrapper._MAX_LAZY_GRAPHS = original_limit
        wrapper._lazy_graphs.clear()


# ---------------------------------------------------------------------------
# 5. _find_bucket logic (CPU, no CUDA graph)
# ---------------------------------------------------------------------------


def test_find_bucket_returns_nearest_bucket():
    """_find_bucket must return the smallest bucket >= seq_len, or None if all are smaller."""
    from vllm_omni.model_executor.models.omnivoice.omnivoice_generator import (
        _OmniVoiceCUDAGraphForward,
    )

    w = _OmniVoiceCUDAGraphForward.__new__(_OmniVoiceCUDAGraphForward)
    w.capture_bucket_sizes_by_batch = {
        1: [32, 64, 128],
        2: [64, 128, 256],
    }
    w._graphs = {}

    assert w._find_bucket(1, 1) == 32
    assert w._find_bucket(1, 32) == 32
    assert w._find_bucket(1, 33) == 64
    assert w._find_bucket(1, 128) == 128
    assert w._find_bucket(1, 129) is None
    assert w._find_bucket(2, 33) == 64
    assert w._find_bucket(2, 129) == 256
    assert w._find_bucket(3, 64) is None
