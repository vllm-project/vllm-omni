# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for OmniVoice CUDA Graph generator wrapper numerical equivalence.

Verifies that _OmniVoiceCUDAGraphForward produces results equivalent to
eager mode across three scenarios:
  - Exact-size inputs (no padding) → bit-identical
  - Padded inputs (padded to nearest bucket) → correct slicing, exact match
    for position-independent models
  - Oversized inputs (fallback to eager) → bit-identical

Uses a lightweight synthetic generator to keep warmup fast and avoid
loading actual model weights.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

DEVICE = torch.device("cuda:0")
NUM_CB = 8
VOCAB = 1025
HIDDEN = 64
CAPTURE_SIZES = [32, 64, 128]


# ---------------------------------------------------------------------------
# Synthetic generator: matches _OmniVoiceCUDAGraphForward's interface
# ---------------------------------------------------------------------------


class _SyntheticGenerator:
    """Position-independent Embedding+Linear replacing OmniVoiceGenerator's 28-layer transformer.

    Each token position is processed independently (no attention), so padded
    runs produce bit-identical results to non-padded runs on the same positions.
    This makes the padding/slicing logic easy to verify exactly.
    """

    class _Cfg:
        num_audio_codebook = NUM_CB

    def __init__(self, device: torch.device):
        self.config = self._Cfg()
        self.text_embedding = nn.Embedding(1000, HIDDEN).to(device).eval()
        self._linear = nn.Linear(HIDDEN, NUM_CB * VOCAB, bias=False).to(device).eval()
        self._rope_cos: torch.Tensor | None = None
        self._rope_sin: torch.Tensor | None = None

    def _ensure_rope(self, seq_len: int, device: torch.device) -> None:
        if self._rope_cos is None or self._rope_cos.shape[0] < seq_len:
            max_len = max(seq_len, 512)
            self._rope_cos = torch.zeros(max_len, 64, device=device)
            self._rope_sin = torch.zeros(max_len, 64, device=device)

    def _step_forward(
        self,
        input_ids: torch.Tensor,
        audio_mask: torch.Tensor,
        attention_mask: torch.Tensor | None,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        two_b, _, S = input_ids.shape
        x = self.text_embedding(input_ids[:, 0, :].clamp(0, 999))  # [two_b, S, H]
        logits = self._linear(x)  # [two_b, S, 8*1025]
        return logits.view(two_b, S, NUM_CB, VOCAB).permute(0, 2, 1, 3)  # [two_b, 8, S, 1025]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _eager(gen: _SyntheticGenerator, ids: torch.Tensor, mask: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
    """Run _step_forward with zero RoPE tensors (synthetic gen ignores them)."""
    S = ids.shape[-1]
    cos = torch.zeros(S, 64, device=ids.device)
    sin = torch.zeros(S, 64, device=ids.device)
    return gen._step_forward(ids, mask, attn, cos, sin)


def _make_inputs(seq_len: int, device: torch.device = DEVICE):
    two_b = 2
    ids = torch.randint(0, 100, (two_b, NUM_CB, seq_len), dtype=torch.long, device=device)
    mask = torch.ones(two_b, seq_len, dtype=torch.bool, device=device)
    attn = torch.ones(two_b, 1, seq_len, seq_len, dtype=torch.bool, device=device)
    return ids, mask, attn


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gen():
    torch.manual_seed(42)
    return _SyntheticGenerator(DEVICE)


@pytest.fixture(scope="module")
def wrapper(gen):
    from vllm_omni.model_executor.models.omnivoice.omnivoice_generator import (
        _OmniVoiceCUDAGraphForward,
    )

    w = _OmniVoiceCUDAGraphForward(gen, capture_sizes=CAPTURE_SIZES)
    w.warmup(DEVICE)
    return w


# ---------------------------------------------------------------------------
# 1. Exact-size inputs → bit-identical
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seq_len", CAPTURE_SIZES)
def test_exact_size_bit_identical(gen, wrapper, seq_len):
    """When input exactly matches a captured bucket, output must be bit-identical to eager."""
    ids, mask, attn = _make_inputs(seq_len)
    with torch.no_grad():
        eager_out = _eager(gen, ids, mask, attn)
        graph_out = wrapper(ids, mask, attn)
    torch.testing.assert_close(graph_out, eager_out, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# 2. Padded inputs → correct shape, values match at actual positions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seq_len", [1, 15, 33, 60, 100])
def test_padded_output_shape(gen, wrapper, seq_len):
    """Graph output must be sliced back to actual seq_len, not the bucket size."""
    ids, mask, attn = _make_inputs(seq_len)
    with torch.no_grad():
        graph_out = wrapper(ids, mask, attn)
    assert graph_out.shape == (2, NUM_CB, seq_len, VOCAB)


@pytest.mark.parametrize("seq_len", [15, 33, 60, 100])
def test_padded_output_matches_eager(gen, wrapper, seq_len):
    """Padded graph output must equal eager output at actual positions.

    The synthetic model has no attention across positions, so zero-padding
    does not affect non-padded positions — exact match is expected.
    """
    ids, mask, attn = _make_inputs(seq_len)
    with torch.no_grad():
        eager_out = _eager(gen, ids, mask, attn)
        graph_out = wrapper(ids, mask, attn)
    torch.testing.assert_close(graph_out, eager_out, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# 3. Oversized inputs → fallback to eager (lazy capture), bit-identical
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seq_len", [129, 200, 256])
def test_fallback_eager_bit_identical(gen, wrapper, seq_len):
    """Sequences exceeding the largest bucket fall back to lazy capture → bit-identical."""
    ids, mask, attn = _make_inputs(seq_len)
    with torch.no_grad():
        eager_out = _eager(gen, ids, mask, attn)
        graph_out = wrapper(ids, mask, attn)
    torch.testing.assert_close(graph_out, eager_out, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# 4. Determinism
# ---------------------------------------------------------------------------


def test_deterministic_across_calls(wrapper):
    """Same input must produce identical output on repeated CUDA graph replays."""
    ids, mask, attn = _make_inputs(32)
    with torch.no_grad():
        out1 = wrapper(ids, mask, attn)
        out2 = wrapper(ids, mask, attn)
    torch.testing.assert_close(out1, out2, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# 5. _find_bucket logic (CPU, no CUDA graph)
# ---------------------------------------------------------------------------


def test_find_bucket_returns_nearest_bucket():
    """_find_bucket must return the smallest bucket >= seq_len, or None if all are smaller."""
    from vllm_omni.model_executor.models.omnivoice.omnivoice_generator import (
        _OmniVoiceCUDAGraphForward,
    )

    w = _OmniVoiceCUDAGraphForward.__new__(_OmniVoiceCUDAGraphForward)
    w._capture_sizes = [32, 64, 128]
    w._graphs = {}

    assert w._find_bucket(1) == 32
    assert w._find_bucket(32) == 32
    assert w._find_bucket(33) == 64
    assert w._find_bucket(64) == 64
    assert w._find_bucket(128) == 128
    assert w._find_bucket(129) is None
    assert w._find_bucket(1000) is None


# ---------------------------------------------------------------------------
# 6. enable_cuda_graph=False produces same tokens as enable_cuda_graph=True
# ---------------------------------------------------------------------------


def test_cuda_graph_disabled_matches_eager_generator():
    """OmniVoiceGenerator with enable_cuda_graph=False must produce the same
    _step_forward output as one with enable_cuda_graph=True (after warmup).

    Uses a 2-layer config so warmup completes quickly without loading weights.
    """
    from vllm_omni.model_executor.models.omnivoice.omnivoice_generator import OmniVoiceGenerator
    from vllm_omni.transformers_utils.configs.omnivoice import OmniVoiceConfig

    cfg = OmniVoiceConfig()
    cfg.llm_num_hidden_layers = 2
    cfg.num_hidden_layers = 2

    torch.manual_seed(0)
    gen_eager = OmniVoiceGenerator(cfg).to(DEVICE).eval()
    gen_eager.config.enable_cuda_graph = False
    gen_eager._cuda_graph_fwd = None

    torch.manual_seed(0)
    gen_graph = OmniVoiceGenerator(cfg).to(DEVICE).eval()
    gen_graph.config.cuda_graph_capture_sizes = [64]
    from vllm_omni.model_executor.models.omnivoice.omnivoice_generator import _OmniVoiceCUDAGraphForward

    gen_graph._cuda_graph_fwd = _OmniVoiceCUDAGraphForward(gen_graph, capture_sizes=[64])
    gen_graph._cuda_graph_fwd.warmup(DEVICE)

    seq_len = 64
    ids = torch.zeros(2, cfg.num_audio_codebook, seq_len, dtype=torch.long, device=DEVICE)
    mask = torch.ones(2, seq_len, dtype=torch.bool, device=DEVICE)
    attn = torch.ones(2, 1, seq_len, seq_len, dtype=torch.bool, device=DEVICE)

    gen_eager._ensure_rope(seq_len, DEVICE)
    dtype = gen_eager.text_embedding.weight.dtype
    cos = gen_eager._rope_cos[:seq_len].to(device=DEVICE, dtype=dtype)
    sin = gen_eager._rope_sin[:seq_len].to(device=DEVICE, dtype=dtype)

    with torch.no_grad():
        eager_logits = gen_eager._step_forward(ids, mask, attn, cos, sin)
        graph_logits = gen_graph._cuda_graph_fwd(ids, mask, attn)

    torch.testing.assert_close(graph_logits, eager_logits, atol=0, rtol=0)
