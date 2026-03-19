# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for CUDA Graph decoder wrapper numerical equivalence.

Verifies that CUDA Graph-accelerated decoding produces results equivalent
to eager mode, with special attention to padding cases where zero-padding
may introduce small numerical differences due to attention and convolution.

Architecture note: the real Qwen3TTSTokenizerV2Decoder uses causal
convolutions, so zero-padding on the right has minimal impact (~2e-3).
The synthetic decoder here uses standard (non-causal) Conv1d for a
worst-case test of the wrapper mechanism.
"""

import importlib.util
import os

import pytest
import torch
import torch.nn as nn

pytestmark = [pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]

DEVICE = torch.device("cuda:0")
NUM_QUANTIZERS = 8
TOTAL_UPSAMPLE = 4

# Load CUDAGraphDecoderWrapper: try package import first, fall back to direct file load
try:
    from vllm_omni.model_executor.models.qwen3_tts.cuda_graph_decoder_wrapper import CUDAGraphDecoderWrapper
except Exception:
    _WRAPPER_PATH = os.path.join(
        os.path.dirname(__file__),
        os.pardir,
        os.pardir,
        os.pardir,
        os.pardir,
        "vllm_omni",
        "model_executor",
        "models",
        "qwen3_tts",
        "cuda_graph_decoder_wrapper.py",
    )
    _spec = importlib.util.spec_from_file_location("cuda_graph_decoder_wrapper", os.path.abspath(_WRAPPER_PATH))
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    CUDAGraphDecoderWrapper = _mod.CUDAGraphDecoderWrapper


class SyntheticDecoder(nn.Module):
    """A small decoder mimicking Qwen3TTSTokenizerV2Decoder's interface.

    Uses Conv1d layers so that zero-padding can affect neighboring positions
    via the receptive field, providing a worst-case test for padding effects.
    """

    def __init__(self, num_quantizers=NUM_QUANTIZERS, total_upsample=TOTAL_UPSAMPLE):
        super().__init__()
        hidden = 32
        self.total_upsample = total_upsample
        self.embed = nn.Conv1d(num_quantizers, hidden, kernel_size=3, padding=1)
        self.conv1 = nn.Conv1d(hidden, hidden, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.upsample = nn.ConvTranspose1d(hidden, hidden, kernel_size=total_upsample, stride=total_upsample)
        self.out = nn.Conv1d(hidden, 1, kernel_size=1)

    def forward(self, codes):
        x = codes.float()
        x = torch.relu(self.embed(x))
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.upsample(x)
        return self.out(x).clamp(min=-1, max=1)


@pytest.fixture(scope="module")
def decoder():
    """Create a synthetic decoder on CUDA with fixed weights."""
    torch.manual_seed(42)
    return SyntheticDecoder().to(DEVICE).eval()


@pytest.fixture(scope="module")
def wrapper(decoder):
    """Create a warmed-up CUDAGraphDecoderWrapper."""
    w = CUDAGraphDecoderWrapper(
        decoder=decoder,
        capture_sizes=[25, 50, 100],
        num_quantizers=NUM_QUANTIZERS,
        enabled=True,
    )
    w.warmup(DEVICE)
    return w


def _random_codes(seq_len, device=DEVICE):
    return torch.randint(0, 100, (1, NUM_QUANTIZERS, seq_len), dtype=torch.long, device=device)


# ──────────────────────────────────────────────────────────────────
# 1. Exact-size inputs (no padding needed) → bit-identical
# ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("seq_len", [25, 50, 100])
def test_exact_size_numerical_equivalence(decoder, wrapper, seq_len):
    """When input exactly matches a capture size, output must be bit-identical."""
    codes = _random_codes(seq_len)
    with torch.no_grad():
        eager_out = decoder(codes)
        graph_out = wrapper.decode(codes)
    torch.testing.assert_close(graph_out, eager_out, atol=0, rtol=0)


# ──────────────────────────────────────────────────────────────────
# 2. Padded inputs (zero-padding to nearest capture size)
# ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("seq_len", [10, 30, 47, 73, 99])
def test_padded_output_shape_and_length(decoder, wrapper, seq_len):
    """Padded decode must return output trimmed to actual input length."""
    codes = _random_codes(seq_len)
    with torch.no_grad():
        eager_out = decoder(codes)
        graph_out = wrapper.decode(codes)
    expected_len = seq_len * TOTAL_UPSAMPLE
    assert graph_out.shape == eager_out.shape
    assert graph_out.shape[-1] == expected_len


@pytest.mark.parametrize("seq_len", [10, 30, 47, 73, 99])
def test_padded_interior_positions_close(decoder, wrapper, seq_len):
    """Interior positions (away from padding boundary) should be very close.

    The conv receptive field is at most 5 (kernel_size=5), so positions
    more than 2 timesteps from the end (times the upsample factor) should
    be nearly identical between eager and graph modes.
    """
    codes = _random_codes(seq_len)
    with torch.no_grad():
        eager_out = decoder(codes)
        graph_out = wrapper.decode(codes)

    # Exclude the last (receptive_field * upsample) positions from strict check
    boundary = 3 * TOTAL_UPSAMPLE  # conservative: 3 positions * 4x upsample
    if eager_out.shape[-1] > boundary:
        interior_eager = eager_out[..., :(-boundary)]
        interior_graph = graph_out[..., :(-boundary)]
        torch.testing.assert_close(interior_graph, interior_eager, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("seq_len", [10, 30, 47, 73, 99])
def test_padded_output_bounded(decoder, wrapper, seq_len):
    """Padded output values must remain in [-1, 1] and max diff should be bounded."""
    codes = _random_codes(seq_len)
    with torch.no_grad():
        eager_out = decoder(codes)
        graph_out = wrapper.decode(codes)

    assert graph_out.min() >= -1.0 and graph_out.max() <= 1.0
    max_diff = (graph_out - eager_out).abs().max().item()
    # With non-causal conv, boundary diffs can be large (~0.5).
    # The real causal decoder shows ~2e-3.
    assert max_diff < 1.0, f"Max diff {max_diff} exceeds bound"


# ──────────────────────────────────────────────────────────────────
# 3. Fallback to eager (size exceeds all capture sizes) → bit-identical
# ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("seq_len", [101, 150, 200])
def test_fallback_eager_exact_match(decoder, wrapper, seq_len):
    """Input larger than all capture sizes falls back to eager -> bit-identical."""
    codes = _random_codes(seq_len)
    with torch.no_grad():
        eager_out = decoder(codes)
        graph_out = wrapper.decode(codes)
    torch.testing.assert_close(graph_out, eager_out, atol=0, rtol=0)


# ──────────────────────────────────────────────────────────────────
# 4. Chunked decode equivalence
# ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("total_len", [60, 100, 150, 250])
def test_chunked_decode_shape_match(decoder, wrapper, total_len):
    """Chunked decode output shape must match between eager and graph modes."""
    codes = _random_codes(total_len)
    chunk_size, ctx = 50, 10

    with torch.no_grad():
        eager_out = _eager_chunked(decoder, codes, chunk_size, ctx)
        graph_out = wrapper.chunked_decode_with_cudagraph(codes, chunk_size=chunk_size, left_context_size=ctx)

    assert eager_out.shape == graph_out.shape


@pytest.mark.parametrize("total_len", [50, 100])
def test_chunked_decode_exact_size_equivalence(decoder, wrapper, total_len):
    """Chunked decode with chunks matching capture sizes should be bit-identical."""
    codes = _random_codes(total_len)
    # chunk_size=50 matches a capture size exactly, no context overlap
    chunk_size, ctx = 50, 0

    with torch.no_grad():
        eager_out = _eager_chunked(decoder, codes, chunk_size, ctx)
        graph_out = wrapper.chunked_decode_with_cudagraph(codes, chunk_size=chunk_size, left_context_size=ctx)

    torch.testing.assert_close(graph_out, eager_out, atol=0, rtol=0)


def _eager_chunked(decoder, codes, chunk_size, left_context_size):
    """Eager chunked decode matching the real decoder's chunked_decode logic."""
    wavs = []
    start = 0
    total_len = codes.shape[-1]
    while start < total_len:
        end = min(start + chunk_size, total_len)
        ctx = left_context_size if start - left_context_size > 0 else start
        chunk = codes[..., start - ctx : end]
        wav = decoder(chunk)
        wavs.append(wav[..., ctx * decoder.total_upsample :])
        start = end
    return torch.cat(wavs, dim=-1)


# ──────────────────────────────────────────────────────────────────
# 5. Edge cases and control tests
# ──────────────────────────────────────────────────────────────────


def test_single_frame(decoder, wrapper):
    """Single-frame input (seq_len=1) should work with padding."""
    codes = _random_codes(1)
    with torch.no_grad():
        eager_out = decoder(codes)
        graph_out = wrapper.decode(codes)
    assert graph_out.shape == eager_out.shape
    assert graph_out.shape[-1] == TOTAL_UPSAMPLE


def test_disabled_wrapper_matches_eager(decoder, wrapper):
    """Disabled wrapper should produce bit-identical output to eager."""
    codes = _random_codes(30)
    wrapper.enabled = False
    with torch.no_grad():
        eager_out = decoder(codes)
        graph_out = wrapper.decode(codes)
    wrapper.enabled = True
    torch.testing.assert_close(graph_out, eager_out, atol=0, rtol=0)


def test_batch_size_gt1_falls_back(decoder, wrapper):
    """Batch size > 1 should fall back to eager (bit-identical)."""
    codes = torch.randint(0, 100, (2, NUM_QUANTIZERS, 25), dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        eager_out = decoder(codes)
        graph_out = wrapper.decode(codes)
    torch.testing.assert_close(graph_out, eager_out, atol=0, rtol=0)


def test_deterministic_across_calls(decoder, wrapper):
    """Same input should produce identical CUDA graph output across calls."""
    codes = _random_codes(30)
    with torch.no_grad():
        out1 = wrapper.decode(codes)
        out2 = wrapper.decode(codes)
    torch.testing.assert_close(out1, out2, atol=0, rtol=0)
