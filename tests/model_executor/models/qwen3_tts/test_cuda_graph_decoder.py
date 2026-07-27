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


class ShortOutputDecoder(SyntheticDecoder):
    """Decoder variant that returns fewer samples than seq_len * total_upsample."""

    def forward(self, codes):
        out = super().forward(codes)
        return out[..., :-5] if out.shape[-1] > 5 else out


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
        capture_batch_sizes=[1, 2],
        num_quantizers=NUM_QUANTIZERS,
        enabled=True,
    )
    w.warmup(DEVICE)
    return w


def _random_codes(seq_len, batch_size=1, device=DEVICE):
    return torch.randint(0, 100, (batch_size, NUM_QUANTIZERS, seq_len), dtype=torch.long, device=device)


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


@pytest.mark.parametrize("seq_len", [10, 30, 47, 73, 99])
def test_padded_short_output_decode_length_matches_eager(seq_len):
    """Streaming ``decode()`` of a shorter-than-nominal decoder must match eager length.

    Regression for #4466: Qwen3-Omni Code2Wav returns fewer samples than
    ``seq_len * total_upsample``. The graph ``decode()`` path used to trim to the
    nominal length, so for padded (non-capture-size) chunks it leaked a fixed
    surplus of stale buffer-tail samples. The streaming concat turned that
    per-chunk surplus into a ~46 ms "lag in the middle". The trim must instead be
    relative to the captured output length, matching the eager forward exactly.
    """
    short_decoder = ShortOutputDecoder().to(DEVICE).eval()
    short_wrapper = CUDAGraphDecoderWrapper(
        decoder=short_decoder,
        capture_sizes=[25, 50, 100],
        capture_batch_sizes=[1],
        num_quantizers=NUM_QUANTIZERS,
        enabled=True,
    )
    short_wrapper.warmup(DEVICE)
    codes = _random_codes(seq_len)
    with torch.no_grad():
        eager_out = short_decoder(codes)
        graph_out = short_wrapper.decode(codes)
    surplus = graph_out.shape[-1] - eager_out.shape[-1]
    assert surplus == 0, (
        f"graph length {graph_out.shape[-1]} != eager length {eager_out.shape[-1]} "
        f"(surplus {surplus}); the padded graph output is leaking stale tail samples"
    )
    # Interior content (away from the conv receptive-field boundary) must still match.
    boundary = 3 * TOTAL_UPSAMPLE
    if eager_out.shape[-1] > boundary:
        torch.testing.assert_close(graph_out[..., :-boundary], eager_out[..., :-boundary], atol=1e-5, rtol=1e-5)


def test_compiled_padded_short_output_length_matches_eager(monkeypatch):
    """Compiled replay path must also trim short-output decoders to eager length.

    Same regression as the graph path (#4466) but exercising the torch.compile +
    CUDA Graph branch in ``_decode``. ``torch.compile`` is faked (as in the other
    compiled-path tests) so the test stays fast and deterministic.
    """

    def _fake_compile(model, **_kwargs):
        def _compiled(codes):
            return model(codes)

        return _compiled

    monkeypatch.setattr(torch, "compile", _fake_compile)

    short_decoder = ShortOutputDecoder().to(DEVICE).eval()
    compiled_wrapper = CUDAGraphDecoderWrapper(
        decoder=short_decoder,
        capture_sizes=[25, 50],
        capture_batch_sizes=[1],
        compile_shapes=[(1, 25), (1, 50)],
        num_quantizers=NUM_QUANTIZERS,
        enabled=True,
    )
    compiled_wrapper.warmup(DEVICE)
    codes = _random_codes(30)  # not a capture size -> padded to bucket 50 on the compiled path
    with torch.no_grad():
        eager_out = short_decoder(codes)
        graph_out = compiled_wrapper.decode(codes)
    surplus = graph_out.shape[-1] - eager_out.shape[-1]
    assert surplus == 0, (
        f"compiled graph length {graph_out.shape[-1]} != eager length {eager_out.shape[-1]} (surplus {surplus})"
    )


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


def test_chunked_decode_output_survives_later_replay(wrapper):
    """Chunked output must not alias graph static buffers overwritten by later replays."""
    codes = _random_codes(100)
    overwrite_codes = _random_codes(100)

    with torch.no_grad():
        graph_out = wrapper.chunked_decode_with_cudagraph(codes, chunk_size=50, left_context_size=0)
        expected = graph_out.clone()
        _ = wrapper.decode(overwrite_codes[..., :50])
        _ = wrapper.decode(overwrite_codes)

    torch.testing.assert_close(graph_out, expected, atol=0, rtol=0)


def test_chunked_decode_preserves_short_chunk_concat_semantics():
    """Chunked decode must allow shorter-than-nominal chunk outputs.

    Qwen3-Omni non-async startup can hit a decoder path where a chunk returns
    fewer waveform samples than ``code_len * total_upsample``. The wrapper must
    preserve the original eager concat behavior instead of copying into exact
    nominal slices.
    """
    torch.manual_seed(123)
    short_decoder = ShortOutputDecoder().to(DEVICE).eval()
    short_wrapper = CUDAGraphDecoderWrapper(
        decoder=short_decoder,
        capture_sizes=[50],
        capture_batch_sizes=[1],
        num_quantizers=NUM_QUANTIZERS,
        enabled=True,
    )
    short_wrapper.warmup(DEVICE)
    codes = _random_codes(100)

    with torch.no_grad():
        eager_out = _eager_chunked(short_decoder, codes, chunk_size=50, left_context_size=0)
        graph_out = short_wrapper.chunked_decode_with_cudagraph(codes, chunk_size=50, left_context_size=0)

    assert graph_out.shape == eager_out.shape
    assert graph_out.shape[-1] < 100 * TOTAL_UPSAMPLE
    torch.testing.assert_close(graph_out, eager_out, atol=0, rtol=0)


def test_batched_chunked_decode_variable_lengths_matches_per_request_eager(decoder, wrapper):
    """Variable-length chunk batching should match independent chunked decodes."""
    long_codes = _random_codes(100)
    short_codes = _random_codes(50)
    padded_codes = torch.zeros(2, NUM_QUANTIZERS, 100, dtype=torch.long, device=DEVICE)
    padded_codes[0, :, :] = long_codes[0]
    padded_codes[1, :, :50] = short_codes[0]

    with torch.no_grad():
        eager_long = _eager_chunked(decoder, long_codes, chunk_size=50, left_context_size=0)
        eager_short = _eager_chunked(decoder, short_codes, chunk_size=50, left_context_size=0)
        graph_out = wrapper.batched_chunked_decode_with_cudagraph(
            padded_codes,
            [100, 50],
            chunk_size=50,
            left_context_size=0,
            max_batch_size=2,
        )

    assert graph_out.shape == (2, 1, 100 * TOTAL_UPSAMPLE)
    torch.testing.assert_close(graph_out[0:1], eager_long, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(graph_out[1:2, :, : 50 * TOTAL_UPSAMPLE], eager_short, atol=1e-6, rtol=1e-6)


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


def test_batch_size_gt1_uses_matching_graph(decoder, wrapper):
    """Captured batch size > 1 should replay a matching graph."""
    assert (2, 25) in wrapper.graphs
    codes = _random_codes(25, batch_size=2)
    with torch.no_grad():
        eager_out = decoder(codes)
        graph_out = wrapper.decode(codes)
    torch.testing.assert_close(graph_out, eager_out, atol=0, rtol=0)


def test_uncaptured_batch_size_falls_back(decoder, wrapper):
    """Uncaptured batch sizes should fall back to eager."""
    assert (3, 25) not in wrapper.graphs
    codes = _random_codes(25, batch_size=3)
    with torch.no_grad():
        eager_out = decoder(codes)
        graph_out = wrapper.decode(codes)
    torch.testing.assert_close(graph_out, eager_out, atol=0, rtol=0)


def test_extra_capture_shape_uses_sparse_graph(decoder):
    """Extra capture shapes should not expand to a full batch x size product."""
    sparse_wrapper = CUDAGraphDecoderWrapper(
        decoder=decoder,
        capture_sizes=[25],
        capture_batch_sizes=[1],
        extra_capture_shapes=[(2, 50)],
        num_quantizers=NUM_QUANTIZERS,
        enabled=True,
    )
    sparse_wrapper.warmup(DEVICE)

    assert (1, 25) in sparse_wrapper.graphs
    assert (2, 50) in sparse_wrapper.graphs
    assert (2, 25) not in sparse_wrapper.graphs

    codes = _random_codes(50, batch_size=2)
    with torch.no_grad():
        eager_out = decoder(codes)
        graph_out = sparse_wrapper.decode(codes)
    torch.testing.assert_close(graph_out, eager_out, atol=0, rtol=0)


def test_compile_shape_supports_exact_and_padded_buckets(decoder, monkeypatch):
    """Configured torch.compile shapes should replay exact and padded CUDA Graph buckets."""

    compile_kwargs = {}

    def _fake_compile(model, **_kwargs):
        compile_kwargs.update(_kwargs)

        def _compiled(codes):
            return model(codes) + 0.125

        return _compiled

    monkeypatch.setattr(torch, "compile", _fake_compile)

    compiled_wrapper = CUDAGraphDecoderWrapper(
        decoder=decoder,
        capture_sizes=[25, 50],
        capture_batch_sizes=[1],
        compile_shapes=[(1, 25), (1, 50)],
        num_quantizers=NUM_QUANTIZERS,
        enabled=True,
    )
    compiled_wrapper.warmup(DEVICE)

    exact_codes = _random_codes(25)
    padded_codes = _random_codes(30)
    uncaptured_codes = _random_codes(60)
    padded_static = torch.zeros(1, NUM_QUANTIZERS, 50, dtype=torch.long, device=DEVICE)
    padded_static[:, :, :30] = padded_codes
    with torch.no_grad():
        exact_eager = decoder(exact_codes)
        exact_out = compiled_wrapper.decode(exact_codes)
        padded_graph_expected = decoder(padded_static)[..., : 30 * TOTAL_UPSAMPLE]
        padded_out = compiled_wrapper.decode(padded_codes)
        uncaptured_eager = decoder(uncaptured_codes)
        uncaptured_out = compiled_wrapper.decode(uncaptured_codes)

    torch.testing.assert_close(exact_out, exact_eager + 0.125, atol=0, rtol=0)
    torch.testing.assert_close(padded_out, padded_graph_expected + 0.125, atol=0, rtol=0)
    torch.testing.assert_close(uncaptured_out, uncaptured_eager, atol=0, rtol=0)
    assert compile_kwargs["mode"] == "default"
    assert compile_kwargs["fullgraph"] is False
    assert compile_kwargs["dynamic"] is False


def test_deterministic_across_calls(decoder, wrapper):
    """Same input should produce identical CUDA graph output across calls."""
    codes = _random_codes(30)
    with torch.no_grad():
        out1 = wrapper.decode(codes)
        out2 = wrapper.decode(codes)
    torch.testing.assert_close(out1, out2, atol=0, rtol=0)


# ──────────────────────────────────────────────────────────────────
# 6. compute_capture_sizes
# ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "kwargs,expected_in,not_expected",
    [
        ({}, [2, 4, 8, 16, 32, 64, 128, 256, 325], [512]),
        (
            {"codec_chunk_frames": 33, "codec_left_context_frames": 25},
            [2, 4, 8, 16, 32, 33, 58, 64, 128, 256, 325],
            [512],
        ),
        (
            {"codec_chunk_frames": 25, "codec_left_context_frames": 25},
            [2, 4, 8, 16, 25, 32, 50, 64, 128, 256, 325],
            [512],
        ),
        (
            {
                "codec_chunk_frames": 25,
                "codec_left_context_frames": 72,
                "decode_chunk_size": 400,
                "decode_left_context": 17,
            },
            [2, 4, 8, 16, 25, 32, 64, 97, 128, 256, 417],
            [325, 512],
        ),
    ],
    ids=["default", "streaming_c33", "streaming_c25", "custom_decode_chunk"],
)
def test_compute_capture_sizes(kwargs, expected_in, not_expected):
    """compute_capture_sizes produces expected sizes capped by max useful size."""
    sizes = CUDAGraphDecoderWrapper.compute_capture_sizes(**kwargs)
    for val in expected_in:
        assert val in sizes, f"{val} not in {sizes}"
    for val in not_expected:
        assert val not in sizes, f"{val} should not be in {sizes}"


# ──────────────────────────────────────────────────────────────────
# 7. SnakeBeta Triton kernel vs eager equivalence
# ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "batch,channels,seq_len",
    [(2, 64, 1000), (1, 32, 1), (1, 32, 7), (1, 32, 128), (1, 32, 1024), (1, 32, 4096)],
)
@pytest.mark.parametrize(
    "dtype,atol,rtol",
    [
        (torch.float32, 1e-5, 1e-5),
        (torch.bfloat16, 1e-2, 1e-2),
        (torch.float16, 1e-3, 1e-3),
    ],
    ids=["fp32", "bf16", "fp16"],
)
def test_snakebeta_triton_vs_eager(batch, channels, seq_len, dtype, atol, rtol):
    """Fused Triton SnakeBeta kernel must match eager PyTorch output across dtypes."""
    from vllm_omni.model_executor.models.common.snake_activation import SnakeBeta

    if not SnakeBeta._init_triton():
        pytest.skip("Triton not available")

    torch.manual_seed(42)
    snake = SnakeBeta(in_features=channels).to(DEVICE).to(dtype).eval()
    x = torch.randn(batch, channels, seq_len, device=DEVICE, dtype=dtype)

    with torch.no_grad():
        eager_out = snake._eager_forward(x)
        triton_out = snake._triton_forward(x)

    assert triton_out.dtype == x.dtype
    torch.testing.assert_close(triton_out, eager_out, atol=atol, rtol=rtol)
