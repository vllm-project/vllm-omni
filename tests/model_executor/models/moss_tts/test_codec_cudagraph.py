"""Tests for MossTTSCUDAGraphCodecWrapper numerical equivalence.

Verifies that CUDA Graph-accelerated decoding produces results equivalent
to eager mode, with special attention to the two-argument _decode interface
(codes [NQ, 1, T] + lengths [1]) and the NQ-first input convention.
"""

import pytest
import torch
import torch.nn as nn

from vllm_omni.model_executor.models.moss_tts.audio_tokenizer import (
    MossAudioTokenizerDecoderOutput,
)
from vllm_omni.model_executor.models.moss_tts.moss_codec_cudagraph import (
    MossTTSCUDAGraphCodecWrapper,
)

pytestmark = [pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]

DEVICE = torch.device("cuda:0")
NUM_QUANTIZERS = 8
DOWNSAMPLE_RATE = 4  # synthetic; real checkpoint uses 1920


# ---------------------------------------------------------------------------
# Synthetic codec model
# ---------------------------------------------------------------------------


class SyntheticCodecModel(nn.Module):
    """Minimal stand-in for MossAudioTokenizerModel.

    Exposes the same two-argument _decode(codes, lengths) interface and
    returns a MossAudioTokenizerDecoderOutput, mirroring the real model.

    Input:
        codes:   [NQ, B, T]  long  — RVQ codes
        lengths: [B]          long  — valid frame counts
    Output:
        MossAudioTokenizerDecoderOutput(audio=[B, 1, T*upsample], audio_lengths=[B])
    """

    def __init__(
        self,
        num_quantizers: int = NUM_QUANTIZERS,
        downsample_rate: int = DOWNSAMPLE_RATE,
    ):
        super().__init__()
        self.downsample_rate = downsample_rate
        hidden = 32
        # embed: treat NQ quantizers as input channels for Conv1d
        self.embed = nn.Conv1d(num_quantizers, hidden, kernel_size=3, padding=1)
        self.conv = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.upsample = nn.ConvTranspose1d(hidden, 1, kernel_size=downsample_rate, stride=downsample_rate)

    def _decode(self, codes: torch.Tensor, lengths: torch.Tensor) -> MossAudioTokenizerDecoderOutput:
        """codes: [NQ, B, T], lengths: [B] → audio [B, 1, T*upsample]."""
        nq, b, t = codes.shape
        # treat NQ as channel dim → [B, NQ, T] for Conv1d (expects [B, C, T])
        x = codes.permute(1, 0, 2).to(dtype=self.embed.weight.dtype)  # [B, NQ, T]
        x = torch.relu(self.embed(x))  # [B, hidden, T]
        x = torch.relu(self.conv(x))  # [B, hidden, T]
        audio = self.upsample(x)  # [B, 1, T * upsample]
        audio_lengths = lengths * self.downsample_rate
        return MossAudioTokenizerDecoderOutput(audio=audio, audio_lengths=audio_lengths)

    def batch_decode(
        self,
        codes_list: list[torch.Tensor],
        num_quantizers: int | None = None,
    ) -> MossAudioTokenizerDecoderOutput:
        """Mirrors MossAudioTokenizerModel.batch_decode (fallback path)."""
        device = codes_list[0].device
        nq = num_quantizers or codes_list[0].shape[0]
        max_t = max(c.shape[-1] for c in codes_list)
        codes = torch.zeros(nq, len(codes_list), max_t, device=device, dtype=torch.long)
        lengths = torch.zeros(len(codes_list), device=device, dtype=torch.long)
        for i, c in enumerate(codes_list):
            codes[:nq, i, : c.shape[-1]] = c[:nq]
            lengths[i] = c.shape[-1]
        return self._decode(codes, lengths)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return SyntheticCodecModel().to(DEVICE).eval()


@pytest.fixture(scope="module")
def wrapper(model):
    w = MossTTSCUDAGraphCodecWrapper(
        model=model,
        capture_sizes=[25, 50, 100],
        num_quantizers=NUM_QUANTIZERS,
        enabled=True,
    )
    w.warmup(DEVICE)
    return w


def _random_codes(t: int, device: torch.device = DEVICE) -> torch.Tensor:
    """Return [NQ, T] long tensor (single-request convention)."""
    return torch.randint(0, 64, (NUM_QUANTIZERS, t), dtype=torch.long, device=device)


def _eager_decode(model: SyntheticCodecModel, codes_nq_t: torch.Tensor) -> MossAudioTokenizerDecoderOutput:
    """Run the model directly in eager mode (reference output)."""
    t = codes_nq_t.shape[-1]
    codes_nq_1_t = codes_nq_t.unsqueeze(1)  # [NQ, 1, T]
    lengths = torch.tensor([t], dtype=torch.long, device=codes_nq_t.device)
    with torch.no_grad():
        return model._decode(codes_nq_1_t, lengths)


# ---------------------------------------------------------------------------
# 1. Exact-size inputs — must be bit-identical
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("t", [25, 50, 100])
def test_exact_size_bit_identical(model, wrapper, t):
    codes = _random_codes(t)
    ref = _eager_decode(model, codes)
    with torch.no_grad():
        out = wrapper.decode(codes)
    torch.testing.assert_close(out.audio, ref.audio, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# 2. Padded inputs — output trimmed to actual length
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("t", [10, 30, 47, 73, 99])
def test_padded_output_shape(model, wrapper, t):
    codes = _random_codes(t)
    ref = _eager_decode(model, codes)
    with torch.no_grad():
        out = wrapper.decode(codes)
    expected_len = t * DOWNSAMPLE_RATE
    assert out.audio.shape[-1] == expected_len, f"got {out.audio.shape[-1]}, want {expected_len}"
    assert out.audio.shape == ref.audio.shape


@pytest.mark.parametrize("t", [10, 30, 47, 73, 99])
def test_padded_interior_positions_close(model, wrapper, t):
    """Positions well away from the zero-padding boundary must be numerically close."""
    codes = _random_codes(t)
    ref = _eager_decode(model, codes)
    with torch.no_grad():
        out = wrapper.decode(codes)
    boundary = 2 * DOWNSAMPLE_RATE
    if ref.audio.shape[-1] > boundary:
        torch.testing.assert_close(
            out.audio[..., :-boundary],
            ref.audio[..., :-boundary],
            atol=1e-5,
            rtol=1e-5,
        )


# ---------------------------------------------------------------------------
# 3. Fallback to eager (T > all capture sizes)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("t", [101, 150, 200])
def test_fallback_exact_match(model, wrapper, t):
    codes = _random_codes(t)
    ref = _eager_decode(model, codes)
    with torch.no_grad():
        out = wrapper.decode(codes)
    torch.testing.assert_close(out.audio, ref.audio, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# 4. Static buffer not aliased across calls
# ---------------------------------------------------------------------------


def test_output_not_aliased_after_later_replay(model, wrapper):
    """clone() in decode() must prevent later replays from overwriting the result."""
    codes_a = _random_codes(50)
    codes_b = _random_codes(50)
    with torch.no_grad():
        out_a = wrapper.decode(codes_a)
        saved = out_a.audio.clone()
        _ = wrapper.decode(codes_b)  # replay overwrites static buffer
    torch.testing.assert_close(out_a.audio, saved, atol=0, rtol=0)


def test_deterministic_across_calls(model, wrapper):
    codes = _random_codes(30)
    with torch.no_grad():
        out1 = wrapper.decode(codes)
        out2 = wrapper.decode(codes)
    torch.testing.assert_close(out1.audio, out2.audio, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# 5. Disabled wrapper falls back to eager
# ---------------------------------------------------------------------------


def test_disabled_falls_back_to_eager(model, wrapper):
    codes = _random_codes(30)
    ref = _eager_decode(model, codes)
    wrapper.enabled = False
    with torch.no_grad():
        out = wrapper.decode(codes)
    wrapper.enabled = True
    torch.testing.assert_close(out.audio, ref.audio, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# 6. audio_lengths is consistent with audio shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("t", [25, 30, 50, 99, 100])
def test_audio_lengths_consistent(model, wrapper, t):
    codes = _random_codes(t)
    with torch.no_grad():
        out = wrapper.decode(codes)
    assert out.audio_lengths is not None
    assert int(out.audio_lengths[0].item()) == out.audio.shape[-1]


# ---------------------------------------------------------------------------
# 7. NQ-first input layout: codes_nq_t is [NQ, T], not [T, NQ]
# ---------------------------------------------------------------------------


def test_nq_first_layout_matches_eager(model, wrapper):
    """Verify that wrapper correctly interprets [NQ, T] (NQ-first) layout."""
    t = 25
    codes = _random_codes(t)  # [NQ, T]
    ref = _eager_decode(model, codes)
    with torch.no_grad():
        out = wrapper.decode(codes)
    torch.testing.assert_close(out.audio, ref.audio, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# 8. Non-contiguous input — copy_() must handle strided tensors correctly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("t", [20, 47])
def test_noncontiguous_input_matches_contiguous(model, wrapper, t):
    """Padded path uses copy_() on a slice; must produce the same result
    whether codes_nq_t is contiguous or a non-contiguous view (e.g. a column
    slice of a larger buffer, which is the realistic production layout when the
    caller slices out one request from a batched tensor)."""
    codes_contiguous = _random_codes(t)
    # Build a non-contiguous view: embed in a larger [NQ, 2, T] buffer and
    # take column 0.  The resulting tensor has stride (2*T, 2, 1) → non-contiguous.
    buf = torch.zeros(NUM_QUANTIZERS, 2, t, dtype=torch.long, device=DEVICE)
    buf[:, 0, :] = codes_contiguous
    codes_noncontig = buf[:, 0, :]  # [NQ, T], non-contiguous
    assert not codes_noncontig.is_contiguous()

    with torch.no_grad():
        out_c = wrapper.decode(codes_contiguous)
        out_nc = wrapper.decode(codes_noncontig)
    torch.testing.assert_close(out_c.audio, out_nc.audio, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# 9. Zero-padding uses token 0 — must be a valid codebook index (no OOB)
# ---------------------------------------------------------------------------


def test_zero_padding_no_cuda_error(model, wrapper):
    """Padded inputs fill unused frames with code 0.  Verify that the zero-
    padded region does not corrupt the output shape.

    Note: SyntheticCodecModel uses Conv1d (not nn.Embedding), so it cannot
    trigger a real CUDA index-out-of-bounds on large token ids.  This test
    validates shape correctness only; OOB safety on the real codec requires
    an integration test with MossAudioTokenizerModel."""
    t = 7  # well below the 25-frame bucket → large zero-padded region
    codes = _random_codes(t)
    with torch.no_grad():
        out = wrapper.decode(codes)
    torch.accelerator.synchronize()  # surface any deferred CUDA errors
    assert out.audio.shape[-1] == t * DOWNSAMPLE_RATE


# ---------------------------------------------------------------------------
# 10. Shape alignment: graph path and eager fallback return identical shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("t", [25, 30, 101])
def test_graph_and_eager_shape_identical(model, wrapper, t):
    """Graph replay (t<=100) and eager fallback (t>100) must return tensors
    with the same number of dimensions and the same batch/channel layout so
    downstream consumers can treat both paths uniformly."""
    codes = _random_codes(t)
    ref = _eager_decode(model, codes)
    with torch.no_grad():
        out = wrapper.decode(codes)
    assert out.audio.ndim == ref.audio.ndim, f"ndim mismatch: graph={out.audio.ndim} eager={ref.audio.ndim}"
    assert out.audio.shape[:-1] == ref.audio.shape[:-1], (
        f"batch/channel shape mismatch: graph={out.audio.shape} eager={ref.audio.shape}"
    )


# ---------------------------------------------------------------------------
# 11. Half-precision weights: graph and eager must agree under fp16/bf16
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_half_precision_graph_eager_agree(dtype):
    """CUDA Graph must produce outputs consistent with eager when model weights
    are in half precision.  The wrapper itself stores codes as long and lengths
    as long; only the model's conv weights change dtype."""
    torch.manual_seed(0)
    model_hp = SyntheticCodecModel().to(device=DEVICE, dtype=dtype).eval()
    wrapper_hp = MossTTSCUDAGraphCodecWrapper(
        model=model_hp,
        capture_sizes=[25, 50],
        num_quantizers=NUM_QUANTIZERS,
        enabled=True,
    )
    wrapper_hp.warmup(DEVICE)

    codes = _random_codes(25)
    ref = _eager_decode(model_hp, codes)
    with torch.no_grad():
        out = wrapper_hp.decode(codes)
    # Half-precision accumulation introduces small numerical error; use
    # tolerances that match typical fp16/bf16 rounding.
    atol = 1e-2 if dtype == torch.float16 else 5e-2
    torch.testing.assert_close(out.audio.float(), ref.audio.float(), atol=atol, rtol=1e-3)


# ---------------------------------------------------------------------------
# 12. Empty input (T=0): wrapper must not crash and return zero-length audio
# ---------------------------------------------------------------------------


def test_empty_input_t0(model, wrapper):
    """T=0 input must be handled gracefully: no exception, audio length == 0.

    The outer MossTTSCodecDecoder guards against empty segments, but the
    wrapper should also handle T=0 without crashing so it is safe to call
    directly."""
    codes = _random_codes(0)  # [NQ, 0]
    with torch.no_grad():
        out = wrapper.decode(codes)
    assert out.audio.shape[-1] == 0, f"expected empty audio, got shape {out.audio.shape}"
