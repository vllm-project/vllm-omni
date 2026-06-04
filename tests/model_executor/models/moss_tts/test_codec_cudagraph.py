# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for MossTTSCUDAGraphCodecWrapper numerical equivalence.

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
        # embed: treat codes as float feature vectors summed across NQ
        self.embed = nn.Conv1d(num_quantizers, hidden, kernel_size=3, padding=1)
        self.conv = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.upsample = nn.ConvTranspose1d(hidden, 1, kernel_size=downsample_rate, stride=downsample_rate)

    def _decode(self, codes: torch.Tensor, lengths: torch.Tensor) -> MossAudioTokenizerDecoderOutput:
        """codes: [NQ, B, T], lengths: [B] → audio [B, 1, T*upsample]."""
        nq, b, t = codes.shape
        # sum across NQ dim → [B, NQ, T] for Conv1d (expects [B, C, T])
        x = codes.permute(1, 0, 2).float()  # [B, NQ, T]
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
