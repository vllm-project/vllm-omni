# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Correctness and regression tests for OmniVoiceDecoder.

Covers three areas:
1. NaN regression — verifies the fp16 overflow fix:
     acoustic decoder must operate in float32 even when fc2 weights are fp16.
     Demonstrates that the pre-fix code path (no .float() after fc2) produces
     NaN/Inf with large-scale weights, while the fixed path does not.
2. Output shape / value sanity — correct [B, 1, T*upsample] shape, finite values.
3. Acoustic decoder float32 invariant — weights must be float32 after build.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from vllm_omni.model_executor.models.omnivoice.omnivoice_decoder import (
    HiggsAudioRVQ,
    OmniVoiceDecoder,
)
from vllm_omni.transformers_utils.configs.omnivoice import OmniVoiceConfig

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

DEVICE = torch.device("cuda:0")
UPSAMPLE = 4  # synthetic acoustic decoder stride (real DAC uses 960)
T_FRAMES = 25  # number of audio token frames to decode


# ---------------------------------------------------------------------------
# Tiny acoustic decoder: ConvTranspose1d upsample by UPSAMPLE
# ---------------------------------------------------------------------------


class _TinyAcousticDecoder(nn.Module):
    def __init__(self, in_channels: int = 256, upsample: int = UPSAMPLE):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels, 1, kernel_size=upsample, stride=upsample)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)  # [B, 1, T*upsample]


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def _build_decoder(
    *,
    fc2_dtype: torch.dtype = torch.float32,
    acoustic_dtype: torch.dtype = torch.float32,
    weight_scale: float = 1.0,
) -> OmniVoiceDecoder:
    """Build an OmniVoiceDecoder with synthetic components (no model checkpoint required).

    Args:
        fc2_dtype: dtype for the fc2 linear layer weights.
        acoustic_dtype: dtype for the acoustic decoder weights.
        weight_scale: constant weight value for overflow testing.
    """
    config = OmniVoiceConfig()
    decoder = OmniVoiceDecoder(config)

    decoder.quantizer = HiggsAudioRVQ(num_quantizers=8, codebook_size=1024, codebook_dim=64, hidden_size=1024).to(
        DEVICE
    )

    decoder.fc2 = nn.Linear(1024, 256).to(DEVICE).to(fc2_dtype)

    acoustic = _TinyAcousticDecoder(in_channels=256, upsample=UPSAMPLE).to(DEVICE).to(acoustic_dtype)
    if weight_scale != 1.0:
        with torch.no_grad():
            nn.init.constant_(acoustic.conv.weight, weight_scale)
            nn.init.constant_(acoustic.conv.bias, weight_scale)
    decoder.acoustic_decoder = acoustic

    decoder._loaded = True
    return decoder


# ---------------------------------------------------------------------------
# 1. NaN regression
# ---------------------------------------------------------------------------


def test_no_nan_fp16_fc2_float32_acoustic():
    """fp16 fc2 + float32 acoustic decoder (the fixed path) must not produce NaN or Inf."""
    decoder = _build_decoder(fc2_dtype=torch.float16, acoustic_dtype=torch.float32)
    tokens = torch.randint(0, 1024, (1, 8, T_FRAMES), device=DEVICE)

    audio = decoder(tokens)

    assert not torch.isnan(audio).any(), "NaN in output — fp16 overflow may have reached acoustic decoder"
    assert not torch.isinf(audio).any(), "Inf in output"


def test_fp16_acoustic_decoder_produces_nan_without_fix():
    """Pre-fix scenario: fp16 acoustic decoder with large weights overflows to NaN/Inf.

    This test documents the original bug and will fail (raising AssertionError)
    if the underlying hardware / Triton version somehow avoids overflow, but
    with weights scaled to 300.0 the ConvTranspose1d intermediate activations
    reliably exceed fp16 max (~65504) in practice.
    """
    decoder = _build_decoder(fc2_dtype=torch.float16, acoustic_dtype=torch.float16, weight_scale=300.0)
    with torch.no_grad():
        nn.init.constant_(decoder.fc2.weight, 300.0)
        nn.init.constant_(decoder.fc2.bias, 300.0)
        for q in decoder.quantizer.quantizers:
            nn.init.constant_(q.codebook.weight, 300.0)

    tokens = torch.randint(0, 1024, (1, 8, T_FRAMES), device=DEVICE)

    # Simulate pre-fix forward: no .float() after fc2
    codes = tokens.transpose(0, 1).long()
    quantized = decoder.quantizer.decode(codes)
    quantized_fp16 = decoder.fc2(quantized.transpose(1, 2).to(decoder.fc2.weight.dtype)).transpose(
        1, 2
    )  # stays fp16, no .float()
    audio_broken = decoder.acoustic_decoder(quantized_fp16)

    assert torch.isnan(audio_broken).any() or torch.isinf(audio_broken).any(), (
        "Expected NaN/Inf in pre-fix fp16 path with large weights — test setup may need a higher weight_scale"
    )


def test_acoustic_decoder_receives_float32_input():
    """The .float() cast after fc2 must ensure acoustic decoder always gets float32 input,
    regardless of fc2 weight dtype."""
    decoder = _build_decoder(fc2_dtype=torch.float16, acoustic_dtype=torch.float32)

    captured_dtypes: list[torch.dtype] = []
    original_fwd = decoder.acoustic_decoder.forward

    def _hook(x: torch.Tensor) -> torch.Tensor:
        captured_dtypes.append(x.dtype)
        return original_fwd(x)

    decoder.acoustic_decoder.forward = _hook  # type: ignore[method-assign]
    tokens = torch.randint(0, 1024, (1, 8, T_FRAMES), device=DEVICE)
    decoder(tokens)

    assert captured_dtypes, "acoustic_decoder.forward was never called"
    assert captured_dtypes[0] == torch.float32, (
        f"Acoustic decoder received {captured_dtypes[0]}, expected torch.float32 — "
        "the .float() upcast after fc2 may be missing"
    )


# ---------------------------------------------------------------------------
# 2. Output shape and value sanity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch,t_frames", [(1, 10), (1, 50), (2, 25)])
def test_output_shape(batch, t_frames):
    """Decoder must return [B, 1, T*upsample] waveform tensor."""
    decoder = _build_decoder()
    tokens = torch.randint(0, 1024, (batch, 8, t_frames), device=DEVICE)

    audio = decoder(tokens)

    assert audio.shape == (batch, 1, t_frames * UPSAMPLE), (
        f"Expected ({batch}, 1, {t_frames * UPSAMPLE}), got {audio.shape}"
    )


def test_no_nan_float32_baseline():
    """Float32 fc2 + float32 acoustic decoder must always produce finite output."""
    decoder = _build_decoder(fc2_dtype=torch.float32, acoustic_dtype=torch.float32)
    tokens = torch.randint(0, 1024, (1, 8, T_FRAMES), device=DEVICE)

    audio = decoder(tokens)

    assert not torch.isnan(audio).any()
    assert not torch.isinf(audio).any()


def test_output_is_3d():
    """Output must be 3-D: [B, 1, samples]. OmniVoiceDecoder.forward unsqueezes if needed."""
    decoder = _build_decoder()
    tokens = torch.randint(0, 1024, (1, 8, 10), device=DEVICE)

    audio = decoder(tokens)

    assert audio.dim() == 3, f"Expected 3-D output, got {audio.dim()}-D"
    assert audio.shape[1] == 1, f"Expected channel dim=1, got {audio.shape[1]}"


# ---------------------------------------------------------------------------
# 3. Acoustic decoder float32 invariant
# ---------------------------------------------------------------------------


def test_acoustic_decoder_weights_are_float32():
    """Acoustic decoder weights must be float32 — any fp16 parameter is a regression indicator.

    In production, load_weights() calls self.acoustic_decoder.float() to enforce this.
    The synthetic builder mirrors that invariant.
    """
    decoder = _build_decoder(fc2_dtype=torch.float16, acoustic_dtype=torch.float32)

    for name, param in decoder.acoustic_decoder.named_parameters():
        assert param.dtype == torch.float32, (
            f"acoustic_decoder.{name} is {param.dtype} — must be float32 (see load_weights fix)"
        )
