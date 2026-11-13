# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU math and wiring tests for the Audio VAE C16 diagonal FIR pack."""

import pytest
import torch
import torch.nn as nn
from torch.nn import functional as F

from vllm_omni.diffusion.models.minimax_h3 import vae_audio_fir_channel_pack as pack

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _taps(seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(1, 1, 12, generator=generator, dtype=torch.float32)


class UpSample1d(nn.Module):
    def __init__(self, taps: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("filter", taps)
        self.ratio = 2
        self.stride = 2
        self.pad = 5
        self.pad_left = 15
        self.pad_right = 15
        self.called = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.called += 1
        _, channels, _ = x.shape
        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        x = self.ratio * F.conv_transpose1d(
            x,
            self.filter.expand(channels, -1, -1),
            stride=self.stride,
            groups=channels,
        )
        return x[..., self.pad_left : -self.pad_right]


class LowPassFilter1d(nn.Module):
    def __init__(self, taps: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("filter", taps)
        self.stride = 2
        self.pad_left = 5
        self.pad_right = 6
        self.padding = True
        self.padding_mode = "replicate"
        self.called = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.called += 1
        _, channels, _ = x.shape
        x = F.pad(x, (self.pad_left, self.pad_right), mode=self.padding_mode)
        return F.conv1d(
            x,
            self.filter.expand(channels, -1, -1),
            stride=self.stride,
            groups=channels,
        )


class _FakeDecoder(nn.Module):
    def __init__(self, up: nn.Module, low: nn.Module) -> None:
        super().__init__()
        self.up = up
        self.low = low


class _FakeAudioVAE(nn.Module):
    def __init__(self, up: nn.Module, low: nn.Module) -> None:
        super().__init__()
        self.decoder = _FakeDecoder(up, low)


@pytest.mark.parametrize(
    "count,width",
    [(16, 16), (32, 16), (8, 8), (24, 8), (4, 4), (12, 4), (2, 2), (6, 2), (3, 1), (5, 1), (1, 1)],
)
def test_pack_width(count, width):
    assert pack._pack_width(count) == width


def test_diagonal_filter_keeps_only_the_channel_diagonal():
    taps = _taps()
    weight = pack._diagonal_filter(taps, 16)
    assert weight.shape == (16, 16, 12)
    for channel in range(16):
        torch.testing.assert_close(weight[channel, channel], taps.reshape(-1))
        off_diagonal = weight[channel].clone()
        off_diagonal[channel] = 0
        assert torch.count_nonzero(off_diagonal) == 0


def test_phase_filter_splits_even_and_odd_taps():
    taps = _taps()
    weight = pack._phase_filter(taps, 16)
    assert weight.shape == (32, 16, 6)
    flat = taps.reshape(-1)
    for channel in range(16):
        torch.testing.assert_close(weight[2 * channel, channel], flat[0::2].flip(-1))
        torch.testing.assert_close(weight[2 * channel + 1, channel], flat[1::2].flip(-1))


def test_low_pass_dense_path_matches_depthwise_reference():
    taps = _taps()
    module = LowPassFilter1d(taps)
    module._h3_dense_fir = pack._diagonal_filter(module.filter, 16)
    x = torch.randn(2, 8, 64)
    reference = module.forward(x)
    packed = pack._low(module, x)
    torch.testing.assert_close(packed, reference, atol=1e-5, rtol=1e-5)


def test_up_sample_polyphase_path_matches_transposed_convolution():
    taps = _taps()
    module = UpSample1d(taps)
    module._h3_phase_fir = pack._phase_filter(module.filter, 16)
    x = torch.randn(2, 8, 64)
    reference = module.forward(x)
    packed = pack._up_phase(module, x)
    assert packed.shape == reference.shape
    torch.testing.assert_close(packed, reference, atol=1e-4, rtol=1e-4)


def test_install_wraps_modules_and_is_idempotent():
    up, low = UpSample1d(_taps()), LowPassFilter1d(_taps())
    model = _FakeAudioVAE(up, low)
    assert pack.install(model) == 2
    assert hasattr(up, "_h3_dense_fir") and hasattr(up, "_h3_phase_fir")
    assert hasattr(low, "_h3_dense_fir")
    assert hasattr(up, "_h3_fir_original") and hasattr(low, "_h3_fir_original")
    assert pack.install(model) == 0


def test_cpu_execution_falls_back_to_the_original_forward(monkeypatch):
    up, low = UpSample1d(_taps()), LowPassFilter1d(_taps())
    model = _FakeAudioVAE(up, low)
    pack.install(model)
    monkeypatch.setenv("H3_VAE_OPERATOR_FIR_C16", "1")
    x = torch.randn(2, 8, 64)
    up.forward(x)
    low.forward(x)
    assert up.called == 1 and low.called == 1
    assert not hasattr(up, "_h3_fir_c16_calls")
    assert not hasattr(low, "_h3_fir_c16_calls")


@pytest.mark.parametrize("unsupported", ["absent", "none", "taps", "stride", "ratio"])
def test_install_preserves_unsupported_native_filters(unsupported):
    up, low = UpSample1d(_taps()), LowPassFilter1d(_taps())
    original = up.forward
    if unsupported == "absent":
        del up.filter
    elif unsupported == "none":
        up.filter = None
    elif unsupported == "taps":
        up.filter = up.filter[..., :8]
    elif unsupported == "stride":
        up.stride = 3
    else:
        up.ratio = 3
    assert pack.install(_FakeAudioVAE(up, low)) == 1
    assert getattr(up.forward, "__func__") is getattr(original, "__func__")
    assert not hasattr(up, "_h3_dense_fir")
