# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU tests for the audio VAE transposed-conv -> conv1d phase decomposition."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import vllm_omni.diffusion.models.minimax_h3.npu.audio_tconv as audio_tconv
from vllm_omni.diffusion.models.minimax_h3.npu.audio_tconv import (
    _conv_transpose1d_as_conv1d,
    _depthwise_upsample_as_conv1d,
    install_audio_vae_tconv_conv1d,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize(
    ("in_ch", "out_ch", "kernel", "stride", "padding", "bias"),
    [
        (12, 8, 9, 5, 2, True),  # BigVGAN stage 0/1 shape: K % stride != 0 (zero-pad path)
        (8, 4, 4, 2, 1, True),  # BigVGAN stages 2-6 shape: K == 2 * stride
        (16, 16, 16, 4, 6, True),
        (6, 5, 12, 2, 0, False),  # no padding, no bias
    ],
)
def test_full_transpose_matches_native(in_ch, out_ch, kernel, stride, padding, bias):
    torch.manual_seed(0)
    x = torch.randn(2, in_ch, 37, dtype=torch.float64)
    weight = torch.randn(in_ch, out_ch, kernel, dtype=torch.float64)
    bias = torch.randn(out_ch, dtype=torch.float64) if bias else None

    reference = F.conv_transpose1d(x, weight, bias, stride=stride, padding=padding)
    value = _conv_transpose1d_as_conv1d(x, weight, bias, stride, padding)

    assert value.shape == reference.shape
    torch.testing.assert_close(value, reference, rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    ("channels", "ratio", "kernel"),
    [
        (7, 2, 12),  # the Activation1d default used across the BigVGAN decoder
        (3, 4, 8),
        (5, 2, 6),
    ],
)
def test_depthwise_upsample_matches_native(channels, ratio, kernel):
    torch.manual_seed(0)
    x = torch.randn(2, channels, 51, dtype=torch.float64)
    filter_ = torch.rand(1, 1, kernel, dtype=torch.float64) + 0.1
    pad = kernel // ratio - 1
    pad_left = pad * ratio + (kernel - ratio) // 2
    pad_right = pad * ratio + (kernel - ratio + 1) // 2

    reference = F.pad(x, (pad, pad), mode="replicate")
    reference = ratio * F.conv_transpose1d(reference, filter_.expand(channels, -1, -1), stride=ratio, groups=channels)
    reference = reference[..., pad_left : reference.shape[-1] - pad_right]

    value = _depthwise_upsample_as_conv1d(x, filter_, ratio, pad, pad_left, pad_right, ratio)

    assert value.shape == reference.shape
    torch.testing.assert_close(value, reference, rtol=0, atol=1e-12)


class UpSample1d(nn.Module):
    """Same name, buffers, and attributes as the checkpoint's remote class."""

    def __init__(self, ratio=2, kernel_size=12):
        super().__init__()
        self.ratio = ratio
        self.stride = ratio
        self.kernel_size = kernel_size
        self.pad = kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (kernel_size - self.stride + 1) // 2
        self.register_buffer("filter", torch.rand(1, 1, kernel_size) + 0.1)

    def forward(self, x):
        _, channels, _ = x.shape
        out = F.pad(x, (self.pad, self.pad), mode="replicate")
        out = self.ratio * F.conv_transpose1d(
            out, self.filter.expand(channels, -1, -1), stride=self.stride, groups=channels
        )
        return out[..., self.pad_left : out.shape[-1] - self.pad_right]


def _audio_graph() -> nn.Module:
    root = nn.Module()
    root.stage = nn.ConvTranspose1d(12, 8, 9, stride=5, padding=2)
    root.act = UpSample1d()
    nested = nn.Module()
    nested.ups = nn.ModuleList([nn.ModuleList([root.stage]), nn.ModuleList([root.act])])
    root.nested = nested
    return root


def test_install_is_a_noop_without_env(monkeypatch):
    monkeypatch.delenv(audio_tconv.AUDIO_TCONV_CONV1D_ENV, raising=False)
    root = _audio_graph()

    assert install_audio_vae_tconv_conv1d(root) == (0, 0)
    # Bound methods are recreated per attribute access; compare the underlying functions.
    assert root.stage.forward.__func__ is nn.ConvTranspose1d.forward
    assert root.act.forward.__func__ is UpSample1d.forward


def test_install_patches_both_families_and_cpu_output_is_unchanged(monkeypatch):
    monkeypatch.setenv(audio_tconv.AUDIO_TCONV_CONV1D_ENV, "1")
    root = _audio_graph()
    torch.manual_seed(1)
    x_stage = torch.randn(2, 12, 31)
    x_act = torch.randn(2, 3, 31)
    ref_stage = root.stage(x_stage).clone()
    ref_act = root.act(x_act).clone()

    assert install_audio_vae_tconv_conv1d(root) == (1, 1)

    # On CPU the patched forwards must take their native branch bit-for-bit.
    torch.testing.assert_close(root.stage(x_stage), ref_stage, rtol=0, atol=0)
    torch.testing.assert_close(root.act(x_act), ref_act, rtol=0, atol=0)
    assert root.stage.forward.__func__ is audio_tconv._patched_conv_transpose_forward
    assert root.act.forward.__func__ is audio_tconv._patched_upsample1d_forward


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [("1", True), ("true", True), ("yes", True), ("on", True), ("0", False), ("off", False), ("", False)],
)
def test_env_parsing(monkeypatch, env_value, expected):
    monkeypatch.setenv(audio_tconv.AUDIO_TCONV_CONV1D_ENV, env_value)
    assert audio_tconv._audio_tconv_env_enabled() is expected
