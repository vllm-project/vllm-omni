# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Unit tests for the fused gated residual + LayerNorm operators.

``residual_layer_norm`` and ``qkv_head_layer_norm`` are checked against the
eager PyTorch chains they replace, on the native path (any device) and on the
Triton path (CUDA).
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm_omni.model_executor.models.common.ops import qkv_head_layer_norm, residual_layer_norm

pytestmark = [pytest.mark.core_model, pytest.mark.cuda]

_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _conv_taps(history: torch.Tensor, conv: nn.Conv1d) -> torch.Tensor:
    """``history @ [W_0; ...; W_{K-1}]^T``, the tap-stacked GEMM ``residual_layer_norm`` reduces."""
    weight = conv.weight.permute(2, 0, 1).reshape(-1, conv.in_channels)
    return F.linear(history, weight)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("channels", [64, 512, 300])
def test_gated_residual_modulated_layer_norm(device: str, channels: int) -> None:
    torch.manual_seed(0)
    x = torch.randn(3, 7, channels, device=device)
    y = torch.randn(3, 7, channels, device=device)
    gate, scale, shift = (torch.randn(channels, device=device) for _ in range(3))

    expected_residual = x + gate * y
    expected = F.layer_norm(expected_residual, (channels,), eps=1e-6) * (1 + scale) + shift

    residual = x.clone()
    out = residual_layer_norm(residual, y, gate=gate, weight=1 + scale, bias=shift, eps=1e-6, residual_out=residual)
    torch.testing.assert_close(residual, expected_residual, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)

    plain = residual_layer_norm(x, weight=1 + scale, bias=shift, eps=1e-6)
    torch.testing.assert_close(
        plain, F.layer_norm(x, (channels,), eps=1e-6) * (1 + scale) + shift, rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize("device", _DEVICES)
def test_causal_conv_taps_layer_norm_mish_into_strided_history(device: str) -> None:
    """Conv tap sum + bias + LayerNorm + Mish, stored behind a history's cached frames."""
    torch.manual_seed(1)
    channels, frames, width = 32, 9, 2
    conv = nn.Conv1d(channels, channels, width + 1).to(device)
    norm = nn.LayerNorm(channels).to(device)
    with torch.no_grad():
        norm.weight.normal_()
        norm.bias.normal_()
    history = torch.randn(4, width + frames, channels, device=device)
    with torch.no_grad():
        expected = F.mish(norm(conv(history.transpose(1, 2)).transpose(1, 2)))
        destination = torch.full((4, width + frames, channels), float("nan"), device=device)
        residual_layer_norm(
            None,
            _conv_taps(history, conv),
            taps=width + 1,
            y_bias=conv.bias,
            weight=norm.weight,
            bias=norm.bias,
            eps=norm.eps,
            activation="mish",
            out=destination[:, width:],
        )
    torch.testing.assert_close(destination[:, width:], expected, rtol=1e-5, atol=1e-5)
    assert torch.isnan(destination[:, :width]).all()


@pytest.mark.parametrize("device", _DEVICES)
def test_causal_conv_taps_gated_residual(device: str) -> None:
    """``x += gate * conv(history)`` followed by the modulated LayerNorm of the result."""
    torch.manual_seed(2)
    channels, frames, width = 48, 5, 2
    conv = nn.Conv1d(channels, channels, width + 1).to(device)
    history = torch.randn(2, width + frames, channels, device=device)
    x = torch.randn(2, frames, channels, device=device)
    gate, scale, shift = (torch.randn(channels, device=device) for _ in range(3))
    with torch.no_grad():
        expected_x = x + gate * conv(history.transpose(1, 2)).transpose(1, 2)
        expected = F.layer_norm(expected_x, (channels,), eps=1e-6) * (1 + scale) + shift
        residual = x.clone()
        out = residual_layer_norm(
            residual,
            _conv_taps(history, conv),
            taps=width + 1,
            y_bias=conv.bias,
            gate=gate,
            weight=1 + scale,
            bias=shift,
            eps=1e-6,
            residual_out=residual,
        )
    rtol, atol = (5e-3, 5e-3) if device == "cuda" else (1e-5, 1e-5)
    torch.testing.assert_close(residual, expected_x, rtol=rtol, atol=atol)
    torch.testing.assert_close(out, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("device", _DEVICES)
def test_qkv_head_layer_norm_writes_q_and_interleaved_kv(device: str) -> None:
    torch.manual_seed(4)
    batch, frames, heads, head_dim, cached = 3, 6, 4, 16, 5
    q_norm, k_norm = nn.LayerNorm(head_dim).to(device), nn.LayerNorm(head_dim, eps=1e-6).to(device)
    with torch.no_grad():
        for norm in (q_norm, k_norm):
            norm.weight.normal_()
            norm.bias.normal_()
    qkv = torch.randn(batch, frames, 3 * heads * head_dim, device=device)
    kv = torch.full((batch, heads, frames + cached, 2 * head_dim), float("nan"), device=device)
    with torch.no_grad():
        q = qkv_head_layer_norm(qkv, kv, num_heads=heads, head_dim=head_dim, q_norm=q_norm, k_norm=k_norm)
        split = qkv.view(batch, frames, 3, heads, head_dim).transpose(1, 3)  # (b, h, 3, t, d)
        torch.testing.assert_close(q, q_norm(split[:, :, 0]), rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(kv[:, :, :frames, :head_dim], k_norm(split[:, :, 1]), rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(kv[:, :, :frames, head_dim:], split[:, :, 2])
    # The cached frames behind the current chunk are never written.
    assert torch.isnan(kv[:, :, frames:]).all()
