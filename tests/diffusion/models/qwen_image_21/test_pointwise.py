# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.diffusion.models.qwen_image_21.pointwise import residual, rotary, silu_mul

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@pytest.mark.cpu
def test_cpu_and_training_fallback():
    x = torch.randn(2, 7, 31, requires_grad=True)
    y = torch.randn_like(x)
    gate = torch.randn_like(x)
    torch.testing.assert_close(residual(x, y, gate), x + gate * y, rtol=0, atol=0)
    channel_gate = gate[:, :1].contiguous()
    torch.testing.assert_close(residual(x, y, channel_gate), x + channel_gate * y, rtol=0, atol=0)
    torch.testing.assert_close(silu_mul(gate, y), torch.nn.functional.silu(gate) * y, rtol=0, atol=0)
    assert rotary(x, torch.empty(7, 64, dtype=torch.complex64)) is None
    residual(x, y, gate).sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("sequence", [880, 911, 913, 914])
@torch.no_grad()
def test_production_shapes_preserve_rounding(sequence):
    torch.manual_seed(sequence)
    x = torch.randn(1, sequence, 4096, device="cuda", dtype=torch.bfloat16)
    y, gate = torch.randn_like(x), torch.randn_like(x)
    torch.testing.assert_close(residual(x, y, gate), x + gate * y, rtol=0, atol=0)
    gate = torch.randn(1, sequence, 12288, device="cuda", dtype=torch.bfloat16)
    up = torch.randn_like(gate)
    torch.testing.assert_close(silu_mul(gate, up), torch.nn.functional.silu(gate) * up, rtol=0, atol=0)
    projection = torch.randn(2, sequence, 3, 32, 128, device="cuda", dtype=torch.bfloat16)
    query = projection[:, :, 0]
    freqs = torch.polar(torch.ones(sequence, 64, device="cuda"), torch.randn(sequence, 64, device="cuda"))
    expected = (
        torch.view_as_real(torch.view_as_complex(query.float().reshape(2, sequence, 32, 64, 2)) * freqs.unsqueeze(1))
        .flatten(3)
        .to(query.dtype)
    )
    torch.testing.assert_close(rotary(query, freqs), expected, rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.gpu
@torch.no_grad()
def test_silu_all_bf16_values_preserve_eager_rounding():
    bits = torch.arange(65536, device="cuda", dtype=torch.int32).to(torch.int16)
    gate = bits.view(torch.bfloat16)
    up = torch.ones_like(gate)
    expected = torch.nn.functional.silu(gate) * up
    torch.testing.assert_close(silu_mul(gate, up), expected, rtol=0, atol=0, equal_nan=True)
