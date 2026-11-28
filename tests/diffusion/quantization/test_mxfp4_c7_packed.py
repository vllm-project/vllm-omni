# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.quantization.mxfp4_c7 import pack_mxfp4_c7

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def unpack(packed, scales, dtype):
    raw = packed.view(torch.uint8)
    codes = torch.stack((raw & 15, raw >> 4), dim=-1).flatten(-2).long()
    levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=dtype, device=packed.device)
    values = levels[codes & 7] * torch.where(codes >= 8, -1, 1)
    scale = torch.exp2((scales.to(torch.int16) - 127).to(dtype))
    return (values.reshape(*scales.shape, 32) * scale.unsqueeze(-1)).reshape(values.shape)


def reference(x):
    blocks = x.reshape(*x.shape[:-1], x.shape[-1] // 32, 32)
    exponent = torch.ceil(torch.log2(blocks.abs().amax(-1, keepdim=True).clamp(min=1.17e-38) * (1.0 / 7.0))).clamp(
        -127, 127
    )
    normalized = blocks * torch.exp2(-exponent)
    private = torch.floor(torch.log2(normalized.abs().clamp(min=1.17e-38))).clamp(min=0)
    scaled = normalized * torch.exp2(-private) * 2
    rounded = scaled.sign() * torch.floor(scaled.abs() + 0.5)
    values = (rounded * 0.5 * torch.exp2(private)).clamp(-6, 6)
    return (values * torch.exp2(exponent)).reshape(x.shape)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("rows", [0, 1, 65, 129])
def test_packed_matches_training_values(dtype, rows):
    torch.manual_seed(7)
    x = torch.randn(rows, 128, dtype=dtype) * 3
    packed, scales = pack_mxfp4_c7(x)
    assert packed.shape == (rows, 64)
    assert scales.shape == (rows, 4)
    torch.testing.assert_close(unpack(packed, scales, dtype), reference(x), rtol=0, atol=0)


def test_codebook_nibble_order_and_sign():
    x = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6] * 4)
    packed, scales = pack_mxfp4_c7(x)
    assert packed.view(torch.uint8).tolist() == [0x10, 0x32, 0x54, 0x76, 0x90, 0xBA, 0xDC, 0xFE] * 4
    assert scales.tolist() == [127, 127]


def test_bf16_rounding_and_noncontiguous():
    x = torch.zeros(64, 2, dtype=torch.bfloat16).transpose(0, 1)
    x[:, 0], x[:, 1] = 6, 0.2490234375
    packed, scales = pack_mxfp4_c7(x)
    restored = unpack(packed, scales, x.dtype)
    assert restored[:, 1].tolist() == [0.5, 0.5]


@pytest.mark.parametrize("exponent", [-126, -1, 0, 1, 2, 124, 125])
def test_exponent_bytes_do_not_round_in_bf16(exponent):
    x = torch.full((1, 64), 6 * 2.0**exponent, dtype=torch.bfloat16)
    _, scales = pack_mxfp4_c7(x)
    assert scales.tolist() == [[exponent + 127, exponent + 127]]


@pytest.mark.parametrize("x", [torch.ones(32), torch.ones(0), torch.tensor(1.0), torch.ones(64).half()])
def test_reject_unsupported_input(x):
    with pytest.raises(ValueError):
        pack_mxfp4_c7(x)
