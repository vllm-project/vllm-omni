# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real A5 MX kernels, padding, output chunking, and VAE dtype boundaries."""

import pytest
import torch

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.models.minimax_h3.native_mx import native_mx_linear, quantize_mx_weight
from vllm_omni.diffusion.models.minimax_h3.offline_quantization import reconstruct_mx_weight

pytestmark = [pytest.mark.advanced_model, pytest.mark.diffusion, *hardware_marks(res={"npu": "A5"}, num_cards=1)]
torch_npu = pytest.importorskip("torch_npu")


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("shape", [(2, 3, 24), (7, 32), (3, 64)])
@torch.inference_mode()
def test_native_mx_matches_independently_decoded_operands(bits, shape):
    torch.manual_seed(43)
    device = torch.device("npu:0")
    x = torch.randn(shape, dtype=torch.float32, device=device)
    weight = torch.randn(128, shape[-1], dtype=torch.bfloat16)
    bias = torch.randn(128, dtype=torch.float32, device=device)
    q, scale = quantize_mx_weight(weight, bits, device)
    actual = native_mx_linear(x, q, scale, bits, bias)
    padded_k = q.shape[1] * (2 if bits == 4 else 1)
    xp = torch.nn.functional.pad(x.reshape(-1, shape[-1]).bfloat16(), (0, padded_k - shape[-1]))
    dtype = torch_npu.float4_e2m1fn_x2 if bits == 4 else torch_npu.float8_e4m3fn
    xq, xs = torch_npu.npu_dynamic_mx_quant(xp, dst_type=dtype)
    x_reference = reconstruct_mx_weight(xq.cpu(), xs.cpu().reshape(xp.shape[0], -1))
    w_bytes = q.cpu()
    w_native = w_bytes if bits == 4 else w_bytes.view(torch.float8_e4m3fn)
    w_reference = reconstruct_mx_weight(w_native, scale.cpu().flatten(1))
    expected = (x_reference @ w_reference.T).bfloat16().float() + bias.cpu()
    torch.npu.synchronize()
    assert actual.dtype == x.dtype
    assert tuple(actual.shape) == (*shape[:-1], 128)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.cpu().reshape_as(expected), expected, rtol=0.025, atol=0.125)


@pytest.mark.parametrize("bits", [4, 8])
@torch.inference_mode()
def test_native_mx_wide_adaln_stays_low_bit(bits):
    device = torch.device("npu:0")
    weight = torch.ones(96768, 64, dtype=torch.bfloat16)
    q, scale = quantize_mx_weight(weight, bits, device)
    actual = native_mx_linear(torch.ones(2, 64, dtype=torch.bfloat16, device=device), q, scale, bits, None)
    torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu(), torch.full((2, 96768), 64, dtype=torch.bfloat16))


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize(("out_features", "prefix_rows"), [(128, 7), (96768, 0)])
@torch.inference_mode()
def test_native_mx_nonuniform_weights_with_storage_offsets(bits, out_features, prefix_rows):
    # DLO repoints parameters into a shared byte buffer; wide AdaLN also slices
    # output rows. Uniform weights cannot detect reading from the wrong offset.
    torch.manual_seed(113)
    device = torch.device("npu:0")
    x = torch.randn(5, 128, dtype=torch.bfloat16, device=device)
    rows = torch.arange(out_features)
    weight = torch.randn(out_features, 128, dtype=torch.bfloat16)
    weight *= torch.pow(2.0, (rows % 9 - 4).float()).bfloat16().unsqueeze(-1)
    q, scale = quantize_mx_weight(weight, bits, device)
    if prefix_rows:
        q = torch.cat((torch.zeros_like(q[:prefix_rows]), q))[prefix_rows:]
        scale = torch.cat((torch.zeros_like(scale[:prefix_rows]), scale))[prefix_rows:]
        assert q.storage_offset() != 0
        assert q.is_contiguous()  # contiguous() alone does not remove this offset.
    mx_dtype = torch_npu.float4_e2m1fn_x2 if bits == 4 else torch_npu.float8_e4m3fn
    xq, xs = torch_npu.npu_dynamic_mx_quant(x, dst_type=mx_dtype)
    xd = reconstruct_mx_weight(xq.cpu(), xs.cpu().flatten(1))
    weight_bytes = q.cpu()
    wd = reconstruct_mx_weight(
        weight_bytes if bits == 4 else weight_bytes.view(torch.float8_e4m3fn), scale.cpu().flatten(1)
    )
    expected = (xd @ wd.T).bfloat16()
    actual = native_mx_linear(x, q, scale, bits, None)
    torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.025, atol=0.125)
