# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from tests.helpers.mark import hardware_marks
from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4Config, NPUMxfp4OnlineLinearMethod
from vllm_omni.quantization.mxfp4_fake_config import fake_quantize_mxfp4

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, *hardware_marks(res={"npu": "A5"}, num_cards=1)]


@pytest.mark.parametrize("kind", ["ties", "random", "3d"])
def test_c7_native_matmul_matches_qdq(kind):
    import torch_npu  # noqa: F401

    torch.npu.set_device(0)
    torch.manual_seed(42)
    if kind == "ties":
        weight = torch.zeros(64, 64, device="npu", dtype=torch.bfloat16)
        weight[:, 0], weight[:, 1] = 6, 0.2490234375
        x = weight[:2].clone()
    else:
        weight = torch.randn(64, 128, device="npu", dtype=torch.bfloat16)
        x = torch.randn((2, 3, 128) if kind == "3d" else (5, 128), device="npu", dtype=torch.bfloat16)
    layer = torch.nn.Linear(weight.shape[1], 64, bias=False, device="npu", dtype=weight.dtype).requires_grad_(False)
    layer.weight.copy_(weight)
    method = NPUMxfp4OnlineLinearMethod(DiffusionMXFP4Config(mxfp4_recipe="mxfp4_c7"))
    method.process_weights_after_loading(layer)
    result = method.apply(layer, x)
    expected = torch.nn.functional.linear(
        fake_quantize_mxfp4(x, "mxfp4_c7").float(), fake_quantize_mxfp4(weight, "mxfp4_c7").float()
    ).to(x.dtype)
    torch.testing.assert_close(result, expected, rtol=0.02, atol=0.0625)
    assert layer.weight.element_size() * layer.weight.numel() == weight.numel() // 2
    if kind == "ties":
        assert result[0, 0].item() == 36.25


@pytest.mark.parametrize("rows", [1023, 1024, 1025])
def test_c7_batched_encoding_preserves_training_values(rows):
    import torch_npu  # noqa: F401

    from tests.diffusion.quantization.test_mxfp4_c7_packed import unpack
    from vllm_omni.quantization.mxfp4_c7 import pack_mxfp4_c7

    torch.npu.set_device(0)
    torch.manual_seed(17)
    x = torch.randn(rows, 128, dtype=torch.bfloat16, device="npu")
    x[:, 0], x[:, 1] = 6, 0.2490234375
    packed, scales = pack_mxfp4_c7(x)
    actual = unpack(packed.cpu(), scales.cpu(), x.dtype)
    expected = fake_quantize_mxfp4(x, "mxfp4_c7").cpu()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
