# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from tests.helpers.mark import hardware_marks

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@pytest.mark.cpu
def test_h3_mxfp8_scope_preserves_conditioning(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3.quantization import projection_quant_config
    from vllm_omni.platforms import current_omni_platform
    from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config

    monkeypatch.setattr(current_omni_platform, "is_cuda", lambda: True)
    config = DiffusionMXFP8Config(ignored_layers=["blocks.1.attn.qkv_proj"])
    for prefix in ("blocks.0.attn.qkv_proj", "blocks.1.attn.out_proj", "blocks.2.mlp.fc1", "blocks.49.mlp.fc2"):
        assert projection_quant_config(config, prefix) is config
    for prefix in (
        "condition_proj",
        "blocks.0.adaln_proj.linear",
        "blocks.0.attn.to_gate_compress",
        "token_refiner.blocks.0.mlp.fc1",
        "final_layer.adaln_proj.linear",
    ):
        assert projection_quant_config(config, prefix) is None
    assert config.ignored_layers == ["blocks.1.attn.qkv_proj"]
    ordinary = object()
    assert projection_quant_config(ordinary, "condition_proj") is ordinary


@pytest.fixture
def cuda_device():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (10, 12):
        pytest.skip("requires Blackwell")
    if not hasattr(torch.nn.functional, "scaled_mm"):
        pytest.skip("requires public block-scaled GEMM API")
    return torch.device("cuda")


@pytest.mark.cuda
@pytest.mark.parametrize("_hardware", [pytest.param(None, marks=hardware_marks(res={"cuda": ["B200"]}, num_cards=1))])
@pytest.mark.parametrize("rows,hidden", [(1, 32), (127, 96), (137, 128), (256, 512), (11992, 14336)])
def test_swiglu_matches_vllm_bytes(cuda_device, rows, hidden, _hardware):
    from vllm_omni.diffusion.layers.activation import SiluAndMul
    from vllm_omni.diffusion.layers.mxfp8 import mxfp8_quantize_swizzled, silu_mul_mxfp8_cuda

    generator = torch.Generator(device=cuda_device).manual_seed(1101)
    x = torch.randn(rows, 2 * hidden, dtype=torch.bfloat16, device=cuda_device, generator=generator)
    reference = SiluAndMul().forward_cuda(x)
    expected_q, expected_scale = mxfp8_quantize_swizzled(reference)
    q, scale, activation = silu_mul_mxfp8_cuda(x, return_bf16=True)
    assert torch.equal(activation.view(torch.uint8), reference.view(torch.uint8))
    assert torch.equal(q.view(torch.uint8), expected_q.view(torch.uint8))
    assert torch.equal(scale, expected_scale)


@pytest.mark.cuda
@pytest.mark.parametrize("_hardware", [pytest.param(None, marks=hardware_marks(res={"cuda": ["B200"]}, num_cards=1))])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rows", [0, 1, 137])
def test_native_gemm_shape_and_accuracy(cuda_device, dtype, rows, _hardware):
    from vllm_omni.quantization.mxfp8_config import CUDAMxfp8OnlineLinearMethod

    torch.manual_seed(71)
    layer = torch.nn.Linear(128, 64, bias=False, device=cuda_device, dtype=dtype)
    weight = layer.weight.detach().clone()
    method = CUDAMxfp8OnlineLinearMethod()
    method.process_weights_after_loading(layer)
    x = torch.randn(2, rows, 128, device=cuda_device, dtype=dtype)
    actual = method.apply(layer, x)
    assert actual.shape == (2, rows, 64)
    assert actual.dtype == dtype
    if rows:
        expected = torch.nn.functional.linear(x.float(), weight.float())
        relative_rms = (actual.float() - expected).square().mean().sqrt() / expected.square().mean().sqrt()
        assert relative_rms.item() < 0.06
        assert torch.isfinite(actual).all()
