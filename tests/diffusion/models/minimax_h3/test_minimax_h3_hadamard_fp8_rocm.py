# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Synthetic gfx950 checks; no checkpoint is required and no speed is asserted."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from torch import nn
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.triton_utils import tl, triton

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.models.minimax_h3 import hadamard_fp8 as h64
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, *hardware_marks(res={"rocm": "MI350X"}, num_cards=1)]


def _device() -> torch.device:
    if not current_omni_platform.is_rocm() or not current_omni_platform.is_available():
        pytest.skip("requires a ROCm gfx950 GPU")
    device = current_omni_platform.get_torch_device()
    capability = current_omni_platform.get_device_capability(device.index or 0)
    if capability is None or capability.to_int() != 95:
        pytest.skip("requires gfx950")
    return device


@triton.jit
def _quantize_reference(x_ptr, q_ptr, s_ptr, k: tl.constexpr, block: tl.constexpr):
    # This reference receives a separately materialized Torch FP32 rotation.
    columns = tl.arange(0, block)
    row = tl.program_id(0)
    values = tl.load(x_ptr + row * k + columns, columns < k, other=0)
    maximum = tl.max(tl.abs(values), 0)
    scale = tl.where(maximum > 0, maximum * 0.0022321429569274187, 1.0)
    inverse = tl.inline_asm_elementwise("v_rcp_f32 $0, $1;", "=v,v", [scale], dtype=tl.float32, is_pure=True, pack=1)
    scaled = values * inverse
    clamped = tl.inline_asm_elementwise(
        "v_med3_f32 $0, $1, $2, $3;",
        "=v,v,v,v",
        [scaled, tl.full((block,), -448.0, tl.float32), tl.full((block,), 448.0, tl.float32)],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    tl.store(q_ptr + row * k + columns, clamped.to(tl.float8e4nv, fp_downcast_rounding="rtne"), columns < k)
    tl.store(s_ptr + row, scale)


@pytest.mark.parametrize("rows", [0, 1, 17, 129])
def test_fused_rotation_matches_fp32_reference_and_zero_rows(rows):
    device = _device()
    generator = torch.Generator(device=device).manual_seed(171117)
    x = torch.randn((rows, 14336), device=device, dtype=torch.bfloat16, generator=generator)
    signs = h64._signs("blocks.0.mlp.fc2").to(device)
    if rows:
        x[0].zero_()
    if rows > 1:
        x[1].fill_(0.25)
        x[2].zero_()
        x[2, 1031] = 128
    q, scales = h64.hadamard64_quant(x, signs)
    assert q.shape == x.shape and q.dtype == torch.float8_e4m3fn
    assert scales.shape == (rows, 1) and scales.dtype == torch.float32
    if not rows:
        return
    rotated = h64._rotate_rows(x, signs).contiguous()
    reference_q = torch.empty_like(q)
    reference_s = torch.empty_like(scales)
    _quantize_reference[(rows,)](rotated, reference_q, reference_s, 14336, 16384, num_warps=8, enable_fp_fusion=False)
    assert torch.equal(q.view(torch.uint8), reference_q.view(torch.uint8))
    assert torch.equal(scales.view(torch.int32), reference_s.view(torch.int32))
    assert torch.count_nonzero(q[0].float()) == 0
    assert scales[0].item() == 1.0
    assert bool(torch.isfinite(q.float()).all() and torch.isfinite(scales).all() and (scales > 0).all())


def test_hadamard_quantization_graph_replay():
    device = _device()
    x = torch.ones((17, 14336), dtype=torch.bfloat16, device=device)
    signs = h64._signs("blocks.0.mlp.fc2").to(device)
    outputs: list[torch.Tensor] = []

    def run() -> None:
        outputs[:] = h64.hadamard64_quant(x, signs)

    for _ in range(3):
        run()
    current_omni_platform.synchronize()
    graph = current_omni_platform.capture_graph(run)
    assert graph is not None
    x.zero_()
    current_omni_platform.replay_graph(graph, run)
    assert torch.count_nonzero(outputs[0].float()) == 0
    assert torch.equal(outputs[1], torch.ones_like(outputs[1]))


def test_native_postload_retains_one_rowwise_fp8_gemm():
    from vllm.model_executor.kernels.linear import init_fp8_linear_kernel
    from vllm.model_executor.kernels.linear.scaled_mm.pytorch import RowWiseTorchFP8ScaledMMLinearKernel
    from vllm.model_executor.layers.quantization.online.fp8 import (
        Fp8PtpcOnlineLinearMethod,
        _fp8_channel_scale,
        _fp8_quant_per_channel,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import kFp8DynamicTokenSym, kFp8StaticChannelSym

    device = _device()
    generator = torch.Generator(device=device).manual_seed(171117)
    source = torch.randn((5376, 14336), dtype=torch.bfloat16, device=device, generator=generator)
    layer = nn.Module()
    layer.tp_size = 1
    layer.weight = nn.Parameter(source, requires_grad=False)
    # Isolate the native post-load/apply contract from full-model construction.
    method = object.__new__(Fp8PtpcOnlineLinearMethod)
    method.requantization_source = None
    method.requantization_source_parameters = {}
    with set_current_vllm_config(VllmConfig()):
        kernel = init_fp8_linear_kernel(
            activation_quant_key=kFp8DynamicTokenSym,
            weight_quant_key=kFp8StaticChannelSym,
            weight_shape=(5376, 14336),
            input_dtype=torch.bfloat16,
            out_dtype=torch.bfloat16,
            module_name="MiniMaxH3HadamardTest",
        )
    assert type(kernel) is RowWiseTorchFP8ScaledMMLinearKernel
    method.fp8_linear = kernel
    layer.quant_method = method
    h64._install_linear("blocks.0.mlp.fc2", layer)
    method.process_weights_after_loading(layer)
    assert layer._already_called_process_weights_after_loading
    assert layer.weight.dtype == torch.float8_e4m3fn
    assert layer.weight.shape == (14336, 5376) and layer.weight.stride(0) == 1
    assert method.fp8_linear is kernel and isinstance(kernel.quant_fp8, h64.Hadamard64QuantFP8)
    signs = h64._signs("blocks.0.mlp.fc2").to(device)
    for selected in (slice(0, 4), slice(-4, None)):
        rotated = h64._rotate_rows(source[selected], signs)
        reference_scale = _fp8_channel_scale(rotated.abs().amax(-1, keepdim=True))
        reference_weight = _fp8_quant_per_channel(rotated, reference_scale)
        assert torch.equal(layer.weight.t()[selected].view(torch.uint8), reference_weight.view(torch.uint8))
        assert torch.equal(layer.weight_scale[selected], reference_scale)
    x = torch.randn((17, 14336), dtype=torch.bfloat16, device=device, generator=generator)
    with patch.object(torch, "_scaled_mm", wraps=torch._scaled_mm) as scaled_mm:
        output = method.apply(layer, x)
    assert scaled_mm.call_count == 1
    assert output.dtype == torch.bfloat16 and output.shape == (17, 5376)
    assert bool(torch.isfinite(output).all())
    saved_weight, saved_quantizer = layer.weight, kernel.quant_fp8
    method.process_weights_after_loading(layer)
    assert layer.weight is saved_weight and kernel.quant_fp8 is saved_quantizer
