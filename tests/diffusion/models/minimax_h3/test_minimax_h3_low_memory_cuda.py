# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from vllm.platforms import current_platform

from vllm_omni.quantization.svdquant_config import (
    DiffusionSVDQuantConfig,
    DiffusionSVDQuantLinearMethod,
    _nvfp4_kernel,
    _supports_capability,
)
from vllm_omni.quantization.tools.export_minimax_h3_low_memory_checkpoint import pack_nvfp4

pytestmark = [pytest.mark.local_model, pytest.mark.cuda, pytest.mark.diffusion]


@pytest.mark.parametrize("input_size,output_size", [(256, 128), (5376, 21504), (14336, 5376)])
@pytest.mark.parametrize("activation_bits", [4, 16])
@pytest.mark.parametrize("rows", [17, 32768])
def test_native_svdquant_and_weight_only_reference_on_production_shapes(input_size, output_size, activation_bits, rows):
    if not torch.accelerator.is_available():
        pytest.skip("CUDA accelerator required")
    if not current_platform.is_cuda() or not _supports_capability(current_platform.get_device_capability()):
        pytest.skip("SVDQuant NVFP4 requires SM100, SM103 or SM120")
    device = torch.device("cuda", torch.accelerator.current_device_index())
    generator = torch.Generator(device=device).manual_seed(6493)
    grid = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=device)
    weight = grid[torch.randint(0, 8, (output_size, input_size), generator=generator, device=device)]
    weight *= torch.randint(0, 2, weight.shape, generator=generator, device=device) * 2 - 1
    weight[:, 15::16] = 6
    packed, scales, outer = pack_nvfp4(weight)
    unsigned = packed.view(torch.uint8).long()
    codes = torch.stack((unsigned & 15, unsigned >> 4), dim=-1).reshape(output_size, input_size)
    signed_grid = torch.cat((grid, -grid))
    dense = (
        (signed_grid[codes].reshape(output_size, input_size // 16, 16) * scales.T.float().unsqueeze(-1) * outer.float())
        .reshape(output_size, input_size)
        .bfloat16()
    )
    del weight, codes, unsigned
    method = DiffusionSVDQuantLinearMethod(DiffusionSVDQuantConfig(activation_bits=activation_bits))
    with device:
        layer = torch.nn.Module()
        method.create_weights(layer, input_size, [output_size], input_size, output_size, torch.bfloat16)
    layer.qweight.data.copy_(packed)
    layer.wscales.data.copy_(scales)
    layer.wtscale.data.copy_(outer)
    layer.smooth_factor.data.fill_(1)
    layer.wcscales.data.fill_(1)
    layer.proj_down.data.copy_(torch.randn(input_size, 32, generator=generator, device=device).bfloat16() * 0.01)
    layer.proj_up.data.copy_(torch.randn(output_size, 32, generator=generator, device=device).bfloat16() * 0.01)
    inputs = grid[torch.randint(0, 8, (rows, input_size), generator=generator, device=device)]
    inputs *= torch.randint(0, 2, inputs.shape, generator=generator, device=device) * 2 - 1
    inputs[:, 15::16] = 6
    inputs = inputs.bfloat16()
    expected = torch.addmm(torch.nn.functional.linear(inputs, dense), inputs @ layer.proj_down, layer.proj_up.T)
    method.process_weights_after_loading(layer)
    actual = method.apply(layer, inputs)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=2.0)
    relative_error = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert relative_error.item() < 0.01
    print(
        {
            "activation_bits": activation_bits,
            "shape": [rows, input_size, output_size],
            "relative_error": relative_error.item(),
            "backend": type(_nvfp4_kernel()).__name__ if activation_bits == 4 else "reference-dequant-bf16",
        }
    )
