# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import MethodType, SimpleNamespace

import pytest
import torch
from vllm.config import set_current_vllm_config
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape

from vllm_omni.diffusion.vllm_config import create_base_diffusion_vllm_config
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.cuda, pytest.mark.diffusion]


@pytest.mark.skipif(not current_omni_platform.is_cuda() or not torch.cuda.is_available(), reason="NVIDIA CUDA required")
def test_diffusion_fp8_uses_cuda_activation_quantizer() -> None:
    config = create_base_diffusion_vllm_config(torch.device("cuda"), SimpleNamespace(additional_config={}))
    with set_current_vllm_config(config):
        quantizer = QuantFP8(static=False, group_shape=GroupShape.PER_TOKEN)
        selected = quantizer._forward_method
        assert (selected.__func__ if isinstance(selected, MethodType) else selected) is QuantFP8.forward_cuda
        inputs = torch.randn((16, 128), device="cuda", dtype=torch.bfloat16)
        output, scale = quantizer(inputs)
        reference, reference_scale = quantizer.forward_native(inputs)

    assert output.dtype == torch.float8_e4m3fn
    assert torch.isfinite(output.float()).all()
    assert torch.isfinite(scale).all()
    dequantized = output.float() * scale
    reference_dequantized = reference.float() * reference_scale
    relative_rms = (dequantized - reference_dequantized).norm() / reference_dequantized.norm()
    assert relative_rms < 0.03
