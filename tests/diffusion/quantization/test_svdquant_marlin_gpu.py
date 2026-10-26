# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression coverage for SVDQuant MXFP4 loading through Marlin."""

import pytest
import torch
from torch.nn import functional as F
from vllm.config import set_current_vllm_config
from vllm.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.kernels.linear.mxfp4.base import MxFp4LinearLayerConfig
from vllm.model_executor.kernels.linear.mxfp4.marlin import MarlinMxFp4LinearKernel
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import is_fp4_marlin_supported
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp4Dynamic
from vllm.utils.network_utils import get_open_port
from vllm.utils.torch_utils import set_default_torch_dtype

import vllm_omni.quantization.svdquant_config as svdquant
from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.vllm_config import create_diffusion_vllm_config
from vllm_omni.quantization.svdquant_config import DiffusionSVDQuantConfig

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@hardware_test(res={"cuda": ["B200"]}, num_cards=1)
def test_mxfp4_replicated_linear_loads_with_forced_marlin(mocker):
    """Marlin padding must not break replicated projection loading or bias."""
    if not is_fp4_marlin_supported():
        pytest.skip("Marlin FP4 is unavailable")

    config = DiffusionSVDQuantConfig(rank=8, precision="mxfp4")
    od_config = OmniDiffusionConfig(model="test", dtype=torch.bfloat16, quantization_config=config)
    vllm_config = create_diffusion_vllm_config(torch.device("cuda:0"), od_config)

    try:
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
            backend="nccl",
        )
        with set_current_vllm_config(vllm_config), set_default_torch_dtype(torch.bfloat16), torch.device("cuda:0"):
            initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
            kernel = MarlinMxFp4LinearKernel(MxFp4LinearLayerConfig(activation_quant_key=kMxfp4Dynamic))
            mocker.patch.object(svdquant, "_mxfp4_kernel", return_value=kernel)

            # N=96 forces Marlin to pad the output dimension to 128.
            layer = ReplicatedLinear(
                128,
                96,
                bias=True,
                params_dtype=torch.bfloat16,
                quant_config=config,
                prefix="test.proj_mlp",
                disable_tp=True,
            )
            assert layer.input_size_per_partition == 128

            generator = torch.Generator(device="cuda").manual_seed(0)
            inputs = torch.randn(4, 128, generator=generator, dtype=torch.bfloat16)
            bias = torch.randn(96, generator=generator, dtype=torch.bfloat16)
            tensors = {
                # Byte 0x22 encodes two E2M1 +1.0 values; E8M0 127 is 2**0.
                "qweight": torch.full((96, 64), 0x22, device="cuda", dtype=torch.uint8).view(torch.int8),
                "wscales": torch.full((4, 96), 127, device="cuda", dtype=torch.uint8),
                "proj_down": torch.zeros((128, 8), device="cuda", dtype=torch.bfloat16),
                "proj_up": torch.zeros((96, 8), device="cuda", dtype=torch.bfloat16),
                "smooth_factor": torch.ones(128, device="cuda", dtype=torch.bfloat16),
                "bias": bias,
            }
            for name, value in tensors.items():
                param = getattr(layer, name)
                param.weight_loader(param, value)

            layer.quant_method.process_weights_after_loading(layer)
            assert layer.bias.shape == (128,)

            actual = layer(inputs)[0]
            expected = F.linear(inputs, torch.ones((96, 128), device="cuda", dtype=torch.bfloat16), bias)
            torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()
