# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise online FP8 loading and inference through the native CUDA path."""

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
from vllm.model_executor.layers.linear import ReplicatedLinear, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.online.fp8 import Fp8PerTensorOnlineLinearMethod
from vllm.utils.network_utils import get_open_port
from vllm.utils.torch_utils import set_default_torch_dtype

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.vllm_config import create_diffusion_vllm_config
from vllm_omni.quantization import build_quant_config

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@pytest.mark.parametrize("method", ["fp8", "fp8_per_tensor"])
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_online_fp8_loads_bf16_weights_and_runs_native_kernel(method):
    ignored_key = "ignored_layers" if method == "fp8" else "ignore"
    config = build_quant_config(method, **{ignored_key: ["skip"]})
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
            layer = ReplicatedLinear(
                128, 128, bias=False, params_dtype=torch.bfloat16, quant_config=config, prefix="linear", disable_tp=True
            )
            assert isinstance(layer.quant_method, Fp8PerTensorOnlineLinearMethod)
            assert layer.weight.is_meta

            generator = torch.Generator(device="cuda").manual_seed(42)
            weights = torch.randn(128, 128, generator=generator, dtype=torch.bfloat16) / 128**0.5
            inputs = torch.randn(32, 128, generator=generator, dtype=torch.bfloat16)
            expected = F.linear(inputs, weights).float()

            # This is the loader installed by the native online quantizer.
            # It materializes and quantizes the BF16 checkpoint in one pass.
            layer.weight.weight_loader(layer.weight, weights)
            assert layer.weight.dtype == torch.float8_e4m3fn
            actual = layer(inputs)[0].float()
            relative_mse = (actual - expected).square().mean() / expected.square().mean()
            assert relative_mse.item() < 0.002

            ignored = ReplicatedLinear(
                128, 128, bias=False, params_dtype=torch.bfloat16, quant_config=config, prefix="skip", disable_tp=True
            )
            assert isinstance(ignored.quant_method, UnquantizedLinearMethod)
            assert ignored.weight.dtype == torch.bfloat16
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()
