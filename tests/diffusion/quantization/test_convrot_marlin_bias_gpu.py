# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression coverage for biased ConvRot layers on the Marlin fallback."""

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
from vllm.model_executor.kernels.linear.mxfp8.marlin import MarlinMxfp8LinearKernel
from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import Mxfp8LinearLayerConfig
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import is_fp4_marlin_supported
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import is_fp8_marlin_supported
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp4Dynamic
from vllm.utils.network_utils import get_open_port
from vllm.utils.torch_utils import set_default_torch_dtype

import vllm_omni.quantization.convrot_config as convrot_config
from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.data import OmniDiffusionConfig, TransformerConfig
from vllm_omni.diffusion.vllm_config import create_diffusion_vllm_config
from vllm_omni.quantization import build_quant_config

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


def _make_marlin_kernel(quant_method):
    if quant_method == "convrot_mxfp4":
        return MarlinMxFp4LinearKernel(MxFp4LinearLayerConfig(activation_quant_key=kMxfp4Dynamic))
    return MarlinMxfp8LinearKernel(Mxfp8LinearLayerConfig(weight_shape=(768, 1024)))


@pytest.mark.parametrize(
    ("quant_method", "factory_name", "kernel_cls", "is_supported", "max_error"),
    [
        (
            "convrot_mxfp4",
            "init_mxfp4_linear_kernel",
            MarlinMxFp4LinearKernel,
            is_fp4_marlin_supported,
            0.3,
        ),
        (
            "convrot_mxfp8",
            "init_mxfp8_linear_kernel",
            MarlinMxfp8LinearKernel,
            is_fp8_marlin_supported,
            0.1,
        ),
    ],
)
@pytest.mark.parametrize("bias_first", [False, True])
@pytest.mark.parametrize("offload_after_quant", [False, True])
@hardware_test(res={"cuda": ["H100", "H200", "B200"]}, num_cards=1)
def test_marlin_does_not_finalize_before_bias(
    mocker,
    quant_method,
    factory_name,
    kernel_cls,
    is_supported,
    max_error,
    bias_first,
    offload_after_quant,
):
    """A cached bias reference must remain valid through checkpoint load."""
    if not is_supported():
        pytest.skip("Marlin fallback is unavailable")

    config = build_quant_config(quant_method)
    # This regression only needs the native config container, not a checkpoint.
    # Avoid a Hub lookup so the test remains self-contained with HF_HUB_OFFLINE=1.
    od_config = OmniDiffusionConfig(
        model=None,
        dtype=torch.bfloat16,
        tf_model_config=TransformerConfig.from_dict({}),
        quantization_config=config,
    )
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
            kernel = _make_marlin_kernel(quant_method)
            mocker.patch.object(convrot_config, factory_name, return_value=kernel)
            layer = ReplicatedLinear(
                1024,
                768,
                bias=True,
                params_dtype=torch.bfloat16,
                quant_config=config,
                prefix=f"test.{quant_method}",
                disable_tp=True,
            )
            if offload_after_quant:
                layer.quant_method.enable_offload_after_quant()

            generator = torch.Generator(device="cuda").manual_seed(0)
            inputs = torch.randn(4, 1024, generator=generator, dtype=torch.bfloat16)
            weight = torch.randn(768, 1024, generator=generator, dtype=torch.bfloat16) / 32
            bias = torch.randn(768, generator=generator, dtype=torch.bfloat16)

            # H3 caches the parameter mapping before iterating checkpoint
            # entries. Reproduce that so a replaced parameter would be missed.
            params = dict(layer.named_parameters())
            if bias_first:
                params["bias"].weight_loader(params["bias"], bias)

            params["weight"].weight_loader(params["weight"], weight)
            if not bias_first:
                assert not getattr(layer, "_already_called_process_weights_after_loading", False)
                assert layer.bias.data_ptr() == params["bias"].data_ptr()
                params["bias"].weight_loader(params["bias"], bias)

            # Bias-first cannot be proven complete by the weight loader, so the
            # normal post-load sweep must finish it.
            layer.quant_method.process_weights_after_loading(layer)

            expected_device = "cpu" if offload_after_quant else "cuda"
            assert all(parameter.device.type == expected_device for parameter in layer.parameters())
            if offload_after_quant:
                layer.to("cuda:0")

            assert isinstance(layer.quant_method.kernel, kernel_cls)
            actual = layer(inputs)[0]
            expected = F.linear(inputs, weight, bias)
            error = ((actual.float() - expected.float()).norm() / expected.float().norm()).item()
            assert error < max_error
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()
