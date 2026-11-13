# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Real mixed-checkpoint loading and MAPS dispatch through NVFP4 emulation."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tests.helpers.mark import hardware_marks

pytestmark = [
    pytest.mark.full_model,
    pytest.mark.diffusion,
    *hardware_marks(res={"cuda": ["H100", "B200"]}, num_cards=1),
]


@pytest.fixture
def emulation_config(tmp_path):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import cleanup_dist_env_and_memory, init_distributed_environment, initialize_model_parallel

    from vllm_omni.platforms import current_omni_platform

    current_omni_platform.set_device(torch.device("cuda:0"))
    init_distributed_environment(
        world_size=1, rank=0, local_rank=0, distributed_init_method=f"file://{tmp_path / 'rendezvous'}"
    )
    config = VllmConfig()
    config.kernel_config.linear_backend = "emulation"
    try:
        with set_current_vllm_config(config):
            initialize_model_parallel(tensor_model_parallel_size=1)
            config.model_config = SimpleNamespace(dtype=torch.bfloat16)
            yield config
    finally:
        cleanup_dist_env_and_memory()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA GPU")
def test_modelopt_mixed_nvfp4_emulation_maps(emulation_config) -> None:
    from vllm.model_executor.kernels.linear.nvfp4.emulation import EmulationNvFp4LinearKernel
    from vllm.model_executor.layers.linear import ReplicatedLinear, UnquantizedLinearMethod
    from vllm.model_executor.layers.quantization.modelopt import ModelOptMixedPrecisionConfig
    from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import ref_nvfp4_quant_dequant

    from vllm_omni.diffusion.models.cosmos3.mixed_precision import (
        Cosmos3MixedPrecisionConfig,
        Cosmos3MixedPrecisionRuntime,
    )

    quant = ModelOptMixedPrecisionConfig.from_config(
        {
            "quant_method": "modelopt",
            "quant_algo": "MIXED_PRECISION",
            "group_size": 16,
            "exclude_modules": [],
            "quantized_layers": {"gen_layers.0": {"quant_algo": "NVFP4"}},
        }
    )
    with torch.device("cuda"), torch.inference_mode():
        layers = [
            ReplicatedLinear(
                64, 128, quant_config=quant, params_dtype=torch.bfloat16, prefix=f"gen_layers.{i}", disable_tp=True
            )
            for i in range(2)
        ]
        assert isinstance(layers[1].quant_method, UnquantizedLinearMethod)
        layer = layers[0]
        transformer = torch.nn.Module()
        transformer.gen_layers = torch.nn.ModuleList(layers)
        runtime = Cosmos3MixedPrecisionRuntime(Cosmos3MixedPrecisionConfig(reasoner="native"))
        runtime.install(transformer)
        assert isinstance(layer.quant_method.base_method.kernel, EmulationNvFp4LinearKernel)

        # E2M1 nibble 2 = +1, nibble A = -1. Vary block scales so an incorrect
        # assumption that these row-major scales are swizzled cannot pass.
        packed = torch.full((128, 32), 0xA2, dtype=torch.uint8)
        scales = (torch.arange(512).reshape(128, 4) % 7 + 1).to(torch.float8_e4m3fn)
        global_scale = torch.tensor([0.125], dtype=torch.float32)
        input_scale = torch.tensor([1 / 448], dtype=torch.float32)
        bias = (torch.arange(128).float() / 128).bfloat16()
        for name, value in (
            ("weight", packed),
            ("weight_scale", scales),
            ("weight_scale_2", global_scale),
            ("input_scale", input_scale),
            ("bias", bias),
        ):
            param = getattr(layer, name)
            param.weight_loader(param, value)
        weight_ptr = layer.weight.data_ptr()
        layer.quant_method.process_weights_after_loading(layer)
        assert layer.weight.data_ptr() == weight_ptr
        assert not layer._cosmos3_nvfp4_scale_swizzled

        signs = torch.tensor([1, -1], dtype=torch.float32).repeat(32)
        dense = (scales.float().repeat_interleave(16, dim=1) * global_scale * signs).bfloat16()
        torch.testing.assert_close(layer.quant_method.strategy.materialize(layer), dense, rtol=0, atol=0)
        x = torch.linspace(-1.7, 2.3, 3 * 64, dtype=torch.float32).reshape(1, 3, 64).bfloat16()
        quantized_x = ref_nvfp4_quant_dequant(x.reshape(-1, 64), input_scale.reciprocal(), block_size=16).reshape_as(x)
        high_reference = F.linear(x, dense) + bias
        native_reference = F.linear(quantized_x, dense) + bias
        assert not torch.allclose(high_reference, native_reference, rtol=0.01, atol=0.01)

        for _ in range(2):
            for step in range(10):
                runtime.set_step(step, 10)
                actual, _ = layer(x)
                expected = high_reference if step < 3 or step >= 7 else native_reference
                torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.01)
            runtime.reset()
            assert not runtime.use_high_precision("generation")
