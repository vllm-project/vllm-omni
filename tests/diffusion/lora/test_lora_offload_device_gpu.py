# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""A LoRA layer wrapped while its base weights are offloaded to CPU must still run on the GPU."""

import pytest
import torch
from vllm.config import set_current_vllm_config
from vllm.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.lora.peft_helper import PEFTHelper
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.utils.network_utils import get_open_port
from vllm.utils.torch_utils import set_default_torch_dtype

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager
from vllm_omni.diffusion.vllm_config import create_diffusion_vllm_config

pytestmark = [pytest.mark.advanced_model, pytest.mark.diffusion]


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_lora_applies_when_layers_are_wrapped_while_base_weights_are_on_cpu():
    cuda = torch.device("cuda:0")
    od_config = OmniDiffusionConfig(model="test", dtype=torch.bfloat16)
    vllm_config = create_diffusion_vllm_config(cuda, od_config)

    try:
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
            backend="nccl",
        )
        with set_current_vllm_config(vllm_config), set_default_torch_dtype(torch.bfloat16), torch.no_grad():
            initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
            pipeline = torch.nn.Module()
            pipeline.transformer = torch.nn.Module()
            pipeline.transformer.proj = ReplicatedLinear(
                64, 64, bias=False, params_dtype=torch.bfloat16, disable_tp=True, prefix="proj", return_bias=False
            )
            generator = torch.Generator().manual_seed(0)
            pipeline.transformer.proj.weight.data.copy_(torch.randn(64, 64, generator=generator))
            pipeline.transformer.proj.to(cuda)
            inputs = torch.randn(8, 64, generator=generator).to(device=cuda, dtype=torch.bfloat16)
            base = pipeline.transformer.proj(inputs).float()

            # Startup `lora_path`: layers are wrapped before the first forward, while offload keeps
            # the base weights on CPU. The first forward later brings them to the GPU.
            pipeline.transformer.proj.to("cpu")
            manager = DiffusionLoRAManager(pipeline=pipeline, device=cuda, dtype=torch.bfloat16)
            manager._replace_layers_with_lora(PEFTHelper(r=8, lora_alpha=8, target_modules=["proj"]))
            wrapper = manager._lora_modules["transformer.proj"]
            lora_a = (torch.randn(8, 64, generator=generator) * 0.1).to(torch.bfloat16)
            lora_b = (torch.randn(64, 8, generator=generator) * 0.1).to(torch.bfloat16)
            wrapper.set_lora(0, lora_a, lora_b)
            pipeline.transformer.to(cuda)

            actual = wrapper.apply(inputs).float()

            expected = base + (inputs.float() @ lora_a.float().t().to(cuda)) @ lora_b.float().t().to(cuda)
            assert wrapper.lora_a_stacked[0].device == cuda
            torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()
