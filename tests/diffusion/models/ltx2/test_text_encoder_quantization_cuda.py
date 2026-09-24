# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise CPU-loaded Gemma3 weights through real CUDA FP8 conversion."""

from types import SimpleNamespace

import pytest
import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization.fp8 import Fp8Config

from tests.diffusion.models.ltx2.test_text_encoder_quantization import encoder  # noqa: F401
from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.models.ltx2.quantization import prepare_gemma3_fp8
from vllm_omni.platforms import current_omni_platform
from vllm_omni.quantization import ComponentQuantizationConfig

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.diffusion,
    pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="requires CUDA FP8"),
]


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_cpu_loaded_encoder_quantizes_on_cuda_and_restores_placement(encoder, tmp_path):  # noqa: F811
    config = VllmConfig()
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=(tmp_path / "dist").as_uri(),
        )
        initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    config.model_config = SimpleNamespace(dtype=torch.bfloat16, quantization=None)
    try:
        with set_current_vllm_config(config), torch.inference_mode():
            original_head = encoder.lm_head.weight
            count = prepare_gemma3_fp8(
                encoder, ComponentQuantizationConfig({"text_encoder": Fp8Config()}), torch.device("cuda")
            )
            assert count == 14
            assert all(parameter.device.type == "cpu" for parameter in encoder.parameters())
            assert encoder.lm_head.weight is original_head
            encoder.to("cuda")
            for module in encoder.modules():
                if isinstance(module, ReplicatedLinear):
                    module.quant_method.process_weights_after_loading(module)
                    assert module.weight.dtype == torch.float8_e4m3fn
            ids = torch.tensor([[1, 2, 3, 0]], device="cuda")
            outputs = encoder(ids, attention_mask=ids.ne(0), output_hidden_states=True).hidden_states
            assert len(outputs) == 3
            assert all(torch.isfinite(hidden).all() for hidden in outputs)
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()
