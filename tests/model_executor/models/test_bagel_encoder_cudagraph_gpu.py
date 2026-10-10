# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from unittest.mock import Mock

import pytest
import torch
from torch import nn
from transformers import SiglipVisionConfig
from vllm.config import CompilationConfig, ModelConfig, MultiModalConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import cleanup_dist_env_and_memory, init_distributed_environment, initialize_model_parallel
from vllm.model_executor.models.bagel import BagelVisionMLP, PositionEmbedding
from vllm.model_executor.models.siglip import SiglipVisionModel
from vllm.transformers_utils.configs.bagel import BagelConfig
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.worker.encoder_cudagraph import EncoderCudaGraphManager

from vllm_omni.model_executor.models.bagel.bagel import OmniBagelForConditionalGeneration
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.cuda]


@pytest.fixture(params=["cuda", "cpu"])
def encoder(tmp_path, request):
    if not current_omni_platform.is_cuda():
        pytest.skip("CUDA graph test requires CUDA")
    init_distributed_environment(world_size=1, rank=0, distributed_init_method=f"file://{tmp_path}/dist", local_rank=0)
    initialize_model_parallel(tensor_model_parallel_size=1)
    try:
        with set_current_vllm_config(VllmConfig()), set_default_torch_dtype(torch.bfloat16):
            model = OmniBagelForConditionalGeneration.__new__(OmniBagelForConditionalGeneration)
            nn.Module.__init__(model)
            config = SiglipVisionConfig(
                image_size=28,
                patch_size=14,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=2,
                num_attention_heads=4,
            )
            model.config = BagelConfig(vit_config=config, llm_config=dict(hidden_size=64), vit_max_num_patch_per_side=2)
            model.vit_model = SiglipVisionModel(config)
            model.connector = BagelVisionMLP(64, 64, 64)
            model.vit_pos_embed = PositionEmbedding(2, 64)
            model.to(device=current_omni_platform.get_torch_device(), dtype=torch.bfloat16).eval()
            if request.param == "cpu":
                # Full-model loading retains this non-persistent buffer on
                # CPU in float32; moving the whole fixture hid that case.
                model.vit_pos_embed = PositionEmbedding(2, 64)
            # vLLM linear layers allocate empty weights for checkpoint loading.
            generator = torch.Generator(device=model.vit_model.device).manual_seed(17)
            with torch.no_grad():
                for param in model.parameters():
                    param.normal_(0, 0.02, generator=generator)
            yield model
    finally:
        cleanup_dist_env_and_memory()


@pytest.mark.parametrize("budgets", [[4, 8], [2]])
@torch.inference_mode()
def test_manager_replay_and_eager_fallback(encoder, budgets):
    config = VllmConfig()
    config.compilation_config = CompilationConfig(
        encoder_cudagraph_token_budgets=budgets,
        encoder_cudagraph_max_vision_items_per_batch=2,
    )
    config.model_config = Mock(spec=ModelConfig)
    config.model_config.multimodal_config = MultiModalConfig(limit_per_prompt={"video": 0})
    manager = EncoderCudaGraphManager(config, encoder.vit_model.device, torch.bfloat16, encoder)
    try:
        manager.capture(torch.cuda.graph_pool_handle())
        graph_count = manager.get_cumulative_stats()["num_budgets"]
        previous = None
        previous_expected = None
        for count in [1, 2, 1, 3, 2]:
            pixels = torch.randn(count, 3, 28, 28, device=encoder.vit_model.device, dtype=torch.bfloat16)
            expected = encoder._process_image_input({"pixel_values": pixels})
            actual = manager.execute({"pixel_values": pixels})
            for a, b in zip(actual, expected, strict=True):
                torch.testing.assert_close(a, b, rtol=0.01, atol=0.01)
            if previous is not None:
                for a, b in zip(previous, previous_expected, strict=True):
                    torch.testing.assert_close(a, b, rtol=0.01, atol=0.01)
            previous, previous_expected = actual, expected
        stats = manager.get_cumulative_stats()
        assert stats["num_budgets"] == graph_count
        assert stats["graph_misses" if budgets == [2] else "graph_hits"] == 9
        assert stats["graph_hits" if budgets == [2] else "graph_misses"] == 0
    finally:
        manager.clear()
