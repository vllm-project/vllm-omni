# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.utils.network_utils import get_file_store_init_method

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.cache.teacache.extractors import extract_sensenova_u1_context
from vllm_omni.diffusion.models.sensenova_u1.pipeline_sensenova_u1 import SenseNovaU1DenoisingAdapter
from vllm_omni.diffusion.models.sensenova_u1.sensenova_u1_transformer import SenseNovaU1ForCausalLM
from vllm_omni.transformers_utils.configs.sensenova_u1 import SenseNovaU1Config

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@pytest.fixture
def tiny_model():
    init_distributed_environment(
        world_size=1, rank=0, local_rank=0, distributed_init_method=get_file_store_init_method()
    )
    initialize_model_parallel()
    try:
        config = SenseNovaU1Config(
            llm_config={
                "hidden_size": 64,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "num_hidden_layers": 2,
                "intermediate_size": 128,
                "vocab_size": 32,
                "max_position_embeddings": 128,
                "max_position_embeddings_hw": 128,
            }
        )
        with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cuda"))):
            model = SenseNovaU1ForCausalLM(config.llm_config).to(device="cuda", dtype=torch.bfloat16)
            with torch.no_grad():
                torch.manual_seed(42)
                for parameter in model.parameters():
                    parameter.normal_(0, 0.05)
            yield model
    finally:
        cleanup_dist_env_and_memory()


@pytest.mark.parametrize("prefix_len", [0, 3])
@pytest.mark.parametrize("masked", [False, True])
@hardware_test(res={"cuda": "L4"})
@torch.inference_mode()
def test_teacache_matches_real_decoder(tiny_model, prefix_len, masked):
    past = None
    if prefix_len:
        indexes = torch.zeros(3, prefix_len, dtype=torch.long, device="cuda")
        indexes[0] = torch.arange(prefix_len, device="cuda")
        past = tiny_model(
            inputs_embeds=torch.randn(1, prefix_len, 64, device="cuda", dtype=torch.bfloat16),
            indexes=indexes,
            use_cache=True,
            compute_logits=False,
        ).past_key_values
    prefix = [(layer.keys.clone(), layer.values.clone()) for layer in past.layers] if past is not None else []

    adapter = SenseNovaU1DenoisingAdapter(tiny_model)
    mask = None
    if masked:
        mask = torch.zeros(1, 1, 4, prefix_len + 4, device="cuda", dtype=torch.bfloat16)
        mask[..., 0, -1] = -torch.inf
    for offset in (0, 2):
        kwargs = dict(
            inputs_embeds=torch.randn(1, 4, 64, device="cuda", dtype=torch.bfloat16),
            indexes=torch.tensor([[3, 3, 3, 3], [0, 0, 1, 1], [0, 1, 0, 1]], device="cuda") + offset,
            image_gen_indicators=torch.ones(1, 4, dtype=torch.bool, device="cuda"),
            attention_mask={"full_attention": mask},
            past_key_values=past,
            update_cache=False,
            use_cache=past is not None,
            compute_logits=False,
        )
        expected = tiny_model(**kwargs)
        context = extract_sensenova_u1_context(adapter, **kwargs)
        actual = context.postprocess(*context.run_transformer_blocks())
        assert torch.isfinite(actual.hidden_states).all()
        torch.testing.assert_close(actual.hidden_states, expected.hidden_states, rtol=0, atol=0)
        assert actual.past_key_values is past
        for layer, (keys, values) in zip(past.layers if past is not None else [], prefix):
            torch.testing.assert_close(layer.keys, keys, rtol=0, atol=0)
            torch.testing.assert_close(layer.values, values, rtol=0, atol=0)
