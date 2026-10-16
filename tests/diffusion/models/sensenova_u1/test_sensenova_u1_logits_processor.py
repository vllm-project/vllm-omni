# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test: SenseNovaU1ForCausalLM instantiation under the diffusion
config shim must not crash on missing ``head_dtype`` and the resulting
LogitsProcessor must have ``head_dtype is None`` (= use model dtype)."""

import os

import pytest
import torch
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
)

from vllm_omni.diffusion.models.sensenova_u1.sensenova_u1_transformer import (
    SenseNovaU1ForCausalLM,
)
from vllm_omni.diffusion.vllm_config import _DiffusionVllmModelConfig
from vllm_omni.transformers_utils.configs.sensenova_u1 import (
    SenseNovaU1Config,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _build_tiny_sensenova_u1(quant_config=None, prefix="model"):
    """Build a minimal SenseNovaU1ForCausalLM for unit tests."""
    config = SenseNovaU1Config(
        llm_config={
            "hidden_size": 64,
            "num_attention_heads": 2,
            "num_hidden_layers": 1,
            "intermediate_size": 128,
            "vocab_size": 32,
            "max_position_embeddings": 128,
            "max_position_embeddings_hw": 128,
        },
    )
    return SenseNovaU1ForCausalLM(config.llm_config, quant_config=quant_config, prefix=prefix)


def test_logits_processor_head_dtype_under_diffusion_shim():
    # Initialize the distributed environment
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29543")
    init_distributed_environment(world_size=1, rank=0, local_rank=0, distributed_init_method="env://")
    initialize_model_parallel()

    # Create a vLLM config with a wrapped diffusion config
    fake_diff_config = _DiffusionVllmModelConfig(
        model="sensenova-u1-test",
        dtype=torch.bfloat16,
        max_model_len=8192,
        original_max_model_len=8192,
    )
    vllm_config = VllmConfig(
        device_config=DeviceConfig(device="cpu"),
    )
    vllm_config.modelconfig = fake_diff_config  # type: ignore[assignment]

    # Build a tiny version of the Causal LM component
    with set_current_vllm_config(vllm_config):
        model = _build_tiny_sensenova_u1()

    # Ensure that we have a set head_dtype attribute. Currently, the head_dtype
    # is set to None
    head_dtype = model.logits_processor.head_dtype
    assert head_dtype is None
    cleanup_dist_env_and_memory()
