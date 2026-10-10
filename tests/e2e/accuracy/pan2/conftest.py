# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch


@pytest.fixture
def pan2_transformer_runtime(tmp_path):
    """Construct native parallel layers with real single-rank GPU groups, inside a diffusion forward context."""
    from vllm.config import VllmConfig, set_current_vllm_config

    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.distributed.parallel_state import (
        destroy_distributed_env,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm_omni.diffusion.forward_context import set_forward_context

    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method=f"file://{tmp_path / 'pan2-distributed'}",
    )
    try:
        initialize_model_parallel()
        od_config = OmniDiffusionConfig(model_class_name="PAN2ModularPipeline", dtype=torch.bfloat16)
        with (
            set_current_vllm_config(VllmConfig()),
            set_forward_context(omni_diffusion_config=od_config),
            set_current_diffusion_config(od_config),
        ):
            yield od_config
    finally:
        destroy_distributed_env()
