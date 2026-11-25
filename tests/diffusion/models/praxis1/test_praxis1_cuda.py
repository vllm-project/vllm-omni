# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Weight-free GPU checks of the Praxis-1 transformer: block-sparse and dense attention agree."""

import os

import pytest
import torch

from tests.helpers.mark import hardware_marks
from vllm_omni.platforms import current_omni_platform

pytestmark = [
    pytest.mark.core_model,
    *hardware_marks(res={"cuda": "L4"}, num_cards=1),
    pytest.mark.diffusion,
    pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="Praxis-1 attention kernels run on CUDA"),
]

TEXT, VIDEO = "TextEncodingTensor", "VideoTensor"
TINY_HW = (2, 4)


def tiny_config() -> dict:
    """A Praxis-1 transformer config with small dimensions; MoE sizes are multiples of 8."""
    backbone = {
        "dim": 64,
        "timestep_dim": 16,
        "num_registers": 4,
        "dense_ffn_hidden": 96,
        "attention": {"num_heads": 4, "num_kv_heads": 2, "head_dim": 16},
        "rope": {"effective_dim": 8, "base": 26_000_000, "efficient_cache_max": 10**6},
        "moe": {"num_routed_experts": 4, "top_k_routed": 2, "expert_hidden": 32, "routed_scaling_factor": 2.5},
        "encoder": {
            "depth": 3,
            "dense_blocks": [0],
            "tap_blocks": [0, 2],
            "input_projections": {TEXT: {"c_in": 32}, VIDEO: {"c_in": 192}},
        },
        "decoder": {
            "depth": 2,
            "dense_blocks": [0],
            "modulation_dim": 64,
            "unpatch_factor": [1, 2, 2],
            "input_projections": {VIDEO: {"c_in": 48}},
        },
    }
    expert = {
        "dim": 64,
        "num_layers": 4,
        "num_heads": 1,
        "head_dim": 64,
        "trunk_dim": 64,
        "action_patch_dim": 40,
        "proprio_patch_dim": 10,
        "num_future_per_chunk": 2,
        "max_chunks": 16,
        "tap_norm": True,
        "clean_only_conditioning": False,
        "rope": {"effective_dim": 64, "base": 26_000_000, "table_size": 10**6},
    }
    return {
        "backbone": backbone,
        "action_expert": expert,
        "chunking": {"n_chunks": 4, "latents_per_chunk": 6},
    }


@pytest.fixture(scope="module")
def vllm_runtime():
    """World-size-1 vLLM state for parallel linears, fused MoE and its workspace."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.v1.worker.workspace import init_workspace_manager

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29537")
    config = VllmConfig()
    with set_current_vllm_config(config):
        init_distributed_environment(world_size=1, rank=0, local_rank=0, distributed_init_method="env://")
        initialize_model_parallel()
    init_workspace_manager(torch.device("cuda"))
    yield config
    cleanup_dist_env_and_memory()


def build_transformer(attention_impl: str, dtype: torch.dtype, seed: int = 0):
    """Build a randomly initialized tiny model; each model owns the vLLM config its MoE layers register in."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.utils.torch_utils import set_default_torch_dtype

    from vllm_omni.diffusion.models.praxis1.praxis1_transformer import Praxis1Transformer

    device = torch.device("cuda")
    vllm_config = VllmConfig()
    with set_current_vllm_config(vllm_config), torch.device(device), set_default_torch_dtype(dtype):
        model = Praxis1Transformer.from_config(tiny_config(), attention_impl).eval().requires_grad_(False)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.05)
        for name, buffer in model.named_buffers():
            if name.endswith("expert_bias"):
                buffer.copy_(torch.randn(buffer.shape, generator=generator) * 0.01)
            elif name.endswith(("freqs", "phases")):
                buffer.copy_(torch.rand(buffer.shape, generator=generator) * 2 - 1)
    with set_current_vllm_config(vllm_config), torch.device(device):
        for module in model.modules():
            quant_method = getattr(module, "quant_method", None)
            if quant_method is not None and hasattr(quant_method, "process_weights_after_loading"):
                quant_method.process_weights_after_loading(module)
    return model, vllm_config


def sample(model, vllm_config, depth: int, settings):
    from vllm.config import set_current_vllm_config
    from vllm.forward_context import set_forward_context

    device, dtype = torch.device("cuda"), model.dtype
    h, w = TINY_HW
    generator = torch.Generator(device="cpu").manual_seed(1)

    def randn(*shape):
        return torch.randn(shape, generator=generator).to(device, dtype)

    text, uncond, anchor = randn(1, 5, 32), randn(1, 3, 32), randn(1, 192, 1, h, w)
    cleans = [randn(1, 192, 6, h, w) for _ in range(depth)]
    proprio = torch.randn(1, depth + 1, 10, generator=generator).to(device)
    with set_current_vllm_config(vllm_config), set_forward_context(None, vllm_config):
        return model.sample_window_joint(text, uncond, anchor, cleans, proprio, seed=7, settings=settings)


@pytest.fixture
def float32_moe_tiles():
    """Fused-MoE tiles that fit float32 activations in an L4's 99 KiB of shared memory.

    vLLM sizes its default tiles for 16-bit activations; above 512 routed tokens
    they need 128 KiB in float32. The pipeline itself only runs bfloat16.
    """
    from vllm.model_executor.layers.fused_moe import override_config

    with override_config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8, "SPLIT_K": 1}):
        yield


@pytest.mark.parametrize("depth", [0, 2])
def test_block_sparse_and_dense_attention_agree(vllm_runtime, float32_moe_tiles, depth):
    from vllm_omni.model_extras.praxis1 import SamplerSettings

    settings = SamplerSettings(num_steps=2, t_shift=12.5, cfg_scale=1.5, cfg_video_scale=4.0)
    flex, flex_config = build_transformer("flex", torch.float32)
    video_flex, actions_flex = sample(flex, flex_config, depth, settings)
    assert video_flex.shape == (1, 192, 6, *TINY_HW) and actions_flex.shape == (1, 6, 40)
    assert torch.isfinite(video_flex).all() and torch.isfinite(actions_flex).all()
    video_again, actions_again = sample(flex, flex_config, depth, settings)
    torch.testing.assert_close(video_again, video_flex, rtol=0, atol=0)
    torch.testing.assert_close(actions_again, actions_flex, rtol=0, atol=0)
    del flex
    dense, dense_config = build_transformer("backend", torch.float32)
    video_dense, actions_dense = sample(dense, dense_config, depth, settings)
    torch.testing.assert_close(video_dense, video_flex, rtol=1e-3, atol=2e-3)
    torch.testing.assert_close(actions_dense, actions_flex, rtol=1e-3, atol=2e-3)


def test_bf16_sampler_runs_every_context_depth(vllm_runtime):
    from vllm_omni.model_extras.praxis1 import SamplerSettings

    model, config = build_transformer("flex", torch.bfloat16)
    for depth in range(4):
        video, actions = sample(model, config, depth, SamplerSettings(num_steps=1, cfg_video_scale=1.0))
        assert video.dtype == actions.dtype == torch.float32
        assert torch.isfinite(video).all() and torch.isfinite(actions).all()
    with pytest.raises(ValueError, match="depth"):
        sample(model, config, 4, SamplerSettings(num_steps=1, cfg_video_scale=1.0))
