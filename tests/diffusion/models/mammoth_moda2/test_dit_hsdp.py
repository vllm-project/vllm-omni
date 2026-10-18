# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""HSDP boundaries and real FSDP2 wrapping for the Mammoth DiT."""

import pytest
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor
from vllm.utils.network_utils import get_file_store_init_method

from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.distributed.hsdp import shard_model
from vllm_omni.diffusion.models.mammoth_moda2.mammothmoda2_dit_model import Transformer2DModel

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture
def tiny_dit() -> Transformer2DModel:
    config = OmniDiffusionConfig(diffusion_attention_config={"default": {"backend": "TORCH_SDPA"}})
    with set_current_diffusion_config(config):
        return Transformer2DModel(
            hidden_size=48,
            num_layers=2,
            num_refiner_layers=1,
            num_attention_heads=6,
            num_kv_heads=2,
            multiple_of=8,
            axes_dim_rope=(4, 2, 2),
            axes_lens=(8, 8, 8),
            text_feat_dim=16,
        ).eval()


def test_hsdp_selects_main_and_refiner_blocks(tiny_dit: Transformer2DModel):
    # Check the actual module tree, including refiners and nested attention/MLP modules.
    selected = [
        name
        for name, module in tiny_dit.named_modules()
        if any(condition(name, module) for condition in tiny_dit._hsdp_shard_conditions)
    ]
    assert selected == [
        "noise_refiner.0",
        "ref_image_refiner.0",
        "context_refiner.0",
        "layers.0",
        "layers.1",
    ]


def test_hsdp_wraps_blocks_and_remaining_root_parameters(tiny_dit: Transformer2DModel):
    # World size one exercises real FSDP2 ownership without requiring CUDA.
    # It does not establish multi-GPU forward correctness or memory savings.
    if dist.is_initialized():
        pytest.skip("This test owns its singleton process group")
    dist.init_process_group("gloo", rank=0, world_size=1, init_method=get_file_store_init_method())
    try:
        mesh = init_device_mesh("cpu", (1, 1), mesh_dim_names=("replicate", "shard"))
        shard_model(tiny_dit, mesh=mesh, hsdp_shard_conditions=tiny_dit._hsdp_shard_conditions)
        assert isinstance(tiny_dit, FSDPModule)
        for blocks in (tiny_dit.layers, tiny_dit.noise_refiner, tiny_dit.ref_image_refiner, tiny_dit.context_refiner):
            assert all(isinstance(block, FSDPModule) for block in blocks)
        assert all(isinstance(parameter, DTensor) for parameter in tiny_dit.parameters())
    finally:
        dist.destroy_process_group()
