# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real collectives through a small complete DiT, including its refiners."""

from pathlib import Path

import pytest
import torch
from torch.distributed import broadcast
from vllm.distributed import get_tp_group
from vllm.model_executor.models.utils import AutoWeightsLoader

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.diffusion.models.mammoth_moda2.mammothmoda2_dit_model import Transformer2DModel
from vllm_omni.diffusion.models.mammoth_moda2.rope_real import RotaryPosEmbedReal
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.parallel]


def _worker(rank: int, world: int, init_method: str, result_dir: Path) -> None:
    current_omni_platform.set_device(rank)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    init_distributed_environment(
        world_size=world, rank=rank, local_rank=rank, distributed_init_method=init_method, backend="nccl"
    )
    initialize_model_parallel(tensor_parallel_size=world)
    try:
        group = get_tp_group().device_group
        config = dict(
            hidden_size=126,
            num_layers=2,
            num_refiner_layers=1,
            num_attention_heads=21,
            num_kv_heads=7,
            multiple_of=16,
            axes_dim_rope=(2, 2, 2),
            axes_lens=(32, 32, 32),
            text_feat_dim=12,
            in_channels=4,
        )
        dtype = torch.float32
        backend_config = OmniDiffusionConfig(diffusion_attention_config={"default": {"backend": "TORCH_SDPA"}})
        with set_current_diffusion_config(backend_config):
            local = Transformer2DModel.from_config(config).eval().to(device="cuda", dtype=dtype)
        weights_path = result_dir / "weights.pt"
        if world == 1:
            torch.save({name: value.cpu() for name, value in local.state_dict().items()}, weights_path)
        else:
            weights = torch.load(weights_path, weights_only=True)
            assert AutoWeightsLoader(local).load_weights(weights.items()) == set(local.state_dict())
        assert local.layers[0].attn.heads == 3 * ((7 + world - 1) // world)
        freqs = RotaryPosEmbedReal.get_freqs_real(config["axes_dim_rope"], (32, 32, 32), theta=10000)
        # Input generation must not depend on TP-specific parameter initialization.
        torch.manual_seed(43)
        kwargs = dict(
            timestep=torch.ones(2, device="cuda"),
            text_hidden_states=torch.randn(2, 5, 12, device="cuda", dtype=dtype),
            text_attention_mask=torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], device="cuda", dtype=torch.bool),
            freqs_cis=freqs,
        )
        latents = torch.randn(2, 4, 4, 4, device="cuda", dtype=dtype)
        with torch.inference_mode():
            actual = local(latents, **kwargs)
        if rank == 0:
            torch.save(actual.cpu(), result_dir / f"tp{world}.pt")
        # Output is replicated; every rank must agree even after the two
        # independently sharded branches have been combined at each block.
        rank_zero = actual.clone()
        broadcast(rank_zero, src=0, group=group)
        torch.testing.assert_close(actual, rank_zero, atol=0, rtol=0)
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


@hardware_test(res={"cuda": "L4"}, num_cards=2)
def test_tp2_matches_tp1_fp32(tmp_path: Path) -> None:
    if not torch.cuda.is_available() or torch.accelerator.device_count() < 2:
        pytest.skip("Requires two CUDA devices")
    for world in (1, 2):
        torch.multiprocessing.spawn(
            _worker,
            args=(world, (tmp_path / f"rendezvous{world}").as_uri(), tmp_path),
            nprocs=world,
            join=True,
        )
    expected = torch.load(tmp_path / "tp1.pt", weights_only=True)
    actual = torch.load(tmp_path / "tp2.pt", weights_only=True)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
