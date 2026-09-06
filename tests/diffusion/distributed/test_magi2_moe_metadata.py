# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.models.magi2.mh_moe import (
    Magi2MultiHeadMoE,
    Magi2MultiHeadMoEConfig,
)
from vllm_omni.diffusion.models.magi2.parallel import Magi2ParallelGroup
from vllm_omni.platforms import current_omni_platform

_WORLD_SIZE = 4
pytestmark = [
    pytest.mark.core_model,
    pytest.mark.diffusion,
    pytest.mark.parallel,
    pytest.mark.sp,
]


def _moe_config() -> Magi2MultiHeadMoEConfig:
    return Magi2MultiHeadMoEConfig(
        hidden_size=32,
        num_heads=4,
        num_experts=2,
        top_k=2,
        expert_intermediate_size=8,
        params_dtype=torch.float32,
    )


def _initialize_moe(model: Magi2MultiHeadMoE, seed: int) -> None:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for parameter in model.parameters():
            values = torch.randn(
                parameter.shape,
                dtype=parameter.dtype,
                generator=generator,
            )
            parameter.copy_(values.to(parameter.device) * 0.025)


def _copy_ep_shard(source: Magi2MultiHeadMoE, target: Magi2MultiHeadMoE) -> None:
    source_parameters = dict(source.named_parameters())
    with torch.no_grad():
        for name, parameter in target.named_parameters():
            parameter.copy_(target.ep_slice(source_parameters[name]))


def _cuda_worker(rank: int, rendezvous: str) -> None:
    device = torch.device(f"{current_omni_platform.device_type}:{rank}")
    current_omni_platform.set_device(device)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=_WORLD_SIZE,
    )
    try:
        singleton = Magi2ParallelGroup(None, world_size=1, rank=0)
        sp_group = Magi2ParallelGroup(
            dist.group.WORLD,
            world_size=_WORLD_SIZE,
            rank=rank,
        )
        oracle = Magi2MultiHeadMoE(_moe_config(), ep_group=singleton).to(device)
        _initialize_moe(oracle, seed=19)
        distributed = Magi2MultiHeadMoE(_moe_config(), ep_group=sp_group).to(device)
        _copy_ep_shard(oracle, distributed)

        split_sizes = [rank_size + 2 for rank_size in range(_WORLD_SIZE)]
        generator = torch.Generator(device="cpu").manual_seed(101 + rank)
        local_input = torch.randn(
            split_sizes[rank],
            32,
            dtype=torch.float32,
            generator=generator,
        ).to(device)
        with torch.no_grad():
            expected = oracle(local_input)

        scalar_all_gathers = 0
        original_all_gather = dist.all_gather

        def counted_all_gather(output_tensors, input_tensor, *args, **kwargs):
            nonlocal scalar_all_gathers
            if input_tensor.numel() == 1 and all(tensor.numel() == 1 for tensor in output_tensors):
                scalar_all_gathers += 1
            return original_all_gather(output_tensors, input_tensor, *args, **kwargs)

        dist.all_gather = counted_all_gather
        try:
            with torch.no_grad():
                actual = distributed(
                    local_input,
                    sequence_split_sizes=split_sizes,
                )
        finally:
            dist.all_gather = original_all_gather

        max_error = (actual - expected).abs().max()
        dist.all_reduce(max_error, op=dist.ReduceOp.MAX)
        scalar_call_count = torch.tensor(scalar_all_gathers, device=device)
        dist.all_reduce(scalar_call_count, op=dist.ReduceOp.MAX)
        if scalar_call_count.item() != 0:
            raise AssertionError("request-scoped split metadata triggered a redundant all_gather")
        if max_error.item() > 1e-5:
            raise AssertionError(f"SP4 MoE output differs from the local oracle: max error={max_error.item():.8g}")
    finally:
        dist.destroy_process_group()


@hardware_test(res={"cuda": "L4"}, num_cards=4)
def test_magi2_sp4_reuses_sequence_metadata_with_real_collectives() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        rendezvous = f"file://{os.path.join(temp_dir, 'nccl-rendezvous')}"
        mp.spawn(
            _cuda_worker,
            args=(rendezvous,),
            nprocs=_WORLD_SIZE,
            join=True,
        )
