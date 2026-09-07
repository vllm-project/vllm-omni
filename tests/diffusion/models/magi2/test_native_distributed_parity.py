# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import os
import tempfile
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import vllm_omni.diffusion.models.magi2.attention as attention_module
import vllm_omni.diffusion.models.magi2.layers as layers_module
import vllm_omni.diffusion.models.magi2.mh_moe as mh_moe_module
import vllm_omni.diffusion.models.magi2.parallel as parallel_module
from vllm_omni.diffusion.models.magi2.attention import VarlenHandler
from vllm_omni.diffusion.models.magi2.configuration_magi2 import (
    Magi2MHCConfig,
    Magi2MoEConfig,
    Magi2PreviewConfig,
)
from vllm_omni.diffusion.models.magi2.modeling_magi2 import (
    Magi2PreviewTransformer,
    Modality,
)
from vllm_omni.diffusion.models.magi2.parallel import Magi2ParallelGroup

_WORLD_SIZE = 4
pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


def _tiny_config() -> Magi2PreviewConfig:
    # Four attention, KV, and MoE heads make every head axis divisible by each
    # requested TP/SP layout. The MLP and shared-expert widths are likewise
    # divisible by TP=4, so this exercises all native row/column collectives.
    return Magi2PreviewConfig(
        num_layers=1,
        hidden_size=32,
        head_dim=8,
        num_query_groups=4,
        video_in_channels=4,
        audio_in_channels=4,
        text_in_channels=4,
        intermediate_factor=1.5,
        multimodal_layers=(0,),
        params_dtype=torch.float32,
        mhc=Magi2MHCConfig(num_streams=2),
        moe=Magi2MoEConfig(
            num_heads=4,
            num_experts=2,
            top_k=2,
            expert_intermediate_size=8,
            shared_expert_intermediate_size=8,
            modality_shared_expert_intermediate_size=8,
            layers=(0,),
        ),
    )


def _initialize_model(model: Magi2PreviewTransformer) -> None:
    generator = torch.Generator(device="cpu").manual_seed(71)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name == "pre_adapter.rope.bands":
                continue
            parameter.copy_(
                torch.randn(
                    parameter.shape,
                    dtype=parameter.dtype,
                    generator=generator,
                )
                * 0.025
            )
        moe = model.block.layers[0].mlp.moe_mlp
        moe.router.expert_bias.zero_()
        moe.router.expert_bias_ema.zero_()


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, VarlenHandler]:
    generator = torch.Generator(device="cpu").manual_seed(113)
    packed = torch.randn(8, 4, generator=generator) * 0.2
    coordinates = torch.tensor(
        [
            [0, 0, 0, 2, 2, 2, 2, 2, 2],
            [0, 1, 1, 2, 2, 2, 2, 2, 2],
            [0, 0, 0, 1, 2, 2, 1, 2, 2],
            [0, 1, 1, 1, 2, 2, 1, 2, 2],
            [1, 0, 0, 2, 2, 2, 2, 2, 2],
            [1, 1, 1, 2, 2, 2, 2, 2, 2],
            [0, 0, 0, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.float32,
    )
    modalities = torch.tensor(
        [
            Modality.VIDEO,
            Modality.AUDIO,
            Modality.TEXT,
            Modality.TIME,
            Modality.VIDEO,
            Modality.AUDIO,
            Modality.TEXT,
            Modality.TIME,
        ]
    )
    cumulative = torch.tensor([0, packed.shape[0]], dtype=torch.int32)
    varlen = VarlenHandler(cumulative, cumulative, packed.shape[0], packed.shape[0])
    return packed, coordinates, modalities, varlen


@contextmanager
def _patched_groups(
    tp_group: Magi2ParallelGroup,
    sp_group: Magi2ParallelGroup,
):
    # TP owns native weight and MoE-head shards when enabled. SP owns Ulysses
    # tokens and is also the MoE-head group in the released SP-only layout.
    ep_group = tp_group if tp_group.world_size > 1 else sp_group
    with ExitStack() as stack:
        stack.enter_context(patch.object(layers_module, "get_magi2_tp_group", return_value=tp_group))
        stack.enter_context(patch.object(mh_moe_module, "get_magi2_ep_group", return_value=ep_group))
        stack.enter_context(patch.object(parallel_module, "get_magi2_ulysses_group", return_value=sp_group))
        stack.enter_context(patch.object(attention_module, "get_magi2_ulysses_group", return_value=sp_group))
        yield


def _current_group(
    rank: int,
    rank_groups: tuple[tuple[tuple[int, ...], dist.ProcessGroup], ...],
) -> Magi2ParallelGroup:
    for ranks, process_group in rank_groups:
        if rank in ranks:
            return Magi2ParallelGroup(
                process_group,
                world_size=len(ranks),
                rank=ranks.index(rank),
            )
    return Magi2ParallelGroup(None, world_size=1, rank=0)


def _new_groups(
    rank_sets: tuple[tuple[int, ...], ...],
) -> tuple[tuple[tuple[int, ...], dist.ProcessGroup], ...]:
    groups = []
    for ranks in rank_sets:
        process_group = dist.new_group(ranks=list(ranks), backend="gloo")
        groups.append((ranks, process_group))
    return tuple(groups)


def _distributed_worker(rank: int, rendezvous: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=_WORLD_SIZE,
    )
    try:
        singleton = Magi2ParallelGroup(None, world_size=1, rank=0)
        with _patched_groups(singleton, singleton):
            oracle = Magi2PreviewTransformer(_tiny_config())
            _initialize_model(oracle)
            checkpoint = [(name, value.detach().clone()) for name, value in oracle.state_dict().items()]
            inputs = _inputs()
            with torch.no_grad():
                expected = oracle(*inputs)

        tp2_groups = _new_groups(((0, 1), (2, 3)))
        sp2_groups = _new_groups(((0, 2), (1, 3)))
        world_group = Magi2ParallelGroup(
            dist.group.WORLD,
            world_size=_WORLD_SIZE,
            rank=rank,
        )
        layouts = (
            ("tp4", world_group, singleton),
            ("tp2sp2", _current_group(rank, tp2_groups), _current_group(rank, sp2_groups)),
            ("sp4", singleton, world_group),
        )

        for layout_name, tp_group, sp_group in layouts:
            tp_group = Magi2ParallelGroup(
                tp_group.group,
                tp_group.world_size,
                tp_group.rank,
                replicated_sequence=True,
            )
            with _patched_groups(tp_group, sp_group):
                model = Magi2PreviewTransformer(_tiny_config())
                assert model.load_weights(checkpoint) == set(model.state_dict())
                with torch.no_grad():
                    actual = model(*inputs)

            max_abs_error = (actual - expected).abs().max()
            dist.all_reduce(max_abs_error, op=dist.ReduceOp.MAX)
            close = torch.tensor(
                int(torch.allclose(actual, expected, atol=2e-5, rtol=2e-4)),
                dtype=torch.int32,
            )
            dist.all_reduce(close, op=dist.ReduceOp.MIN)
            if not close.item():
                raise AssertionError(
                    f"{layout_name} differs from the single-rank oracle; "
                    f"global max absolute error={max_abs_error.item():.8g}"
                )
            dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="requires torch.distributed gloo",
)
def test_four_rank_tp_sp_layouts_match_single_rank_oracle() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        rendezvous = f"file://{os.path.join(temp_dir, 'gloo-rendezvous')}"
        mp.spawn(
            _distributed_worker,
            args=(rendezvous,),
            nprocs=_WORLD_SIZE,
            join=True,
        )
