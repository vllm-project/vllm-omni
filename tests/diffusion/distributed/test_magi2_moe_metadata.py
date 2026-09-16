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
from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.models.magi2.attention import VarlenHandler
from vllm_omni.diffusion.models.magi2.configuration_magi2 import (
    Magi2MHCConfig,
    Magi2MoEConfig,
    Magi2PreviewConfig,
)
from vllm_omni.diffusion.models.magi2.mh_moe import (
    Magi2MultiHeadMoE,
    Magi2MultiHeadMoEConfig,
)
from vllm_omni.diffusion.models.magi2.modeling_magi2 import (
    Magi2PreviewTransformer,
    Modality,
)
from vllm_omni.diffusion.models.magi2.parallel import Magi2ParallelGroup
from vllm_omni.platforms import current_omni_platform

_WORLD_SIZE = 4
_SP2_WORLD_SIZE = 2
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


def _transformer_config() -> Magi2PreviewConfig:
    # BF16 because the CUDA attention path selects FlashAttention, which rejects
    # fp32. Two MoE layers keep the collective-count assertion non-vacuous.
    return Magi2PreviewConfig(
        num_layers=2,
        hidden_size=32,
        head_dim=8,
        num_query_groups=4,
        video_in_channels=4,
        audio_in_channels=4,
        text_in_channels=4,
        intermediate_factor=1.5,
        multimodal_layers=(0, 1),
        params_dtype=torch.bfloat16,
        mhc=Magi2MHCConfig(num_streams=2),
        moe=Magi2MoEConfig(
            num_heads=4,
            num_experts=2,
            top_k=2,
            expert_intermediate_size=8,
            shared_expert_intermediate_size=8,
            modality_shared_expert_intermediate_size=8,
            layers=(0, 1),
        ),
    )


def _initialize_transformer(model: Magi2PreviewTransformer) -> None:
    generator = torch.Generator(device="cpu").manual_seed(71)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name == "pre_adapter.rope.bands":
                continue
            values = torch.randn(parameter.shape, dtype=torch.float32, generator=generator) * 0.025
            parameter.copy_(values.to(device=parameter.device, dtype=parameter.dtype))
        for layer in model.block.layers:
            layer.mlp.moe_mlp.router.expert_bias.zero_()
            layer.mlp.moe_mlp.router.expert_bias_ema.zero_()


def _transformer_inputs(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, VarlenHandler]:
    # Nine tokens make the Ulysses split uneven under SP=2 ([5, 4]), which is
    # the layout that must agree between the reused metadata and the tokens
    # each rank actually owns.
    generator = torch.Generator(device="cpu").manual_seed(113)
    packed = (torch.randn(9, 4, generator=generator) * 0.2).to(device=device, dtype=torch.bfloat16)
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
            [1, 0, 0, 1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.float32,
        device=device,
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
            Modality.VIDEO,
        ],
        device=device,
    )
    cumulative = torch.tensor([0, packed.shape[0]], dtype=torch.int32, device=device)
    varlen = VarlenHandler(cumulative, cumulative, packed.shape[0], packed.shape[0])
    return packed, coordinates, modalities, varlen


@contextmanager
def _patched_groups(tp_group: Magi2ParallelGroup, sp_group: Magi2ParallelGroup):
    ep_group = tp_group if tp_group.world_size > 1 else sp_group
    with ExitStack() as stack:
        stack.enter_context(patch.object(layers_module, "get_magi2_tp_group", return_value=tp_group))
        stack.enter_context(patch.object(mh_moe_module, "get_magi2_ep_group", return_value=ep_group))
        stack.enter_context(patch.object(parallel_module, "get_magi2_ulysses_group", return_value=sp_group))
        stack.enter_context(patch.object(attention_module, "get_magi2_ulysses_group", return_value=sp_group))
        yield


@contextmanager
def _count_scalar_all_gathers():
    """Count metadata collectives while still running the real operation."""

    counter = {"calls": 0}
    original_all_gather = dist.all_gather

    def counted_all_gather(output_tensors, input_tensor, *args, **kwargs):
        if input_tensor.numel() == 1 and all(tensor.numel() == 1 for tensor in output_tensors):
            counter["calls"] += 1
        return original_all_gather(output_tensors, input_tensor, *args, **kwargs)

    dist.all_gather = counted_all_gather
    try:
        yield counter
    finally:
        dist.all_gather = original_all_gather


@contextmanager
def _forced_collective_metadata():
    """Reproduce the pre-optimization behavior by discarding reused metadata."""

    original_forward = Magi2MultiHeadMoE.forward

    def collective_forward(self, x, *, sequence_split_sizes=None):
        return original_forward(self, x, sequence_split_sizes=None)

    with patch.object(Magi2MultiHeadMoE, "forward", collective_forward):
        yield


def _sp2_transformer_worker(rank: int, rendezvous: str) -> None:
    device = torch.device(f"{current_omni_platform.device_type}:{rank}")
    current_omni_platform.set_device(device)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=_SP2_WORLD_SIZE,
    )
    try:
        singleton = Magi2ParallelGroup(None, world_size=1, rank=0)
        with _patched_groups(singleton, singleton):
            reference = Magi2PreviewTransformer(_transformer_config()).to(device)
            _initialize_transformer(reference)
            checkpoint = [(name, value.detach().clone()) for name, value in reference.state_dict().items()]
        del reference

        sp_group = Magi2ParallelGroup(dist.group.WORLD, world_size=_SP2_WORLD_SIZE, rank=rank)
        inputs = _transformer_inputs(device)
        with _patched_groups(singleton, sp_group):
            model = Magi2PreviewTransformer(_transformer_config()).to(device)
            if model.load_weights(checkpoint) != set(model.state_dict()):
                raise AssertionError("SP2 transformer did not load every checkpoint weight")

            # Candidate: the production path reusing request-scoped metadata.
            with _count_scalar_all_gathers() as reuse_counter, torch.no_grad():
                candidate = model(*inputs)

            # Baseline: the same ranks and weights forced back onto the
            # per-layer collective, so the comparison isolates this change.
            with _forced_collective_metadata():
                with _count_scalar_all_gathers() as collective_counter, torch.no_grad():
                    baseline = model(*inputs)

        moe_layers = len(_transformer_config().moe.layers)
        reuse_calls = torch.tensor(reuse_counter["calls"], device=device)
        collective_calls = torch.tensor(collective_counter["calls"], device=device)
        mismatch = torch.tensor(int(not torch.equal(candidate, baseline)), device=device)
        dist.all_reduce(reuse_calls, op=dist.ReduceOp.MAX)
        dist.all_reduce(collective_calls, op=dist.ReduceOp.MIN)
        dist.all_reduce(mismatch, op=dist.ReduceOp.MAX)

        if collective_calls.item() != moe_layers:
            raise AssertionError(
                "the forced baseline did not exercise one metadata collective per MoE layer "
                f"(expected {moe_layers}, saw {int(collective_calls.item())})"
            )
        if reuse_calls.item() != 0:
            raise AssertionError(
                "the production SP transformer still performs per-layer sequence-size collectives "
                f"({int(reuse_calls.item())} scalar all_gather calls)"
            )
        if mismatch.item():
            max_error = (candidate.float() - baseline.float()).abs().max()
            raise AssertionError(f"metadata reuse changed SP2 output; max error={max_error.item():.8g}")
    finally:
        dist.destroy_process_group()


@hardware_test(res={"cuda": "L4"}, num_cards=2)
def test_magi2_sp2_transformer_reuses_metadata_with_real_collectives() -> None:
    """Cover the production transformer path, not just a directly called MoE module."""

    with tempfile.TemporaryDirectory() as temp_dir:
        rendezvous = f"file://{os.path.join(temp_dir, 'nccl-sp2-rendezvous')}"
        mp.spawn(
            _sp2_transformer_worker,
            args=(rendezvous,),
            nprocs=_SP2_WORLD_SIZE,
            join=True,
        )
