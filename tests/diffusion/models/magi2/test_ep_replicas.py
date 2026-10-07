# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Head-EP exchange over the whole SP group.

When the SP ranks split into consecutive head-EP groups, ranks at the same
position of every group own the same MoE head shard.  Exchanging over the SP
group sends each owner part of every rank's tokens instead of all tokens of
its own group.  Each owner keeps its token count, so only which rows share
its MoE batch, and in which order, changes; the MoE output must not change by
a single bit.
"""

from __future__ import annotations

import os
import tempfile
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import vllm_omni.diffusion.distributed.parallel_state as parallel_state
import vllm_omni.diffusion.models.magi2.mh_moe as mh_moe
import vllm_omni.diffusion.models.magi2.modeling_magi2 as modeling
from tests.diffusion.models.magi2.test_bf16_moe_wiring import _gpu_device
from tests.diffusion.models.magi2.test_native_distributed_parity import _tiny_config
from vllm_omni.diffusion.distributed.head_parallel import replica_token_counts
from vllm_omni.diffusion.models.magi2.parallel import (
    Magi2ParallelGroup,
    balanced_split_sizes,
    get_magi2_ep_replicas,
)

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model]

_EP4_RUNS = [[0, 1, 2, 3], [4, 5, 6, 7]]


def _patch_ranks(monkeypatch, ep_group_ranks):
    # Groups in these tests are rank tuples; the global EP layout is a list of runs.
    monkeypatch.setattr(dist, "get_process_group_ranks", list)
    monkeypatch.setattr(parallel_state, "get_expert_parallel_group_ranks", lambda: ep_group_ranks)


@pytest.mark.cpu
@pytest.mark.parametrize("rank", range(8))
def test_ep4_groups_tile_the_sp8_group(monkeypatch, rank):
    _patch_ranks(monkeypatch, _EP4_RUNS)
    sp = Magi2ParallelGroup(group=tuple(range(8)), world_size=8, rank=rank)
    ep = Magi2ParallelGroup(group=tuple(_EP4_RUNS[rank // 4]), world_size=4, rank=rank % 4)
    assert get_magi2_ep_replicas(ep, sp) == 2


@pytest.mark.cpu
@pytest.mark.parametrize(
    "ep_ranks,ep_rank,sp_ranks,sp_rank,ep_group_ranks",
    [
        # EP spans the SP group (SP4xCFG2): nothing to replicate.
        ((0, 1, 2, 3), 1, (0, 1, 2, 3), 1, [[0, 1, 2, 3], [4, 5, 6, 7]]),
        # Strided EP groups do not tile the SP group in order.
        ((0, 2), 1, (0, 1, 2, 3), 2, [[0, 2], [1, 3]]),
        # Runs missing from the head-EP layout do not own replicated shards.
        ((4, 5), 0, (4, 5, 6, 7), 0, [[0, 1], [2, 3]]),
        # A single-rank EP group never exchanges.
        ((3,), 0, (0, 1, 2, 3), 3, [[0], [1], [2], [3]]),
    ],
)
def test_ep_replicas_fall_back_to_one(monkeypatch, ep_ranks, ep_rank, sp_ranks, sp_rank, ep_group_ranks):
    _patch_ranks(monkeypatch, ep_group_ranks)
    ep = Magi2ParallelGroup(group=ep_ranks, world_size=len(ep_ranks), rank=ep_rank)
    sp = Magi2ParallelGroup(group=sp_ranks, world_size=len(sp_ranks), rank=sp_rank)
    assert get_magi2_ep_replicas(ep, sp) == 1


@pytest.mark.cpu
def test_ep_replicas_need_process_groups_and_a_head_ep_layout(monkeypatch):
    sp = Magi2ParallelGroup(group=tuple(range(8)), world_size=8, rank=0)
    ep = Magi2ParallelGroup(group=(0, 1, 2, 3), world_size=4, rank=0)
    tp = Magi2ParallelGroup(group=(0, 1, 2, 3), world_size=4, rank=0, replicated_sequence=True)
    _patch_ranks(monkeypatch, _EP4_RUNS)
    assert get_magi2_ep_replicas(tp, sp) == 1
    assert get_magi2_ep_replicas(Magi2ParallelGroup(None, 4, 0), sp) == 1

    def uninitialized():
        raise AssertionError("expert parallel group ranks are not initialized")

    monkeypatch.setattr(parallel_state, "get_expert_parallel_group_ranks", uninitialized)
    assert get_magi2_ep_replicas(ep, sp) == 1


@pytest.mark.cpu
def test_ep_replicas_reject_an_ep_group_outside_its_run(monkeypatch):
    _patch_ranks(monkeypatch, _EP4_RUNS)
    sp = Magi2ParallelGroup(group=tuple(range(8)), world_size=8, rank=5)
    ep = Magi2ParallelGroup(group=(0, 1, 2, 3), world_size=4, rank=1)
    with pytest.raises(ValueError, match="run of SP ranks"):
        get_magi2_ep_replicas(ep, sp)


def _moe(ep_group: Magi2ParallelGroup) -> mh_moe.Magi2MultiHeadMoE:
    moe_config = _tiny_config().moe
    config = mh_moe.Magi2MultiHeadMoEConfig(
        hidden_size=32,
        num_heads=moe_config.num_heads,
        num_experts=moe_config.num_experts,
        top_k=moe_config.top_k,
        expert_intermediate_size=moe_config.expert_intermediate_size,
        params_dtype=torch.float32,
    )
    return mh_moe.Magi2MultiHeadMoE(config, ep_group=ep_group)


@pytest.mark.cpu
def test_dispatch_group_defaults_to_the_ep_group_and_must_align():
    ep = Magi2ParallelGroup(group=(4, 5), world_size=2, rank=1)
    moe = _moe(ep)
    assert moe.dispatch_group is ep and moe.dispatch_replicas == 1
    for misaligned in (
        Magi2ParallelGroup(group=(4, 5, 6), world_size=3, rank=1),
        Magi2ParallelGroup(group=(4, 5, 6, 7), world_size=4, rank=2),
    ):
        with pytest.raises(ValueError, match="aligned"):
            moe.set_dispatch_group(misaligned)
    sp = Magi2ParallelGroup(group=(4, 5, 6, 7), world_size=4, rank=3)
    moe.set_dispatch_group(sp)
    assert moe.dispatch_group is sp and moe.dispatch_replicas == 2
    # The weights still follow the EP rank.
    assert moe.local_head_start == 2


@pytest.mark.cpu
@pytest.mark.parametrize("is_musa", [True, False])
def test_layer_exchanges_over_the_sp_group_on_musa(monkeypatch, is_musa):
    _patch_ranks(monkeypatch, _EP4_RUNS)
    sp = Magi2ParallelGroup(group=tuple(range(8)), world_size=8, rank=5)
    ep = Magi2ParallelGroup(group=(4, 5, 6, 7), world_size=4, rank=1)
    monkeypatch.setattr(modeling, "get_magi2_expert_parallel_config", lambda: SimpleNamespace())
    monkeypatch.setattr(modeling, "get_magi2_ulysses_group", lambda: sp)
    monkeypatch.setattr(mh_moe, "get_magi2_ep_group", lambda: ep)
    monkeypatch.setattr(modeling.current_omni_platform, "is_musa", lambda: is_musa)

    layer = modeling.Magi2MultiHeadMoELayer(_tiny_config())
    cp_split_sizes = [11, 11, 11, 11, 10, 10, 10, 10]
    assert layer.moe_mlp.ep_group is ep
    if is_musa:
        assert layer.moe_mlp.dispatch_group is sp and layer.moe_mlp.dispatch_replicas == 2
        assert layer.ep_sequence_split_sizes(cp_split_sizes) == cp_split_sizes
    else:
        assert layer.moe_mlp.dispatch_group is ep and layer.moe_mlp.dispatch_replicas == 1
        assert layer.ep_sequence_split_sizes(cp_split_sizes) == [10, 10, 10, 10]


def _shard_transform(moe: mh_moe.Magi2MultiHeadMoE, seen: list[int]):
    # A per-token function of the head shard, identical on both owners of a shard.
    def local_forward(x_heads: torch.Tensor) -> torch.Tensor:
        seen.append(x_heads.shape[0])
        return x_heads * (moe.local_head_start + 2) - x_heads.roll(1, dims=-1)

    return local_forward


def _plumbing_worker(rank: int, rendezvous: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4, timeout=timedelta(seconds=60))
    try:
        runs = ((0, 1), (2, 3))
        run_groups = [dist.new_group(ranks=list(run), backend="gloo") for run in runs]
        ep = Magi2ParallelGroup(run_groups[rank // 2], world_size=2, rank=rank % 2)
        sp = Magi2ParallelGroup(dist.group.WORLD, world_size=4, rank=rank)
        moe = _moe(ep)
        seen: list[int] = []
        moe._local_forward = _shard_transform(moe, seen)
        exchanges: list[int] = []
        dispatch = mh_moe.ep_dispatch

        def recording_dispatch(x_heads, group, sizes, replicas=1):
            exchanges.append(replicas)
            return dispatch(x_heads, group, sizes, replicas)

        mh_moe.ep_dispatch = recording_dispatch
        for counts in ((6, 6, 6, 6), (7, 6, 5, 6), (7, 6, 6, 5), (3, 0, 1, 6)):
            generator = torch.Generator().manual_seed(100 * rank + sum(counts))
            x = torch.randn(counts[rank], 32, generator=generator)
            moe.dispatch_group, moe.dispatch_replicas = ep, 1
            expected = moe(x, list(counts[(rank // 2) * 2 : (rank // 2) * 2 + 2]))
            moe.set_dispatch_group(sp)
            seen.clear()
            exchanges.clear()
            actual = moe(x, list(counts))
            assert torch.equal(actual, expected)
            # Runs with different token totals keep the exchange inside each run.
            assert exchanges == [2 if sum(counts[:2]) == sum(counts[2:]) else 1]
            # Every owner receives as many tokens as the exchange inside its run gives it.
            assert seen == [sum(counts[(rank // 2) * 2 : (rank // 2) * 2 + 2])]
            parts = replica_token_counts(counts, 2)
            assert sum(rank_parts[rank // 2] for rank_parts in parts) == seen[0]
            # Without host counts the MoE gathers them over the SP group.
            assert torch.equal(moe(x), expected)
        dist.barrier()
        for group in run_groups:
            dist.destroy_process_group(group)
    finally:
        dist.destroy_process_group()


@pytest.mark.cpu
@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="requires Gloo")
def test_moe_forward_over_the_sp_group_matches_the_ep_group(tmp_path):
    mp.spawn(_plumbing_worker, args=(f"file://{tmp_path / 'gloo-init'}",), nprocs=4, join=True)


def _production_moe(
    device: torch.device,
    num_heads: int,
    *,
    seed: int,
    ep_group: Magi2ParallelGroup | None = None,
) -> mh_moe.Magi2MultiHeadMoE:
    """MAGI-2 Preview MoE heads (256-wide, 256 experts, top-6, 1280) with random weights.

    The weights depend only on the seed and the head shard, so ranks that own
    the same shard hold identical weights.
    """
    config = mh_moe.Magi2MultiHeadMoEConfig(
        hidden_size=num_heads * 256,
        num_heads=num_heads,
        num_experts=256,
        top_k=6,
        expert_intermediate_size=1280,
        params_dtype=torch.bfloat16,
        route_scale=4.9,
    )
    moe = mh_moe.Magi2MultiHeadMoE(config, ep_group=ep_group or Magi2ParallelGroup(None, 1, 0)).to(device)
    generator = torch.Generator(device=device).manual_seed(seed + moe.local_head_start)
    with torch.no_grad():
        for parameter, scale in (
            (moe.gate, 256**-0.5),
            (moe.W_gate, 256**-0.5),
            (moe.W_up, 256**-0.5),
            (moe.W_down, 1280**-0.5),
            (moe.router.expert_bias_ema, 0.1),
        ):
            parameter.copy_(
                torch.randn(parameter.shape, generator=generator, device=device, dtype=torch.float32) * scale
            )
        moe.router.expert_bias.zero_()
    moe.prepare_bf16_weights()
    return moe


def _owner_rows(counts: tuple[int, ...], replicas: int) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Global token rows each head owner processes, for both exchanges."""
    shards = len(counts) // replicas
    starts = [sum(counts[:rank]) for rank in range(len(counts))]
    run_rows = [
        torch.arange(starts[replica * shards], sum(counts[: (replica + 1) * shards])) for replica in range(replicas)
    ]
    parts = replica_token_counts(counts, replicas)
    replica_rows = []
    for replica in range(replicas):
        rows = []
        for start, rank_parts in zip(starts, parts):
            begin = start + sum(rank_parts[:replica])
            rows.append(torch.arange(begin, begin + rank_parts[replica]))
        replica_rows.append(torch.cat(rows))
    return run_rows, replica_rows


@pytest.mark.musa
@pytest.mark.parametrize(
    "counts",
    [
        # EP4 = SP8xCFG1: 29616 tokens, 3702 per SP rank, 14808 per owner.
        (3702,) * 8,
        tuple(balanced_split_sizes(29613, 8)),
    ],
)
def test_owner_moe_rows_do_not_depend_on_their_batch(monkeypatch, counts):
    """One head shard of the production MoE: owners of the SP-group exchange
    see other rows, in another order, but return the same bits per row."""
    device = _gpu_device("musa")
    monkeypatch.delenv("MAGI2_DETERMINISTIC", raising=False)
    monkeypatch.delenv("MAGI2_ROUTER_BIAS_SOURCE", raising=False)
    moe = _production_moe(device, 3, seed=7)
    generator = torch.Generator(device=device).manual_seed(11)
    x = torch.randn((sum(counts), 3, 256), generator=generator, device=device).to(torch.bfloat16)

    run_rows, replica_rows = _owner_rows(counts, 2)
    outputs = []
    for owners in (run_rows, replica_rows):
        output = torch.empty_like(x)
        for rows in owners:
            rows = rows.to(device)
            output[rows] = moe._local_forward(x[rows].contiguous())
        outputs.append(output)
    assert [rows.numel() for rows in run_rows] == [rows.numel() for rows in replica_rows]
    assert torch.equal(outputs[1], outputs[0])


def _device_worker(rank: int, rendezvous: str, world_size: int) -> None:
    torch.accelerator.set_device_index(rank)
    device = torch.device("musa", rank)
    dist.init_process_group(
        "mccl", init_method=rendezvous, rank=rank, world_size=world_size, timeout=timedelta(seconds=300)
    )
    try:
        shards = world_size // 2
        run_groups = [dist.new_group(ranks=list(range(start, start + shards))) for start in (0, shards)]
        run = rank // shards
        ep = Magi2ParallelGroup(run_groups[run], world_size=shards, rank=rank % shards)
        sp = Magi2ParallelGroup(dist.group.WORLD, world_size=world_size, rank=rank)
        moe = _production_moe(device, 12, seed=0, ep_group=ep)
        counts = (3702,) * world_size
        generator = torch.Generator(device=device).manual_seed(rank)
        x = torch.randn((counts[rank], 12 * 256), generator=generator, device=device).to(torch.bfloat16)
        expected = moe(x, list(counts[run * shards : (run + 1) * shards]))
        moe.set_dispatch_group(sp)
        actual = moe(x, list(counts))
        assert torch.equal(actual, expected)
        dist.barrier()
        for group in run_groups:
            dist.destroy_process_group(group)
    finally:
        dist.destroy_process_group()


@pytest.mark.musa
def test_moe_forward_over_the_sp_group_is_bit_exact_on_musa():
    if getattr(torch.version, "musa", None) is None or not torch.musa.is_available():
        pytest.skip("MUSA runtime and devices required")
    world_size = 8 if torch.musa.device_count() >= 8 else 4
    if torch.musa.device_count() < world_size:
        pytest.skip("requires four MUSA devices")
    with tempfile.TemporaryDirectory() as directory:
        rendezvous = f"file://{os.path.join(directory, 'mccl-init')}"
        mp.spawn(_device_worker, args=(rendezvous, world_size), nprocs=world_size, join=True)
