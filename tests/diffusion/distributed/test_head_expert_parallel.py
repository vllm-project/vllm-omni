# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from dataclasses import asdict
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import vllm_omni.diffusion.distributed.parallel_state as state
import vllm_omni.diffusion.model_metadata as metadata
from tests.diffusion.distributed.test_expert_parallel_layout import _FakeGroup
from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.vllm_config import create_diffusion_vllm_config

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model]
_MODEL = "TestHeadEPPipeline"


def _config(ep_size=None, *, tp=1, sp=4, pp=1, cfg=1, dp=1, enabled=True):
    return OmniDiffusionConfig(
        model_class_name=_MODEL,
        dtype=torch.float32,
        parallel_config=DiffusionParallelConfig(
            tensor_parallel_size=tp,
            ulysses_degree=sp,
            pipeline_parallel_size=pp,
            cfg_parallel_size=cfg,
            data_parallel_size=dp,
            enable_expert_parallel=enabled,
            expert_parallel_size=ep_size,
        ),
    )


@pytest.fixture
def head_model(monkeypatch):
    monkeypatch.setitem(
        metadata._DIFFUSION_MODEL_METADATA, _MODEL, metadata.DiffusionModelMetadata(expert_parallel_style="head")
    )


@pytest.mark.cpu
@pytest.mark.parametrize("value", [0, -1, True, "2", 2.5])
def test_invalid_public_ep_size(value):
    with pytest.raises(ValueError):
        DiffusionParallelConfig(enable_expert_parallel=True, expert_parallel_size=value)


@pytest.mark.cpu
def test_ep_size_requires_enable_flag():
    with pytest.raises(ValueError, match="requires enable_expert_parallel"):
        DiffusionParallelConfig(expert_parallel_size=2)


@pytest.mark.cpu
def test_metadata_opt_in_and_config_roundtrip(head_model):
    od = _config(2, cfg=2)
    assert od.parallel_config.world_size == 8
    assert od.use_head_expert_parallel
    assert not od.is_moe  # Model capability does not require HF whole-token expert fields.
    assert DiffusionParallelConfig(**asdict(od.parallel_config)).expert_parallel_size == 2
    assert not _config(enabled=False).use_head_expert_parallel
    od.model_class_name = "Magi2Pipeline"
    assert not od.use_head_expert_parallel  # Model adapter is a separate opt-in.


@pytest.mark.cpu
@pytest.mark.parametrize("head", [False, True])
def test_vllm_view_does_not_fold_native_head_groups(monkeypatch, head):
    monkeypatch.setitem(
        metadata._DIFFUSION_MODEL_METADATA,
        _MODEL,
        metadata.DiffusionModelMetadata(expert_parallel_style="head" if head else "vllm"),
    )
    od = _config(2 if head else None, tp=2, sp=4, cfg=2, dp=2)
    od.tf_model_config = {"num_experts": 8}
    config = create_diffusion_vllm_config(torch.device("cpu"), od)
    assert config.parallel_config.tensor_parallel_size == 2
    assert config.parallel_config.data_parallel_size == (2 if head else 4)
    assert config.parallel_config.prefill_context_parallel_size == (1 if head else 4)
    assert config.parallel_config.enable_expert_parallel is not head


def _fake_state(monkeypatch, od, rank=0, fail_ep=False):
    created = []

    def factory(group_ranks, local_rank, backend, parallel_mode, **kwargs):
        if fail_ep and parallel_mode == "expert":
            raise RuntimeError("EP creation failed")
        group = _FakeGroup(group_ranks, local_rank, parallel_mode, **kwargs)
        group.destroy = Mock()
        created.append(group)
        return group

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: od.parallel_config.world_size)
    monkeypatch.setattr(dist, "new_group", lambda ranks: tuple(ranks))
    monkeypatch.setattr(
        state, "get_world_group", lambda: SimpleNamespace(rank_in_group=rank, local_rank=rank, device_group=object())
    )
    monkeypatch.setattr(state, "get_forward_context", lambda: SimpleNamespace(omni_diffusion_config=od))
    monkeypatch.setattr(state, "init_model_parallel_group", factory)
    monkeypatch.setattr(
        state,
        "init_vllm_model_parallel_group",
        Mock(side_effect=AssertionError("native EP must not create FusedMoE communicators")),
    )
    for name in ("_DP", "_CFG", "_SP", "_PP", "_FS", "_HSDP_REPLICATE", "_EXPERT_PARALLEL_GROUP_RANKS"):
        monkeypatch.setattr(state, name, None)
    for name in ("_TP", "_PCP", "_DP", "_EP", "_PP"):
        monkeypatch.setattr(state.vllm_parallel_state, name, None)
    return created


def _initialize(od, backend="gloo"):
    pc = od.parallel_config
    state.initialize_model_parallel(
        tensor_parallel_size=pc.tensor_parallel_size,
        sequence_parallel_size=pc.sequence_parallel_size,
        ulysses_degree=pc.ulysses_degree,
        pipeline_parallel_size=pc.pipeline_parallel_size,
        cfg_parallel_size=pc.cfg_parallel_size,
        data_parallel_size=pc.data_parallel_size,
        enable_expert_parallel=pc.enable_expert_parallel,
        backend=backend,
    )


@pytest.mark.cpu
@pytest.mark.parametrize(
    "tp,sp,pp,cfg,dp,ep", [(1, 4, 1, 2, 1, None), (1, 8, 1, 1, 1, 4), (2, 4, 2, 2, 2, 2), (1, 4, 1, 2, 2, 1)]
)
def test_head_ep_stays_inside_each_sp_replica(monkeypatch, head_model, tp, sp, pp, cfg, dp, ep):
    od = _config(ep, tp=tp, sp=sp, pp=pp, cfg=cfg, dp=dp)
    rank = od.parallel_config.world_size - 1
    created = _fake_state(monkeypatch, od, rank)
    _initialize(od)
    groups = state.get_expert_parallel_group_ranks()
    size = ep or sp
    assert sorted(rank for members in groups for rank in members) == list(range(od.parallel_config.world_size))
    for members in groups:
        assert len(members) == size
        # Fixed TP/PP/CFG/DP coordinates; only a contiguous part of the SP axis varies.
        assert len({(r % tp, r // (tp * sp)) for r in members}) == 1
        assert [r // tp % sp for r in members] == list(range(members[0] // tp % sp, members[0] // tp % sp + size))
    ep_group = state.vllm_parallel_state._EP
    assert ep_group.local_group == next(members for members in groups if rank in members)
    assert state.vllm_parallel_state._DP is state._DP
    assert state._DP.world_size == dp
    assert state.vllm_parallel_state._PCP is None
    state.destroy_model_parallel()
    assert state.vllm_parallel_state._EP is None
    assert state._EXPERT_PARALLEL_GROUP_RANKS is None
    for group in created:
        group.destroy.assert_called_once()


@pytest.mark.cpu
@pytest.mark.parametrize("size,head", [(3, True), (8, True), (2, False)])
def test_invalid_group_size_fails_before_creation(monkeypatch, size, head):
    monkeypatch.setitem(
        metadata._DIFFUSION_MODEL_METADATA,
        _MODEL,
        metadata.DiffusionModelMetadata(expert_parallel_style="head" if head else "vllm"),
    )
    od = _config(size)
    od.tf_model_config = {"num_experts": 8}
    created = _fake_state(monkeypatch, od)
    with pytest.raises(ValueError, match="expert_parallel_size"):
        _initialize(od)
    assert created == []
    assert state.vllm_parallel_state._EP is None


@pytest.mark.cpu
def test_failed_head_group_creation_uses_existing_rollback(monkeypatch, head_model):
    od = _config(2)
    created = _fake_state(monkeypatch, od, fail_ep=True)
    with pytest.raises(RuntimeError, match="EP creation failed"):
        _initialize(od)
    assert created
    assert state._EXPERT_PARALLEL_GROUP_RANKS is None
    assert state.vllm_parallel_state._EP is None
    for group in created:
        group.destroy.assert_called_once()


def _group_worker(rank, rendezvous, device_type):
    torch.set_num_threads(1)
    backend = "gloo" if device_type == "cpu" else "mccl"
    if device_type != "cpu":
        torch.accelerator.set_device_index(rank)
    dist.init_process_group(backend, init_method=rendezvous, world_size=4, rank=rank, timeout=timedelta(seconds=60))
    metadata._DIFFUSION_MODEL_METADATA[_MODEL] = metadata.DiffusionModelMetadata(expert_parallel_style="head")
    try:
        state.init_distributed_environment(world_size=4, rank=rank, local_rank=rank, backend=backend)
        for tp, sp, cfg, dp, ep_size in (
            (1, 2, 2, 1, 2),
            (1, 4, 1, 1, 2),
            (1, 2, 1, 2, 2),
            (2, 2, 1, 1, 2),
            (1, 4, 1, 1, 1),
        ):
            od = _config(ep_size, tp=tp, sp=sp, cfg=cfg, dp=dp)
            with set_forward_context(omni_diffusion_config=od):
                _initialize(od, backend)
                ep = state.vllm_parallel_state.get_ep_group()
                assert ep.world_size == ep_size
                assert ep.ranks in state.get_expert_parallel_group_ranks()
                for dtype in (torch.float32, torch.bfloat16, torch.int64):
                    tensor = torch.full((ep_size,), rank, dtype=dtype, device=device_type)
                    output = torch.empty_like(tensor)
                    dist.all_to_all_single(output, tensor, group=ep.device_group)
                    assert output.cpu().tolist() == ep.ranks
                state.destroy_model_parallel()
                assert state.vllm_parallel_state._EP is None
                assert state._EXPERT_PARALLEL_GROUP_RANKS is None
            dist.barrier()
    finally:
        state.destroy_model_parallel()
        state.destroy_distributed_environment()


@pytest.mark.cpu
@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="requires Gloo")
def test_real_four_rank_head_ep_reinitialization(tmp_path):
    mp.spawn(_group_worker, args=(f"file://{tmp_path / 'head-gloo'}", "cpu"), nprocs=4, join=True)


@pytest.mark.musa
def test_real_four_rank_musa_head_ep_reinitialization(tmp_path):
    if not hasattr(torch, "musa") or not torch.musa.is_available() or torch.musa.device_count() < 4:
        pytest.skip("requires four MUSA devices")
    mp.spawn(_group_worker, args=(f"file://{tmp_path / 'head-mccl'}", "musa"), nprocs=4, join=True)
