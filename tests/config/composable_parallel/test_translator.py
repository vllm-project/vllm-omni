# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for StrategySpec -> OmniParallelConfig translation."""

from __future__ import annotations

import dataclasses

import pytest

from vllm_omni.config.composable_parallel import (
    AxisTranslationError,
    Broadcast,
    FanInByStage,
    MeshAxisSpec,
    PartitionByHash,
    PipelineMicrobatch,
    RouteByStage,
    StrategySpec,
    TakeRank,
    Union,
    translate_strategy_stack,
)
from vllm_omni.config.composable_parallel.aggregation import StitchPipeline

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _tp(size: int) -> StrategySpec:
    return StrategySpec("tp", MeshAxisSpec("tp", size), Broadcast(), TakeRank())


def _dp(size: int, policy: str = "round_robin") -> StrategySpec:
    return StrategySpec("dp", MeshAxisSpec("dp", size), RouteByStage(policy), Union())


def _pp(size: int) -> StrategySpec:
    return StrategySpec("pp", MeshAxisSpec("pp", size), PipelineMicrobatch(), StitchPipeline())


def _ep(size: int) -> StrategySpec:
    return StrategySpec("ep", MeshAxisSpec("ep", size), Broadcast(), Union())


def _stage_replica(size: int, policy: str = "round_robin") -> StrategySpec:
    return StrategySpec("stage_replica", MeshAxisSpec("stage_replica", size), RouteByStage(policy), FanInByStage())


def test_tp_only():
    cfg = translate_strategy_stack([_tp(2)])
    assert cfg.tensor_parallel_size == 2
    assert cfg.data_parallel_size == 1
    assert cfg.world_size == 2
    assert cfg.stage_replica_size == 1
    assert cfg.l1_owners == {"tp": "engine"}


def test_dp_is_engine_owned():
    cfg = translate_strategy_stack([_dp(2)])
    assert cfg.data_parallel_size == 2
    assert cfg.world_size == 2
    # engine DP does not emit an omni LB policy (vLLM's internal DP LB is used).
    assert cfg.omni_lb_policy is None
    assert cfg.as_engine_kwargs().get("omni_lb_policy") is None
    assert cfg.l1_owners["dp"] == "engine"


def test_tp_times_dp_world_size():
    cfg = translate_strategy_stack([_tp(2), _dp(2)])
    assert cfg.world_size == 4
    assert cfg.tensor_parallel_size == 2
    assert cfg.data_parallel_size == 2


def test_stage_replica_is_delegated():
    cfg = translate_strategy_stack([_stage_replica(3, "least_queue")])
    # stage_replica is NOT a world dimension.
    assert cfg.world_size == 1
    assert cfg.stage_replica_size == 3
    assert cfg.omni_lb_policy == "least-queue-length"
    assert cfg.l1_owners["stage_replica"] == "delegated"
    # stage_replica_size is not an engine kwarg; it's a per-stage deploy knob.
    assert "stage_replica_size" not in cfg.as_engine_kwargs()


def test_stage_replica_lb_policy_only_when_replicated():
    cfg = translate_strategy_stack([_stage_replica(1)])
    assert cfg.stage_replica_size == 1
    assert cfg.as_engine_kwargs().get("omni_lb_policy") is None


def test_tp_with_stage_replica():
    cfg = translate_strategy_stack([_tp(2), _stage_replica(2)])
    assert cfg.world_size == 2  # tp only
    assert cfg.tensor_parallel_size == 2
    assert cfg.stage_replica_size == 2
    assert cfg.omni_lb_policy == "round-robin"


def test_ep_must_match_tp_dp_product():
    cfg = translate_strategy_stack([_tp(2), _ep(2)])
    assert cfg.enable_expert_parallel is True
    assert cfg.as_engine_kwargs()["enable_expert_parallel"] is True


def test_ep_mismatch_raises():
    with pytest.raises(Exception) as exc:
        translate_strategy_stack([_tp(2), _ep(4)])
    assert "must equal" in str(exc.value)


def test_duplicate_kind_rejected():
    with pytest.raises(AxisTranslationError):
        translate_strategy_stack([_tp(2), _tp(2)])


def test_unsupported_kind_rejected():
    spec = StrategySpec("sp", MeshAxisSpec("sp_ulysses", 2), Broadcast(), TakeRank())
    with pytest.raises(AxisTranslationError):
        translate_strategy_stack([spec])


def test_dp_hash_routing_rejected():
    spec = StrategySpec("dp", MeshAxisSpec("dp", 2), PartitionByHash(), Union())
    with pytest.raises(NotImplementedError):
        translate_strategy_stack([spec])


def test_dp_invalid_routing_policy_rejected():
    spec = StrategySpec("dp", MeshAxisSpec("dp", 2), RouteByStage("bogus"), Union())
    with pytest.raises(AxisTranslationError):
        translate_strategy_stack([spec])


def test_lb_policy_never_in_engine_kwargs_even_when_replicated():
    # omni_lb_policy is pipeline-wide (applied at engine construction), not a
    # per-stage engine arg, so it must never leak into as_engine_kwargs().
    cfg = translate_strategy_stack([_tp(2), _stage_replica(2, "least_queue")])
    assert cfg.omni_lb_policy == "least-queue-length"
    assert "omni_lb_policy" not in cfg.as_engine_kwargs()


def test_stage_replica_hash_routing_rejected():
    spec = StrategySpec("sr", MeshAxisSpec("stage_replica", 2), RouteByStage("hash"), FanInByStage())
    with pytest.raises(NotImplementedError):
        translate_strategy_stack([spec])


def test_dp_wrong_owner_rejected():
    spec = StrategySpec(
        "dp",
        MeshAxisSpec("dp", 2),
        RouteByStage("round_robin"),
        Union(),
        shard_extension={"l1_owner": "delegated"},
    )
    with pytest.raises(AxisTranslationError):
        translate_strategy_stack([spec])


def test_stage_replica_wrong_owner_rejected():
    spec = StrategySpec(
        "sr",
        MeshAxisSpec("stage_replica", 2),
        RouteByStage("round_robin"),
        FanInByStage(),
        shard_extension={"l1_owner": "engine"},
    )
    with pytest.raises(AxisTranslationError):
        translate_strategy_stack([spec])


@pytest.mark.parametrize(
    "specs",
    [
        [_tp(2)],
        [_dp(2)],
        [_tp(2), _dp(2)],
        [_tp(2), _ep(2)],
        [_tp(2), _stage_replica(2)],
        [_pp(2)],
    ],
)
def test_as_engine_kwargs_are_valid_omni_engine_args(specs):
    # Guard against desync: every key emitted by as_engine_kwargs() must be a
    # real OmniEngineArgs field, otherwise constructing the per-stage engine args
    # would explode at runtime. Keep this CPU-only by checking field names rather
    # than instantiating the engine.
    from vllm_omni.engine.arg_utils import OmniEngineArgs

    valid_fields = {f.name for f in dataclasses.fields(OmniEngineArgs)}
    cfg = translate_strategy_stack(specs)
    kwargs = cfg.as_engine_kwargs()
    unknown = set(kwargs) - valid_fields
    assert not unknown, f"as_engine_kwargs emitted non-OmniEngineArgs fields: {unknown}"
