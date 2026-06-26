# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for applying strategy specs onto merged stage configs."""

from __future__ import annotations

import pytest

from vllm_omni.config.composable_parallel import (
    Broadcast,
    FanInByStage,
    MeshAxisSpec,
    RouteByStage,
    StrategyApplyError,
    StrategySpec,
    TakeRank,
    apply_strategy_specs,
)
from vllm_omni.config.stage_config import StageConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _tp(size: int) -> StrategySpec:
    return StrategySpec("tp", MeshAxisSpec("tp", size), Broadcast(), TakeRank())


def _stage_replica(size: int, policy: str = "round_robin") -> StrategySpec:
    return StrategySpec("stage_replica", MeshAxisSpec("stage_replica", size), RouteByStage(policy), FanInByStage())


def _stage(stage_id: int, model_stage: str, engine_args=None, runtime=None) -> StageConfig:
    return StageConfig(
        stage_id=stage_id,
        model_stage=model_stage,
        yaml_engine_args=dict(engine_args or {}),
        yaml_runtime=dict(runtime or {"num_replicas": 1}),
    )


def _qwen_stages() -> list[StageConfig]:
    return [
        _stage(0, "thinker"),
        _stage(1, "talker"),
        _stage(2, "code2wav"),
    ]


def test_apply_tp_by_role():
    stages = _qwen_stages()
    apply_strategy_specs(stages, {"thinker": [_tp(2)]})
    assert stages[0].yaml_engine_args["tensor_parallel_size"] == 2
    # untouched roles keep their config
    assert "tensor_parallel_size" not in stages[1].yaml_engine_args


def test_apply_by_model_stage():
    stages = _qwen_stages()
    apply_strategy_specs(stages, {"thinker": [_tp(2)]})
    assert stages[0].yaml_engine_args["tensor_parallel_size"] == 2


def test_apply_stage_replica_sets_num_replicas_and_surfaces_lb():
    stages = _qwen_stages()
    result = apply_strategy_specs(stages, {"talker": [_stage_replica(2, "round_robin")]})
    assert stages[1].yaml_runtime["num_replicas"] == 2
    assert result.omni_lb_policy == "round-robin"


def test_only_declared_axes_are_written():
    stages = _qwen_stages()
    # strategy declares only stage_replica -> tp must not be forced.
    apply_strategy_specs(stages, {"talker": [_stage_replica(2)]})
    assert "tensor_parallel_size" not in stages[1].yaml_engine_args


def test_conflict_on_explicit_tp():
    stages = _qwen_stages()
    stages[0].yaml_engine_args["tensor_parallel_size"] = 4
    with pytest.raises(StrategyApplyError):
        apply_strategy_specs(stages, {"thinker": [_tp(2)]})


def test_equal_explicit_value_is_noop():
    stages = _qwen_stages()
    stages[0].yaml_engine_args["tensor_parallel_size"] = 2
    apply_strategy_specs(stages, {"thinker": [_tp(2)]})
    assert stages[0].yaml_engine_args["tensor_parallel_size"] == 2


def test_explicit_none_conflicts_with_derived_value():
    # An explicit YAML ``null`` (``tensor_parallel_size: null``) is a *present*
    # value, not a missing key, so a strategy deriving a non-None size must raise
    # rather than silently clobber the explicit None.
    stages = _qwen_stages()
    stages[0].yaml_engine_args["tensor_parallel_size"] = None
    with pytest.raises(StrategyApplyError):
        apply_strategy_specs(stages, {"thinker": [_tp(2)]})


def test_missing_key_is_filled_without_conflict():
    # A genuinely absent key (never set in the YAML) is filled by the strategy
    # and must NOT raise — the contrast case to an explicit None.
    stages = _qwen_stages()
    assert "tensor_parallel_size" not in stages[0].yaml_engine_args
    apply_strategy_specs(stages, {"thinker": [_tp(2)]})
    assert stages[0].yaml_engine_args["tensor_parallel_size"] == 2


def test_num_replicas_conflict():
    stages = _qwen_stages()
    stages[1].yaml_runtime["num_replicas"] = 3
    with pytest.raises(StrategyApplyError):
        apply_strategy_specs(stages, {"talker": [_stage_replica(2)]})


def test_device_count_ok_template():
    stages = _qwen_stages()
    stages[0].yaml_runtime["devices"] = "0,1"
    apply_strategy_specs(stages, {"thinker": [_tp(2)]})
    assert stages[0].yaml_engine_args["tensor_parallel_size"] == 2


def test_device_count_ok_pool():
    # tp=2 -> world=2; 2 replicas -> pool of 4 device ids is valid.
    stages = _qwen_stages()
    stages[1].yaml_runtime["devices"] = "0,1,2,3"
    apply_strategy_specs(stages, {"talker": [_tp(2), _stage_replica(2)]})
    assert stages[1].yaml_runtime["num_replicas"] == 2


def test_device_count_mismatch():
    stages = _qwen_stages()
    stages[0].yaml_runtime["devices"] = "0,1,2"
    with pytest.raises(StrategyApplyError):
        apply_strategy_specs(stages, {"thinker": [_tp(2)]})


def test_unknown_role_raises():
    stages = _qwen_stages()
    with pytest.raises(StrategyApplyError):
        apply_strategy_specs(stages, {"nonexistent": [_tp(2)]})


def test_conflicting_lb_policy_across_roles():
    stages = _qwen_stages()
    with pytest.raises(StrategyApplyError):
        apply_strategy_specs(
            stages,
            {
                "talker": [_stage_replica(2, "round_robin")],
                "code2wav": [_stage_replica(2, "least_queue")],
            },
        )
