# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for applying strategy specs onto pipeline/deploy configs."""

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
from vllm_omni.config.stage_config import DeployConfig, PipelineConfig, StageDeployConfig, StagePipelineConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _tp(size: int) -> StrategySpec:
    return StrategySpec("tp", MeshAxisSpec("tp", size), Broadcast(), TakeRank())


def _stage_replica(size: int, policy: str = "round_robin") -> StrategySpec:
    return StrategySpec("stage_replica", MeshAxisSpec("stage_replica", size), RouteByStage(policy), FanInByStage())


def _qwen_stages() -> tuple[PipelineConfig, DeployConfig]:
    pipeline = PipelineConfig(
        model_type="test",
        stages=tuple(
            StagePipelineConfig(stage_id=i, model_stage=role, final_output=True)
            for i, role in enumerate(("thinker", "talker", "code2wav"))
        ),
    )
    return pipeline, DeployConfig(stages=[StageDeployConfig(stage_id=i) for i in range(3)])


def test_apply_tp_by_role():
    pipeline, deploy = _qwen_stages()
    stages = deploy.stages
    apply_strategy_specs(pipeline, deploy, {"thinker": [_tp(2)]})
    assert stages[0].tensor_parallel_size == 2
    # untouched roles keep their config
    assert stages[1].tensor_parallel_size is None


def test_apply_by_model_stage():
    pipeline, deploy = _qwen_stages()
    stages = deploy.stages
    apply_strategy_specs(pipeline, deploy, {"thinker": [_tp(2)]})
    assert stages[0].tensor_parallel_size == 2


def test_apply_stage_replica_sets_num_replicas_and_surfaces_lb():
    pipeline, deploy = _qwen_stages()
    stages = deploy.stages
    result = apply_strategy_specs(pipeline, deploy, {"talker": [_stage_replica(2, "round_robin")]})
    assert stages[1].num_replicas == 2
    assert result.omni_lb_policy == "round-robin"


def test_only_declared_axes_are_written():
    pipeline, deploy = _qwen_stages()
    stages = deploy.stages
    # strategy declares only stage_replica -> tp must not be forced.
    apply_strategy_specs(pipeline, deploy, {"talker": [_stage_replica(2)]})
    assert stages[1].tensor_parallel_size is None


def test_conflict_on_explicit_tp():
    pipeline, deploy = _qwen_stages()
    stages = deploy.stages
    stages[0].tensor_parallel_size = 4
    with pytest.raises(StrategyApplyError):
        apply_strategy_specs(pipeline, deploy, {"thinker": [_tp(2)]})


def test_conflict_on_nested_parallel_config_tp():
    pipeline, deploy = _qwen_stages()
    deploy.stages[0].engine_extras["parallel_config"] = {"tensor_parallel_size": 4}
    with pytest.raises(StrategyApplyError):
        apply_strategy_specs(pipeline, deploy, {"thinker": [_tp(2)]})


def test_equal_explicit_value_is_noop():
    pipeline, deploy = _qwen_stages()
    stages = deploy.stages
    stages[0].tensor_parallel_size = 2
    apply_strategy_specs(pipeline, deploy, {"thinker": [_tp(2)]})
    assert stages[0].tensor_parallel_size == 2


def test_explicit_none_conflicts_with_derived_value():
    # An explicit ``engine_extras`` null (``tensor_parallel_size: null``) is a *present*
    # value, not a missing key, so a strategy deriving a non-None size must raise
    # rather than silently clobber the explicit None.
    pipeline, deploy = _qwen_stages()
    stages = deploy.stages
    stages[0].engine_extras["tensor_parallel_size"] = None
    with pytest.raises(StrategyApplyError):
        apply_strategy_specs(pipeline, deploy, {"thinker": [_tp(2)]})


def test_missing_key_is_filled_without_conflict():
    # A genuinely absent key (never set in the YAML) is filled by the strategy
    # and must NOT raise — the contrast case to an explicit None.
    pipeline, deploy = _qwen_stages()
    stages = deploy.stages
    assert stages[0].tensor_parallel_size is None
    apply_strategy_specs(pipeline, deploy, {"thinker": [_tp(2)]})
    assert stages[0].tensor_parallel_size == 2


def test_num_replicas_conflict():
    pipeline, deploy = _qwen_stages()
    stages = deploy.stages
    stages[1].num_replicas = 3
    with pytest.raises(StrategyApplyError):
        apply_strategy_specs(pipeline, deploy, {"talker": [_stage_replica(2)]})


def test_device_count_ok_template():
    pipeline, deploy = _qwen_stages()
    stages = deploy.stages
    stages[0].devices = "0,1"
    apply_strategy_specs(pipeline, deploy, {"thinker": [_tp(2)]})
    assert stages[0].tensor_parallel_size == 2


def test_device_count_ok_pool():
    # tp=2 -> world=2; 2 replicas -> pool of 4 device ids is valid.
    pipeline, deploy = _qwen_stages()
    stages = deploy.stages
    stages[1].devices = "0,1,2,3"
    apply_strategy_specs(pipeline, deploy, {"talker": [_tp(2), _stage_replica(2)]})
    assert stages[1].num_replicas == 2


def test_device_count_mismatch():
    pipeline, deploy = _qwen_stages()
    stages = deploy.stages
    stages[0].devices = "0,1,2"
    with pytest.raises(StrategyApplyError):
        apply_strategy_specs(pipeline, deploy, {"thinker": [_tp(2)]})


def test_unknown_role_raises():
    pipeline, deploy = _qwen_stages()
    with pytest.raises(StrategyApplyError):
        apply_strategy_specs(pipeline, deploy, {"nonexistent": [_tp(2)]})


def test_conflicting_lb_policy_across_roles():
    pipeline, deploy = _qwen_stages()
    with pytest.raises(StrategyApplyError):
        apply_strategy_specs(
            pipeline,
            deploy,
            {
                "talker": [_stage_replica(2, "round_robin")],
                "code2wav": [_stage_replica(2, "least_queue")],
            },
        )
