# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration tests: strategy overlay on the real merge_pipeline_deploy seam."""

from __future__ import annotations

from pathlib import Path

import pytest

from vllm_omni.config.composable_parallel import (
    Broadcast,
    FanInByStage,
    MeshAxisSpec,
    RouteByStage,
    StrategyDeviceMismatchError,
    StrategySpec,
    TakeRank,
    apply_strategy_specs,
)
from vllm_omni.config.stage_config import (
    _PIPELINE_REGISTRY,
    load_deploy_config,
    merge_pipeline_deploy,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_DEPLOY = Path(__file__).parents[3] / "vllm_omni" / "deploy" / "qwen2_5_omni.yaml"


def _tp(size: int) -> StrategySpec:
    return StrategySpec("tp", MeshAxisSpec("tp", size), Broadcast(), TakeRank())


def _stage_replica(size: int, policy: str = "round_robin") -> StrategySpec:
    return StrategySpec(
        "stage_replica", MeshAxisSpec("stage_replica", size), RouteByStage(policy), FanInByStage()
    )


def _qwen_stages():
    pipeline = _PIPELINE_REGISTRY["qwen2_5_omni"]
    if not _DEPLOY.exists():
        pytest.skip("qwen2_5_omni deploy config not found")
    deploy = load_deploy_config(_DEPLOY)
    return merge_pipeline_deploy(pipeline, deploy)


def _stage(stages, role):
    return next(s for s in stages if s.model_stage == role)


def test_overlay_tp_on_thinker():
    stages = _qwen_stages()
    # The bundled deploy pins the thinker to one GPU; a TP=2 strategy needs a
    # matching 2-GPU layout (mirrors what a TP2 deploy would declare).
    _stage(stages, "thinker").yaml_runtime["devices"] = "0,1"
    result = apply_strategy_specs(stages, {"thinker": [_tp(2)]})
    assert _stage(result.stages, "thinker").yaml_engine_args["tensor_parallel_size"] == 2


def test_device_guard_rejects_tp_on_single_gpu_deploy():
    # The bundled deploy pins the thinker to a single GPU, so the pre-spawn
    # device check must refuse a TP=2 strategy on it.
    stages = _qwen_stages()
    with pytest.raises(StrategyDeviceMismatchError):
        apply_strategy_specs(stages, {"thinker": [_tp(2)]})


def test_overlay_stage_replica_on_talker():
    stages = _qwen_stages()
    result = apply_strategy_specs(
        stages, {"talker": [_stage_replica(2, "round_robin")]}
    )
    assert _stage(result.stages, "talker").yaml_runtime["num_replicas"] == 2
    assert result.omni_lb_policy == "round-robin"


def test_overlay_mixed_roles():
    stages = _qwen_stages()
    _stage(stages, "thinker").yaml_runtime["devices"] = "0,1"
    result = apply_strategy_specs(
        stages,
        {
            "thinker": [_tp(2)],
            "talker": [_stage_replica(2, "round_robin")],
            "code2wav": [_stage_replica(2, "round_robin")],
        },
    )
    by_role = {s.model_stage: s for s in result.stages}
    assert by_role["thinker"].yaml_engine_args["tensor_parallel_size"] == 2
    assert by_role["talker"].yaml_runtime["num_replicas"] == 2
    assert by_role["code2wav"].yaml_runtime["num_replicas"] == 2
    assert result.omni_lb_policy == "round-robin"
