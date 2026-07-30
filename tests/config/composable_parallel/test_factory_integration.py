# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration tests: strategy overlay on the real merge_pipeline_deploy seam."""

from __future__ import annotations

import logging
from pathlib import Path

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
from vllm_omni.config.config_factory import StageConfigFactory
from vllm_omni.config.pipeline_registry import OMNI_PIPELINES
from vllm_omni.config.stage_config import load_deploy_config, merge_pipeline_deploy

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_DEPLOY = Path(__file__).parents[3] / "vllm_omni" / "deploy" / "qwen2_5_omni.yaml"


def _tp(size: int) -> StrategySpec:
    return StrategySpec("tp", MeshAxisSpec("tp", size), Broadcast(), TakeRank())


def _stage_replica(size: int, policy: str = "round_robin") -> StrategySpec:
    return StrategySpec("stage_replica", MeshAxisSpec("stage_replica", size), RouteByStage(policy), FanInByStage())


def _qwen_stages():
    pipeline = OMNI_PIPELINES["qwen2_5_omni"]
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
    with pytest.raises(StrategyApplyError):
        apply_strategy_specs(stages, {"thinker": [_tp(2)]})


def test_overlay_stage_replica_on_talker():
    stages = _qwen_stages()
    result = apply_strategy_specs(stages, {"talker": [_stage_replica(2, "round_robin")]})
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


def _resolved(stages, role):
    """Return a role's fully-resolved (post-CLI) OmegaConf stage config."""
    stage = next(s for s in stages if s.model_stage == role)
    return stage.to_omegaconf()


def test_device_check_survives_cli_override():
    # Strategy replicates the talker (1-GPU template -> valid at apply time),
    # but a CLI --stage_1_devices with 3 ids must NOT slip past the device
    # guard: effective world=1, replicas=2 admits only 1 or 2 device ids.
    pipeline_cfg = OMNI_PIPELINES["qwen2_5_omni"]
    with pytest.raises(StrategyApplyError):
        StageConfigFactory._create_legacy_from_registry(
            pipeline_cfg,
            cli_overrides={"stage_1_devices": "0,1,2"},
            strategy_specs={"talker": [_stage_replica(2, "round_robin")]},
        )


def test_cli_overrides_strategy_with_warning():
    # Strategy derives num_replicas=2 for the talker; a CLI override to 3 wins
    # (it is the most explicit user action) but must be surfaced loudly rather
    # than silently. The resulting layout (1 template device) stays valid.
    #
    # vLLM's logger sets propagate=False, so attach a handler directly to it
    # rather than relying on pytest's caplog (which listens on the root logger).
    messages: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            messages.append(record.getMessage())

    log = logging.getLogger("vllm_omni.config.config_factory")
    handler = _Capture(level=logging.WARNING)
    log.addHandler(handler)
    try:
        pipeline_cfg = OMNI_PIPELINES["qwen2_5_omni"]
        stages, _ = StageConfigFactory._create_legacy_from_registry(
            pipeline_cfg,
            cli_overrides={"stage_1_num_replicas": 3},
            strategy_specs={"talker": [_stage_replica(2, "round_robin")]},
        )
    finally:
        log.removeHandler(handler)

    # CLI value wins in the resolved config.
    assert _resolved(stages, "talker").runtime.num_replicas == 3
    # ...and the override was warned about, naming the conflicting field.
    assert any("num_replicas" in m and "overrides the strategy-derived" in m for m in messages)
