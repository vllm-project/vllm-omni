# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""The legacy CLI overlay must keep the scheduler class and mode consistent."""

import pytest

from vllm_omni.config.stage_config import StageConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

SYNC = "vllm_omni.core.sched.omni_ar_scheduler.OmniARScheduler"
ASYNC = "vllm_omni.core.sched.omni_ar_scheduler.OmniARAsyncScheduler"


@pytest.mark.parametrize("enabled,original,expected", [(True, SYNC, ASYNC), (False, ASYNC, SYNC)])
def test_cli_async_mode_reselects_builtin_ar_scheduler(enabled, original, expected):
    stage = StageConfig(
        stage_id=0,
        model_stage="llm",
        worker_type="ar",
        scheduler_cls=original,
        yaml_engine_args={"async_scheduling": not enabled},
        runtime_overrides={"async_scheduling": enabled},
    )
    config = stage.to_omegaconf()
    assert config.engine_args.async_scheduling is enabled
    assert config.engine_args.scheduler_cls == expected
    # Resolution must not rewrite the user's input configuration.
    assert stage.scheduler_cls == original
    assert stage.yaml_engine_args == {"async_scheduling": not enabled}


@pytest.mark.parametrize(
    "scheduler,overrides,expected",
    [
        (SYNC, {}, SYNC),
        (ASYNC, {"async_scheduling": None}, ASYNC),
        ("custom.Scheduler", {"async_scheduling": True}, "custom.Scheduler"),
        (SYNC, {"async_scheduling": True, "scheduler_cls": "custom.Scheduler"}, "custom.Scheduler"),
        (SYNC, {"async_scheduling": True, "scheduler_cls": None}, ASYNC),
    ],
)
def test_async_mode_preserves_custom_scheduler_and_unspecified_mode(scheduler, overrides, expected):
    stage = StageConfig(
        stage_id=0,
        model_stage="llm",
        worker_type="ar",
        scheduler_cls=scheduler,
        runtime_overrides=overrides,
    )
    assert stage.to_omegaconf().engine_args.scheduler_cls == expected
