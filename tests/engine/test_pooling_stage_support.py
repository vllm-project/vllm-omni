# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for running a pooling model as a pipeline stage.

Three places in the pipeline layer assumed every stage was generative. Each
one turned a pooling stage into a runtime failure rather than a configuration
error, so each gets a regression test here.
"""

from types import SimpleNamespace

import pytest
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams

from vllm_omni.engine.orchestrator import Orchestrator
from vllm_omni.engine.stage_init_utils import _is_pooling_stage

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TestIsPoolingStage:
    """Detection keys off vLLM's own runner signal, not execution_type."""

    def test_true_when_engine_args_declare_pooling(self):
        assert _is_pooling_stage(SimpleNamespace(engine_args={"runner": "pooling"}))

    def test_true_when_declared_on_the_stage_itself(self):
        assert _is_pooling_stage(SimpleNamespace(engine_args=None, runner="pooling"))

    def test_case_insensitive(self):
        assert _is_pooling_stage(SimpleNamespace(engine_args={"runner": "POOLING"}))

    def test_false_for_generate_runner(self):
        assert not _is_pooling_stage(SimpleNamespace(engine_args={"runner": "generate"}))

    def test_false_when_unset(self):
        assert not _is_pooling_stage(SimpleNamespace(engine_args={}))

    def test_mapping_shaped_stage_config(self):
        assert _is_pooling_stage({"engine_args": {"runner": "pooling"}})


class TestStageSupportedTasks:
    """The orchestrator used to hard-code ("generate",) when building a
    downstream request, so vLLM's input processor rejected PoolingParams with
    "This model does not support pooling" before the stage ever saw them."""

    @staticmethod
    def _orchestrator(runner, supported):
        orch = object.__new__(Orchestrator)
        orch.stage_pools = [
            SimpleNamespace(
                stage_vllm_config=SimpleNamespace(
                    model_config=SimpleNamespace(runner_type=runner, supported_tasks=supported)
                )
            )
        ]
        return orch

    def test_generative_stage_still_gets_generate(self):
        orch = self._orchestrator("generate", ("generate",))
        assert orch._stage_supported_tasks(0) == ("generate",)

    def test_pooling_stage_gets_its_declared_tasks(self):
        orch = self._orchestrator("pooling", ("token_classify",))
        assert orch._stage_supported_tasks(0) == ("token_classify",)

    def test_pooling_stage_without_resolved_tasks_falls_back(self):
        """supported_tasks is resolved inside the stage's engine process, so it
        can still be unset on the config the orchestrator holds."""
        orch = self._orchestrator("pooling", None)
        assert orch._stage_supported_tasks(0) == ("token_classify",)

    def test_missing_model_config_is_generative(self):
        orch = object.__new__(Orchestrator)
        orch.stage_pools = [SimpleNamespace(stage_vllm_config=SimpleNamespace(model_config=None))]
        assert orch._stage_supported_tasks(0) == ("generate",)


class TestPoolingParamsSelection:
    """A pooling stage needs PoolingParams; building SamplingParams for it is
    what the engine rejects."""

    def test_pooling_params_accept_the_configured_task(self):
        assert PoolingParams(task="token_classify").task == "token_classify"

    def test_sampling_and_pooling_are_distinct_types(self):
        assert not isinstance(PoolingParams(task="token_classify"), SamplingParams)
