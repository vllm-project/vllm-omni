# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm_omni.distributed.omni_coordinator import CostAwareBalancer
from vllm_omni.engine.stage_runtime import StageRuntime, _build_load_balancer_factory

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_cost_aware_policy_builds_cost_aware_balancer() -> None:
    factory = _build_load_balancer_factory("cost-aware")

    assert factory is CostAwareBalancer
    assert isinstance(factory(), CostAwareBalancer)


def test_local_stage_runtime_attaches_cost_aware_balancer() -> None:
    runtime = StageRuntime(
        stage_configs=[],
        model="model",
        config_path=None,
        stage_init_timeout=1,
        async_chunk=False,
        omni_lb_policy="cost-aware",
    )
    client = SimpleNamespace(stage_type="diffusion", final_output=False)
    plan = SimpleNamespace(
        stage_idx=0,
        stage_id=0,
        replicas=[SimpleNamespace(metadata=SimpleNamespace(stage_type="diffusion"))],
    )

    pools = runtime._assemble_stage_pools([plan], {0: [client]})

    assert isinstance(pools[0]._lb, CostAwareBalancer)
