# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm_omni.distributed.omni_coordinator import CostAwareBalancer
from vllm_omni.engine.stage_runtime import _build_load_balancer_factory

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_cost_aware_policy_builds_cost_aware_balancer() -> None:
    factory = _build_load_balancer_factory("cost-aware")

    assert factory is CostAwareBalancer
    assert isinstance(factory(), CostAwareBalancer)
