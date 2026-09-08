# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the vendored spec/pattern data types."""

from __future__ import annotations

import pytest

from vllm_omni.config.composable_parallel.aggregation import AggregationPattern, TakeRank
from vllm_omni.config.composable_parallel.routing import RoutingPattern
from vllm_omni.config.composable_parallel.spec import MeshAxisSpec

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_mesh_axis_rejects_bad_kind():
    with pytest.raises(ValueError):
        MeshAxisSpec(kind="not_a_kind", size=2)  # type: ignore[arg-type]


def test_mesh_axis_rejects_nonpositive_size():
    with pytest.raises(ValueError):
        MeshAxisSpec(kind="tp", size=0)


def test_routing_pattern_is_closed():
    with pytest.raises(TypeError):

        class RogueRouting(RoutingPattern):  # noqa: F811
            pass


def test_aggregation_pattern_is_closed():
    with pytest.raises(TypeError):

        class RogueAggregation(AggregationPattern):  # noqa: F811
            pass


def test_take_rank_type_guard():
    with pytest.raises(TypeError):
        TakeRank(rank=True)  # bool is not a valid rank
