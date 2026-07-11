# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for parsing/loading strategy files."""

from __future__ import annotations

import pytest

from vllm_omni.config.composable_parallel.routing import Broadcast, RouteByStage
from vllm_omni.config.composable_parallel.strategy_loader import (
    StrategyLoadError,
    load_strategy_specs,
    parse_strategy_specs,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_parse_basic():
    data = {
        "strategies": {
            "thinker": [{"axis": "tp", "size": 2}],
            "talker": [{"axis": "stage_replica", "size": 2, "routing": "round_robin"}],
        }
    }
    specs = parse_strategy_specs(data)
    assert set(specs) == {"thinker", "talker"}

    (tp_spec,) = specs["thinker"]
    assert tp_spec.mesh_axis.kind == "tp"
    assert tp_spec.mesh_axis.size == 2
    assert isinstance(tp_spec.routing, Broadcast)

    (sr_spec,) = specs["talker"]
    assert sr_spec.mesh_axis.kind == "stage_replica"
    assert isinstance(sr_spec.routing, RouteByStage)
    assert sr_spec.routing.routing_policy == "round_robin"


def test_parse_without_strategies_key():
    # A bare role mapping (no top-level "strategies") is also accepted.
    specs = parse_strategy_specs({"thinker": [{"axis": "tp", "size": 4}]})
    assert specs["thinker"][0].mesh_axis.size == 4


def test_l1_owner_goes_to_shard_extension():
    specs = parse_strategy_specs({"talker": [{"axis": "stage_replica", "size": 2, "l1_owner": "delegated"}]})
    assert specs["talker"][0].shard_extension["l1_owner"] == "delegated"


def test_missing_axis_raises():
    with pytest.raises(StrategyLoadError):
        parse_strategy_specs({"thinker": [{"size": 2}]})


def test_missing_size_raises():
    with pytest.raises(StrategyLoadError):
        parse_strategy_specs({"thinker": [{"axis": "tp"}]})


def test_routing_on_non_policy_axis_raises():
    with pytest.raises(StrategyLoadError):
        parse_strategy_specs({"thinker": [{"axis": "tp", "size": 2, "routing": "random"}]})


def test_bad_size_raises():
    with pytest.raises(StrategyLoadError):
        parse_strategy_specs({"thinker": [{"axis": "tp", "size": "two"}]})


def test_entries_must_be_list():
    with pytest.raises(StrategyLoadError):
        parse_strategy_specs({"thinker": {"axis": "tp", "size": 2}})


def test_non_mapping_entry_raises():
    # A list whose elements are not mappings (e.g. a bare string) must raise a
    # StrategyLoadError, not an opaque TypeError from dict(entry).
    with pytest.raises(StrategyLoadError):
        parse_strategy_specs({"thinker": ["tp"]})


def test_load_from_file(tmp_path):
    path = tmp_path / "strategy.yaml"
    path.write_text(
        "strategies:\n"
        "  thinker:\n"
        "    - axis: tp\n"
        "      size: 2\n"
        "  talker:\n"
        "    - axis: stage_replica\n"
        "      size: 2\n"
        "      routing: least_queue\n"
    )
    specs = load_strategy_specs(str(path))
    assert specs["thinker"][0].mesh_axis.size == 2
    assert specs["talker"][0].routing.routing_policy == "least_queue"
