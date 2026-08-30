# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Read-only pipeline topology introspection (gap 2 of #4560)."""

from __future__ import annotations

import pytest

from vllm_omni.config.stage_config import PipelineConfig, StageExecutionType, StagePipelineConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _stage(stage_id: int, sources: tuple[int, ...] = (), *, final: bool = False) -> StagePipelineConfig:
    return StagePipelineConfig(
        stage_id=stage_id,
        model_stage=f"s{stage_id}",
        execution_type=StageExecutionType.LLM_AR,
        input_sources=sources,
        final_output=final,
    )


def test_topology_identifies_entry_and_terminal_stages() -> None:
    pipeline = PipelineConfig(model_type="linear", stages=(_stage(0), _stage(1, (0,), final=True)))

    topology = pipeline.topology()

    assert topology.entry_stages == (0,)
    assert topology.terminal_stages == (1,)


def test_topology_reports_declared_inputs() -> None:
    pipeline = PipelineConfig(model_type="diamond", stages=(_stage(0), _stage(1, (0,)), _stage(2, (0,))))

    topology = pipeline.topology()

    assert topology.inputs == {0: (), 1: (0,), 2: (0,)}


def test_topology_derives_downstream_consumers() -> None:
    pipeline = PipelineConfig(model_type="diamond", stages=(_stage(0), _stage(1, (0,)), _stage(2, (0,))))

    topology = pipeline.topology()

    assert topology.consumers == {0: (1, 2), 1: (), 2: ()}


def test_topology_handles_multiple_entries_and_terminals() -> None:
    pipeline = PipelineConfig(
        model_type="multi",
        stages=(
            _stage(0, final=True),
            _stage(1, (0,)),
            _stage(2, (1,), final=True),
            _stage(3, (0, 2), final=True),
        ),
    )

    topology = pipeline.topology()

    assert topology.entry_stages == (0,)
    assert topology.terminal_stages == (0, 2, 3)
    assert topology.consumers[0] == (1, 3)
    assert topology.consumers[1] == (2,)
    assert topology.consumers[2] == (3,)


def test_topology_is_cycle_agnostic() -> None:
    """Cyclic topologies (e.g. MiMo Audio, DiTAR) must still introspect cleanly."""
    pipeline = PipelineConfig(model_type="cyclic", stages=(_stage(0, (1,)), _stage(1, (0,), final=True)))

    topology = pipeline.topology()

    assert topology.entry_stages == ()
    assert topology.terminal_stages == (1,)
    assert topology.consumers == {0: (1,), 1: (0,)}


def test_topology_empty_pipeline() -> None:
    topology = PipelineConfig(model_type="empty").topology()

    assert topology.entry_stages == ()
    assert topology.terminal_stages == ()
    assert topology.inputs == {}
    assert topology.consumers == {}
