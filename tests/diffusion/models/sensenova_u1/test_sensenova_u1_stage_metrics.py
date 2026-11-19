# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.sensenova_u1 import pipeline_sensenova_u1 as pipeline_module

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_disabled_profiling_has_no_collective(monkeypatch):
    def unexpected():
        raise AssertionError("Profiling disabled must not look up distributed state")

    monkeypatch.setattr(pipeline_module, "get_tp_group", unexpected)
    pipeline = SimpleNamespace(enable_diffusion_pipeline_profiler=False)
    assert pipeline_module.SenseNovaU1Pipeline._collect_stage_durations(pipeline) == {}


def test_metrics_reduce_complete_phases_with_max(monkeypatch):
    group = object()
    monkeypatch.setattr(pipeline_module, "get_tp_group", lambda: SimpleNamespace(world_size=2, cpu_group=group))

    def all_reduce(values, op, group):
        assert values.device.type == "cpu"
        assert op == torch.distributed.ReduceOp.MAX
        assert group is tp_group
        # Local rank is slower at text preparation, remote rank at generation.
        # Preserve each complete phase, never sum maxima of nested prefix timers.
        values.copy_(torch.maximum(values, torch.tensor([7.0, 1.0], dtype=values.dtype)))

    tp_group = group
    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    pipeline = SimpleNamespace(
        enable_diffusion_pipeline_profiler=True, stage_durations={"generation": 5.0, "text": 2.0}
    )
    assert pipeline_module.SenseNovaU1Pipeline._collect_stage_durations(pipeline) == {"generation": 7.0, "text": 2.0}
    assert pipeline.stage_durations == {"generation": 5.0, "text": 2.0}


def test_single_rank_preserves_stage_values(monkeypatch):
    monkeypatch.setattr(pipeline_module, "get_tp_group", lambda: SimpleNamespace(world_size=1))
    pipeline = SimpleNamespace(
        enable_diffusion_pipeline_profiler=True, stage_durations={"generation": 5.0, "text": 2.0}
    )
    assert pipeline_module.SenseNovaU1Pipeline._collect_stage_durations(pipeline) == pipeline.stage_durations
