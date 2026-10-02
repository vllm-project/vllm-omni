# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Explicit warmup exercises every reachable attention layout."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.attention.strategy import AttentionStrategyRunner, ForwardStrategyPlan
from vllm_omni.diffusion.forward_context import ForwardContext, override_forward_context

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model, pytest.mark.cpu]


def test_two_step_warmup_exercises_late_layout_and_reuses_all_graphs():
    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    model = SimpleNamespace(forward=lambda x, *, attention_layout: x + attention_layout)
    plan = ForwardStrategyPlan(("dense", "mixed", "sparse"), "step_index", ((10, 0), (20, 1), (None, 2)))
    runner = AttentionStrategyRunner(model, plan)
    torch._dynamo.reset()
    try:
        runner.compile(backend=backend, dynamic=True)
        assert runner.warmup_status()["layouts_pending"] == plan.layout_names
        runner.set_warmup(True)
        for step in range(2):
            with override_forward_context(ForwardContext(denoise_step_idx=step, total_denoise_steps=2)):
                torch.testing.assert_close(runner(torch.zeros(5)), torch.zeros(5))
        runner.set_warmup(False)
        assert runner.warmup_status()["layouts_exercised"] == plan.layout_names
        assert not runner.warmup_status()["layouts_pending"]
        assert len(graphs) == 3
        for step, expected in ((0, 0), (10, 1), (20, 2)):
            with override_forward_context(ForwardContext(denoise_step_idx=step, total_denoise_steps=35)):
                torch.testing.assert_close(runner(torch.zeros(5)), torch.full((5,), float(expected)))
        assert len(graphs) == 3
    finally:
        torch._dynamo.reset()


def test_failed_layout_is_not_reported_as_exercised():
    def forward(x, *, attention_layout):
        if attention_layout == 1:
            raise RuntimeError("warmup failed")
        return x

    runner = AttentionStrategyRunner(
        SimpleNamespace(forward=forward), ForwardStrategyPlan(("dense", "sparse"), None, ())
    )
    runner.set_warmup(True)
    with pytest.raises(RuntimeError, match="warmup failed"):
        runner(torch.ones(1))
    runner.set_warmup(False)
    assert runner.warmup_status()["layouts_pending"] == ("sparse",)


def test_eager_warmup_does_not_claim_compiled_readiness():
    runner = AttentionStrategyRunner(
        SimpleNamespace(forward=lambda x, *, attention_layout: x + 1),
        ForwardStrategyPlan(("dense",), None, ()),
    )
    runner.set_warmup(True)
    runner(torch.ones(2))
    runner.set_warmup(False)
    assert not runner.warmup_status()["layouts_pending"]
    runner.compile(backend="eager")
    assert runner.warmup_status()["layouts_pending"] == ("dense",)


def test_worker_reports_unexercised_component():
    from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker

    def make_runner():
        return AttentionStrategyRunner(
            SimpleNamespace(forward=lambda x, *, attention_layout: x + attention_layout),
            ForwardStrategyPlan(("dense", "sparse"), None, ()),
        )

    first, second = make_runner(), make_runner()
    worker = object.__new__(DiffusionWorker)
    worker.model_runner = SimpleNamespace(
        pipeline=SimpleNamespace(
            attention_strategy_components=("first", "second"),
            first=SimpleNamespace(_attention_strategy_runner=first),
            second=SimpleNamespace(_attention_strategy_runner=second),
        )
    )
    worker.attention_strategy_warmup(True)
    first(torch.zeros(2))
    status = worker.attention_strategy_status()
    assert first.warming and second.warming  # Query does not mutate warmup state.
    assert status["first"]["layouts_pending"] == ()
    assert status["second"]["layouts_pending"] == ("dense", "sparse")
    worker.attention_strategy_warmup(False)
    assert not first.warming and not second.warming
