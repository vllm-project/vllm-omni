# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real Dynamo guard budgets and failure behavior across layouts."""

from contextlib import contextmanager

import pytest
import torch

from vllm_omni.diffusion.attention.strategy import AttentionStrategyRunner, ForwardStrategyPlan
from vllm_omni.diffusion.forward_context import ForwardContext, override_forward_context

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def reset_compiler():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def make_runner(layouts):
    executions = []
    eager_calls = []
    graphs = []

    class Model(torch.nn.Module):
        def forward_with_attention_layout(self, x, *, attention_layout, tag="0"):
            if not torch.compiler.is_compiling():
                eager_calls.append("entry")
            x = x + attention_layout
            return x + int(tag)

    def backend(graph, inputs):
        graphs.append(graph)

        def execute(*args):
            executions.append(graph)
            return graph.forward(*args)

        return execute

    plan = ForwardStrategyPlan(
        tuple(str(i) for i in range(layouts)), "step_index", tuple((i + 1, i) for i in range(layouts))
    )
    runner = AttentionStrategyRunner(Model(), plan)
    return runner, backend, graphs, executions, eager_calls


@contextmanager
def layout(index, total):
    with override_forward_context(ForwardContext(denoise_step_idx=index, total_denoise_steps=total)):
        yield


@pytest.mark.parametrize("layouts,dtypes", [(9, [torch.float32]), (3, [torch.float32, torch.float64, torch.bfloat16])])
def test_layouts_have_independent_guard_budgets(layouts, dtypes):
    runner, backend, graphs, executions, eager_calls = make_runner(layouts)
    runner.compile(backend=backend, dynamic=True)
    for replay in range(2):
        for dtype in dtypes:
            for index in range(layouts):
                for length in (4, 7):
                    x = torch.ones(length, dtype=dtype)
                    before = len(executions)
                    with layout(index, layouts):
                        torch.testing.assert_close(runner(x), x + index)
                    assert len(executions) > before
        if replay == 0:
            warmed = len(graphs)
        else:
            assert len(graphs) == warmed
    assert len(graphs) >= layouts * len(dtypes)
    assert eager_calls == []


def test_budget_exhaustion_uses_pytorch_fallback():
    runner, backend, graphs, executions, eager_calls = make_runner(1)
    runner.compile(backend=backend, recompile_limit=2)
    with torch._dynamo.config.patch(fail_on_recompile_limit_hit=False):
        with layout(0, 1):
            for tag in ("0", "1", "2"):
                x = torch.ones(4)
                torch.testing.assert_close(runner(x, tag=tag), x + int(tag))
    assert eager_calls
