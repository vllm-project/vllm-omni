# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Stateful contract example, independent of an optional attention kernel.

The prepared module owns tensor state and passes it explicitly to an opaque op.
This deliberately does not model opaque Python/C++ planning handles.
"""

import gc
import weakref

import pytest
import torch
import torch.nn.functional as F

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@torch.library.custom_op("vllm_omni_test::planned_attention", mutates_args=())
def _planned_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_scale: torch.Tensor,
) -> torch.Tensor:
    return F.scaled_dot_product_attention(query * query_scale, key, value).contiguous()


@_planned_attention.register_fake
def _planned_attention_fake(query, key, value, query_scale):
    return query.new_empty((*query.shape[:-1], value.shape[-1]))


class _PreparedAttention(torch.nn.Module):
    """Initialization/planning runs eagerly; forward only executes tensors."""

    def __init__(self, query_scale: torch.Tensor):
        super().__init__()
        self.register_buffer("query_scale", query_scale.detach().clone())

    def forward(self, query, key, value):
        return _planned_attention(query, key, value, self.query_scale)


def _inputs(length):
    generator = torch.Generator().manual_seed(length)
    return tuple(torch.randn(2, 3, length, 8, generator=generator) for _ in range(3))


def _reference(plan, inputs):
    query, key, value = inputs
    return F.scaled_dot_product_attention(query * plan.query_scale, key, value)


def test_dynamic_replay_uses_each_instances_state_without_recompilation():
    compile_count = 0

    def counting_backend(graph_module, _example_inputs):
        nonlocal compile_count
        compile_count += 1
        return graph_module.forward

    # Same tensor metadata, different values: state must be a graph input,
    # never a Python instance identifier or a captured first-instance constant.
    plans = [_PreparedAttention(torch.full((1, 3, 1, 1), scale)) for scale in (0.5, 2.0)]
    compiled = torch.compile(
        lambda plan, q, k, v: plan(q, k, v),
        backend=counting_backend,
        fullgraph=True,
        dynamic=True,
    )
    for length in (5, 9, 13):
        inputs = _inputs(length)
        for plan in plans:
            torch.testing.assert_close(compiled(plan, *inputs), _reference(plan, inputs))
        assert not torch.allclose(_reference(plans[0], inputs), _reference(plans[1], inputs))
    assert compile_count == 1


def test_compiled_callable_retains_state_until_released():
    plan = _PreparedAttention(torch.full((1, 3, 1, 1), 0.5))
    plan_ref = weakref.ref(plan)
    state_ref = weakref.ref(plan.query_scale)
    inputs = _inputs(5)
    expected = _reference(plan, inputs)
    compiled = torch.compile(plan, backend="eager", fullgraph=True, dynamic=True)
    torch.testing.assert_close(compiled(*inputs), expected)

    # Deleting the caller's reference must not invalidate subsequent replay.
    del plan
    gc.collect()
    assert plan_ref() is not None
    assert state_ref() is not None
    torch.testing.assert_close(compiled(*inputs), expected)

    del compiled
    gc.collect()
    assert plan_ref() is None
    assert state_ref() is None


def test_stateful_custom_op_schema_and_fake_implementation():
    inputs = _inputs(5)
    # Unequal Q/K lengths and V head dimensions exercise the fake output shape.
    query, _, _ = inputs
    key = torch.randn(2, 3, 7, 8)
    value = torch.randn(2, 3, 7, 4)
    scale = torch.full((1, 3, 1, 1), 0.5)
    torch.library.opcheck(_planned_attention, (query, key, value, scale))


def test_stateful_attention_compiles_with_inductor():
    plan = _PreparedAttention(torch.full((1, 3, 1, 1), 0.5))
    compiled = torch.compile(plan, fullgraph=True, dynamic=True)
    for length in (5, 9):
        inputs = _inputs(length)
        torch.testing.assert_close(compiled(*inputs), _reference(plan, inputs))
