# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for the PersonaPlex CUDA-graph wrapper (``graph_capture``).

Verifies the properties the streaming runtime depends on: graphed replay is
numerically identical to eager including in-place streaming-state updates,
argument divergence falls back to eager instead of replaying stale baked-in
values, and a capture failure returns the original callable.
"""

import pytest
import torch
import torch.nn as nn

from tests.helpers.mark import hardware_marks
from vllm_omni.experimental.fullduplex.personaplex.cuda_graphs import (
    _GraphedCallable,
    graph_capture,
)

pytestmark = [
    pytest.mark.core_model,
    *hardware_marks(res={"cuda": "L4"}, num_cards=1),
]
DEVICE = torch.device("cuda:0")
HIDDEN = 32
WARMUP_TICKS = 3


# ---------------------------------------------------------------------------
# Synthetic streaming step
# ---------------------------------------------------------------------------


class SyntheticStreamingStep(nn.Module):
    """Minimal stand-in for the per-frame temporal step.

    The forward both reads and mutates persistent state in place (like a KV
    cache write), the property that makes PersonaPlex capture delicate: a
    phantom warmup forward would corrupt a live session, and a replay must
    apply the state update exactly as eager execution would.
    """

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(HIDDEN, HIDDEN)
        self.register_buffer("state", torch.zeros(HIDDEN))

    def forward(self, x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        self.state.add_(x.mean(dim=0) * scale)
        return self.proj(x) + self.state


def _twin_models() -> tuple[SyntheticStreamingStep, SyntheticStreamingStep]:
    """Two models with identical weights and state (eager reference + capture target)."""
    torch.manual_seed(0)
    ref = SyntheticStreamingStep().to(DEVICE).eval()
    torch.manual_seed(0)
    target = SyntheticStreamingStep().to(DEVICE).eval()
    return ref, target


def _frame(seed: int, batch: int = 1) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(batch, HIDDEN, generator=g).to(DEVICE)


# ---------------------------------------------------------------------------
# 1. Graphed replay is identical to eager, including streaming state
# ---------------------------------------------------------------------------


def test_graphed_matches_eager_with_state():
    ref, target = _twin_models()

    # Warmup ticks run eagerly on both, mirroring the runtime's arming phase.
    with torch.no_grad():
        for i in range(WARMUP_TICKS):
            x = _frame(i)
            ref(x)
            target(x)

    x = _frame(WARMUP_TICKS)
    graphed = graph_capture(target.forward, (x,), label="synthetic")
    assert isinstance(graphed, _GraphedCallable), "capture must succeed, not fall back"

    # Capture only records; state must be untouched by it.
    torch.testing.assert_close(target.state, ref.state, atol=0, rtol=0)

    with torch.no_grad():
        for i in range(WARMUP_TICKS, WARMUP_TICKS + 5):
            x = _frame(i)
            want = ref(x)
            got = graphed(x)
            torch.testing.assert_close(got, want, atol=0, rtol=0)

    torch.testing.assert_close(target.state, ref.state, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# 2. Divergent arguments fall back to eager, never replay stale values
# ---------------------------------------------------------------------------


def test_shape_divergence_falls_back_to_eager():
    ref, target = _twin_models()
    x = _frame(0)
    with torch.no_grad():
        ref(x)
        target(x)
    graphed = graph_capture(target.forward, (x,), label="synthetic")
    assert isinstance(graphed, _GraphedCallable)

    wide = _frame(1, batch=2)
    with torch.no_grad():
        want = ref(wide)
        got = graphed(wide)
    torch.testing.assert_close(got, want, atol=0, rtol=0)

    # A matching-shape call afterwards still replays correctly.
    x2 = _frame(2)
    with torch.no_grad():
        want = ref(x2)
        got = graphed(x2)
    torch.testing.assert_close(got, want, atol=0, rtol=0)


def test_non_tensor_arg_divergence_falls_back_to_eager():
    ref, target = _twin_models()
    x = _frame(0)
    with torch.no_grad():
        ref(x, scale=1.0)
        target(x, scale=1.0)
    graphed = graph_capture(target.forward, (x,), {"scale": 1.0}, label="synthetic")
    assert isinstance(graphed, _GraphedCallable)

    # scale=2.0 was not captured; replaying would bake in scale=1.0.
    x2 = _frame(1)
    with torch.no_grad():
        want = ref(x2, scale=2.0)
        got = graphed(x2, scale=2.0)
    torch.testing.assert_close(got, want, atol=0, rtol=0)
    torch.testing.assert_close(target.state, ref.state, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# 3. Capture failure returns the original callable
# ---------------------------------------------------------------------------


def test_capture_failure_stays_eager():
    def bad_step(x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("not capturable")

    out = graph_capture(bad_step, (torch.ones(1, HIDDEN, device=DEVICE),), label="bad")
    assert out is bad_step
