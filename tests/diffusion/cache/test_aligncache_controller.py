# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace

import pytest
import torch

from vllm_omni.diffusion.cache.aligncache.config import AlignCacheConfig
from vllm_omni.diffusion.cache.aligncache.controller import decide, phase_for, update_feedback, update_gain
from vllm_omni.diffusion.cache.aligncache.memory import TensorLedger
from vllm_omni.diffusion.cache.aligncache.predictor import correct_anchor, fit_scale, predict, write_branch
from vllm_omni.diffusion.cache.aligncache.state import Decision, ExactAnchor, RequestState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def state_with_anchor():
    c = AlignCacheConfig.parse({"align_cache": {"warmup_steps": 0, "tail_steps": 0}})
    state = RequestState.create(1, c)
    x = torch.ones(128)
    state.anchor = ExactAnchor(3, x.clone(), x.clone(), x.clone(), x.clone(), x.clone(), x.dtype)
    state.previous_input = x.clone()
    state.gain = 1.0
    return c, state, x


def test_phase_boundaries_and_short():
    c = AlignCacheConfig()
    assert [phase_for(i, 101, c) for i in (14, 15, 85, 86)] == ["early", "mid", "mid", "late"]
    assert phase_for(0, 1, c) == "early"


def test_age_and_budget_equal_boundaries():
    c, s, x = state_with_anchor()
    assert decide(s, x, 7, 30, c).reason == "budget_skip"  # age == 4
    assert decide(s, x, 8, 30, c).reason == "max_age"
    s.budget = c.base_threshold
    assert decide(s, x, 6, 30, c).reason == "budget_exact"
    assert s.budget == c.base_threshold  # decision is pure
    zero = replace(c, base_threshold=0)
    assert decide(s, x, 6, 30, zero).reason == "threshold_disabled"


def test_gain_uses_exact_input_risk_uses_adjacent_input():
    c, s, x = state_with_anchor()
    s.previous_input = x + 0.9  # previous fast evaluation, not exact
    update_gain(s, x + 1, x + 2, c)
    assert s.gain == pytest.approx(2)
    d = decide(s, x + 1, 6, 30, c)
    assert d.budget == pytest.approx(0.2)


def test_zero_and_nonfinite_guards():
    c, s, x = state_with_anchor()
    s.gain = float("nan")
    assert decide(s, x, 6, 30, c).reason == "nonfinite_risk"
    s.gain = 1
    s.anchor.cond.zero_()
    assert decide(s, x, 6, 30, c).reason == "zero_norm"
    update_gain(s, s.anchor.input, x, c)
    assert s.gain is None


def test_feedback_only_budget_and_guided_mse():
    c, s, x = state_with_anchor()
    for reason in ("warmup_tail", "bootstrap", "max_age", "force_exact", "capacity_exceeded"):
        d = Decision("mid", reason, True, 1, 0.2, 0.1, 1)
        update_feedback(s, d, (x * 2, x), (x, x), 5, c)
    assert s.feedback_updates == 0
    d = Decision("mid", "budget_exact", True, 1, 0.2, 0.1, 1)
    update_feedback(s, d, (x * 2, x), (x, x), 5, c)
    assert s.ratio_ema["mid"] == pytest.approx(1.025)
    assert s.feedback_scale["mid"] == pytest.approx(1 / 1.025)


def test_scale_clamp_order_and_exact_only_learning():
    c, s, x = state_with_anchor()
    assert fit_scale(x * 10, x, 1) == 1.5
    write_branch(s, x, x * 2, "cond", "mid", True, c)
    write_branch(s, x, x * 11, "cond", "mid", True, c)
    assert s.cond.scales["mid"] == pytest.approx(0.85 + 0.15 * 1.18)
    before = s.cond.scales.copy()
    write_branch(s, x, x * 20, "cond", "mid", False, c)
    assert s.cond.scales == before and s.cond.source == "predicted"
    assert s.cond.previous_source == "exact"


def test_no_history_uses_true_residual_and_anchor_gates():
    c, s, x = state_with_anchor()
    assert torch.equal(predict(s, x, "cond", "mid", c), x * 2)
    for phase, index, cond, uncond, reason in (
        ("early", 4, x, x, "early_exact"),
        ("mid", 5, x, x, "max_age"),
        ("mid", 4, x * 2, x, "cond_gate"),
        ("mid", 4, x, x * 2, "disagreement"),
        ("mid", 4, x, x, "applied"),
        ("late", 5, x, x, "weak_gate"),
    ):
        result, actual = correct_anchor(s, cond, uncond, index, phase, c)
        assert actual == reason
        if reason != "applied":
            assert result is uncond


def test_temporary_storage_accounting_and_release():
    x = torch.ones(128)
    ledger = TensorLedger()
    with ledger:
        y = x + 1
        z = y[:64]
        assert ledger.active_bytes == 512
        tmp = y * 2
        assert ledger.active_bytes == 1024
        del tmp
        assert ledger.active_bytes == 512
        with ledger.native():
            workspace = torch.ones(4096)
        assert ledger.active_bytes == 512
    del y, z, workspace
    assert ledger.active_bytes == 0 and ledger.peak == 1024


def test_native_callbacks_run_outside_the_dispatch_mode():
    from torch.utils._python_dispatch import _get_current_dispatch_mode

    ledger = TensorLedger()
    with ledger:
        assert _get_current_dispatch_mode() is ledger
        with ledger.native():
            # Native model kernels must not be routed through the ledger's Python dispatch.
            assert _get_current_dispatch_mode() is None
            workspace = torch.ones(8) + 1
        assert _get_current_dispatch_mode() is ledger
        assert ledger.active_bytes == 0
    assert _get_current_dispatch_mode() is None and workspace.sum() == 16


@pytest.mark.parametrize("observer_on_top", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_native_preserves_other_modes_and_restores_after_nested_exit(observer_on_top, fail):
    from contextlib import ExitStack

    from torch.utils._python_dispatch import TorchDispatchMode, _get_current_dispatch_mode

    class Observer(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            self.calls += 1
            return func(*args, **(kwargs or {}))

    ledger, observer = TensorLedger(), Observer()
    modes = (ledger, observer) if observer_on_top else (observer, ledger)
    with ExitStack() as stack:
        for mode in modes:
            stack.enter_context(mode)
        try:
            with ledger.native():
                assert _get_current_dispatch_mode() is observer
                with ledger.native():
                    workspace = torch.ones(8)
                assert ledger.suspended
                workspace = workspace + 1
                assert observer.calls == 2
                assert ledger.active_bytes == 0
                if fail:
                    raise RuntimeError("native failure")
        except RuntimeError as error:
            assert fail and str(error) == "native failure"
        assert not ledger.suspended
        assert _get_current_dispatch_mode() is modes[-1]
        tracked = torch.ones(8)
        assert observer.calls == 3
        assert ledger.active_bytes == tracked.untyped_storage().nbytes()
    assert _get_current_dispatch_mode() is None


def test_stride64_gates_do_not_subsample_risk_or_fit():
    c, state, x = state_with_anchor()
    residual = torch.arange(128, dtype=torch.float32) + 1
    write_branch(state, x, x + residual, "cond", "mid", True, c)
    assert torch.equal(state.cond.sampled, residual[::64])
    changed = residual.clone()
    changed[1::2] *= 2
    # The fit uses all elements, including those absent from gate sampling.
    expected = float((changed * residual).sum() / residual.square().sum())
    assert fit_scale(changed, residual, 1) == pytest.approx(min(expected, 1.5))
    state.previous_input = x.clone()
    state.gain = 1
    moved = x.clone()
    moved[1] += 0.1
    assert decide(state, moved, 6, 30, c).budget > 0


def test_bf16_prediction_keeps_intermediate_fp32_until_output_boundary():
    c, state, x = state_with_anchor()
    state.anchor = replace(state.anchor, output_dtype=torch.bfloat16)
    state.cond.last = torch.full_like(x, 0.003)
    state.guided.sampled = x[:2]
    raw = predict(state, x.to(torch.bfloat16), "cond", "mid", c)
    assert raw.dtype == torch.float32
    assert not torch.equal(raw, raw.to(torch.bfloat16).float())
