# SPDX-License-Identifier: Apache-2.0
"""Evaluation-based risk routing and budget-only guided MSE feedback."""

import math

from .state import Decision


def phase_for(index, total, config):
    progress = index / max(total - 1, 1)
    return "early" if progress < config.phase.early_end else "mid" if progress < config.phase.late_start else "late"


def mean_abs(x):
    return float(x.float().abs().mean())


def relative_mse(prediction, reference):
    return float(
        (prediction.float() - reference.float()).square().sum() / reference.float().square().sum().clamp_min(1e-12)
    )


def guided(cond, uncond, scale):
    return uncond.float() + (cond.float() - uncond.float()) * scale


def decide(state, x, index, total, config):
    phase = phase_for(index, total, config)
    age = index + 1 if state.anchor is None else index - state.anchor.index
    threshold = config.base_threshold * config.phase.scales[phase] * state.feedback_scale[phase]
    budget = None
    reason = None
    if config.mode == "force_exact":
        reason = "force_exact"
    elif index < config.warmup_steps or index >= total - config.tail_steps:
        reason = "warmup_tail"
    elif threshold == 0:
        reason = "threshold_disabled"
    elif state.anchor is None or state.previous_input is None or state.gain is None:
        reason = "bootstrap"
    else:
        norm = mean_abs(state.anchor.cond)
        if norm <= config.feedback.eps:
            reason = "zero_norm"
        else:
            budget = state.budget + state.gain * mean_abs(x.float() - state.previous_input) / norm
            if not all(math.isfinite(v) for v in (budget, threshold, state.gain)):
                reason = "nonfinite_risk"
            elif age > config.feedback.age_caps[phase]:
                reason = "max_age"
            else:
                reason = "budget_skip" if budget < threshold else "budget_exact"
    return Decision(phase, reason, reason != "budget_skip", age, budget, threshold, state.gain)


def update_feedback(state, decision, drafts, outputs, guidance, config):
    if decision.reason != "budget_exact" or any(v is None for v in drafts):
        return
    error = relative_mse(guided(*drafts, guidance), guided(*outputs, guidance))
    if not math.isfinite(error):
        return
    fb = config.feedback
    ratio = max(fb.ratio_low, min(fb.ratio_high, error / max(decision.budget, fb.eps)))
    p = decision.phase
    state.ratio_ema[p] = fb.ratio_ema * state.ratio_ema[p] + (1 - fb.ratio_ema) * ratio
    state.feedback_scale[p] = max(fb.scale_low[p], min(fb.scale_high[p], 1 / max(state.ratio_ema[p], fb.eps)))
    state.feedback_updates += 1


def update_gain(state, x, cond, config):
    if state.anchor is None:
        return
    delta = mean_abs(x.float() - state.anchor.input)
    gain = mean_abs(cond.float() - state.anchor.cond) / max(delta, config.feedback.eps)
    state.gain = gain if delta > config.feedback.eps and math.isfinite(gain) else None
