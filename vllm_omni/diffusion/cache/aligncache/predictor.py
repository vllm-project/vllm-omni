# SPDX-License-Identifier: Apache-2.0
"""Independent implementation of the documented Wan residual predictor equations."""

import math

import torch

from .controller import guided, relative_mse


def ema(previous, current, decay):
    if current is None:
        return previous
    return current if previous is None else decay * previous + (1 - decay) * current


def fit_scale(target, basis, default):
    denominator = float(basis.float().square().sum())
    value = float((target.float() * basis.float()).sum()) / denominator if denominator > 1e-12 else default
    return max(-1.5, min(1.5, value))


def update_quality(history, sampled, config):
    if history.sampled is not None:
        previous = history.sampled
        old_norm, new_norm = torch.linalg.vector_norm(previous), torch.linalg.vector_norm(sampled)
        ratio = float(new_norm / old_norm.clamp_min(1e-12))
        denominator = float(old_norm * new_norm)
        direction = float((torch.dot(sampled, previous) / denominator).clamp(-1, 1)) if denominator > 1e-12 else None
        history.ratio = ema(history.ratio, ratio, config.ratio_ema)
        history.direction = ema(history.direction, direction, config.direction_ema)
    history.sampled = sampled.clone()


def write_branch(state, x, output, branch, phase, exact, config):
    history = getattr(state, branch)
    residual = output.detach().float() - x.detach().float()
    if exact and config.predictor.online_scale and history.last is not None:
        fitted = fit_scale(residual, history.last, history.scales[phase])
        fitted = max(config.predictor.scale_low, min(config.predictor.scale_high, fitted))
        history.scales[phase] = ema(history.scales[phase], fitted, config.predictor.scale_ema)
    update_quality(history, residual.reshape(-1)[:: config.predictor.stride], config.predictor)
    history.previous, history.last = history.last, residual
    history.previous_source, history.source = history.source, "exact" if exact else "predicted"
    if branch == "uncond":
        sampled = guided(state.cond.sampled, state.uncond.sampled, state.guidance)
        update_quality(state.guided, sampled, config.predictor)


def gate(history, tolerance, minimum):
    ratio = 0 if history.ratio is None else max(0, 1 - abs(history.ratio - 1) / max(tolerance, 1e-6))
    direction = (
        0 if history.direction is None else max(0, min(1, (history.direction - minimum) / max(1 - minimum, 1e-6)))
    )
    return min(ratio, direction)


def predict(state, x, branch, phase, config):
    anchor = state.anchor
    if anchor is None:
        return None
    history = getattr(state, branch)
    if not config.predictor.enabled or state.guided.sampled is None or history.last is None:
        residual = getattr(anchor, f"{branch}_residual")
    else:
        p = config.predictor
        scale = max(p.scale_low, min(p.scale_high, history.scales[phase])) if p.online_scale else 1.0
        residual = history.last * scale
        if p.second_order and history.previous is not None:
            strength = min(
                gate(history, p.ratio_tolerance[phase], p.direction_min[phase]),
                gate(state.guided, p.ratio_tolerance[phase], p.direction_min[phase]),
            )
            beta = p.beta[phase] * strength
            if beta > 1e-4 and strength > 1e-4:
                residual = residual + (history.last - history.previous) * beta
    return x.float() + residual


def correct_anchor(state, cond, uncond, index, phase, config):
    a = config.anchor
    reason = "disabled"
    if not config.predictor.enabled or not a.enabled:
        return uncond, reason
    anchor = state.anchor
    if anchor is None or cond is None:
        return uncond, "no_anchor"
    if a.early_exact and phase == "early":
        return uncond, "early_exact"
    age = index - anchor.index
    if age > a.age_caps[phase]:
        return uncond, "max_age"
    change = relative_mse(cond, anchor.cond)
    if not math.isfinite(change) or change > a.cond_cap[phase]:
        return uncond, "cond_gate"
    corrected = anchor.uncond + (cond.float() - anchor.cond) * state.alpha[phase]
    disagreement = relative_mse(corrected, uncond)
    if not math.isfinite(disagreement) or disagreement > a.disagreement_cap[phase]:
        return uncond, "disagreement"
    cond_gate = 1 if a.cond_cap[phase] <= 0 else max(0, 1 - change / max(a.cond_cap[phase], 1e-6))
    diff_gate = 1 if a.disagreement_cap[phase] <= 0 else max(0, 1 - disagreement / max(a.disagreement_cap[phase], 1e-6))
    age_gate = 1 if a.age_caps[phase] <= 1 else max(0, 1 - max(age - 1, 0) / max(a.age_caps[phase] - 1, 1))
    weight = a.blend[phase] * min(cond_gate, diff_gate, age_gate)
    if weight <= 1e-4:
        return uncond, "weak_gate"
    return uncond.float() + (corrected - uncond.float()) * weight, "applied"


def update_alpha(state, cond, uncond, phase, config):
    if state.anchor is not None and config.predictor.enabled and config.anchor.enabled:
        fitted = fit_scale(uncond.float() - state.anchor.uncond, cond.float() - state.anchor.cond, state.alpha[phase])
        state.alpha[phase] = ema(state.alpha[phase], fitted, config.anchor.alpha_ema)
