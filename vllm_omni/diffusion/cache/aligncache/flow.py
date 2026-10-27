# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exact-anchor velocity controllers for Qwen 2.1 and joint H3 AV.

Qwen feedback uses the CFG-combined relative MSE; H3 shares one forward decision
across video/audio with per-stream relative-L2 feedback. Predicted values are
never stored as exact anchors.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace

import torch

from .config import AlignCacheConfig
from .controller import mean_abs, relative_mse
from .memory import TensorLedger

logger = logging.getLogger(__name__)

TensorContract = tuple[tuple[tuple[int, ...], torch.dtype, torch.device], ...]


def sample_velocity(x: torch.Tensor, qwen: bool) -> torch.Tensor:
    values = x.detach().reshape(-1)
    if not qwen:
        return values[::64].float().clone()
    count = min(values.numel(), 4096)
    if count == values.numel():
        return values.float().clone()
    index = torch.arange(count, device=x.device, dtype=torch.int64)
    start = index * values.numel() // count
    width = (index + 1) * values.numel() // count - start
    h = ((index + 1) * 0x45D9F3B) & 0xFFFFFFFF
    h = ((h ^ (h >> 16)) * 0x45D9F3B) & 0xFFFFFFFF
    return values[start + (h ^ (h >> 16)) % width].float().clone()


@dataclass(frozen=True)
class VelocityAnchor:
    """An owned exact output and its matching input sample/evaluation."""

    input_sample: torch.Tensor
    output: torch.Tensor
    sigma: float
    index: int


@dataclass
class VelocityStream:
    anchors: tuple[VelocityAnchor, ...] = ()
    previous_input: torch.Tensor | None = None
    budget: float = 0.0
    scales: list[float] = field(default_factory=lambda: [1.0] * 3)
    feedback: list[float] = field(default_factory=lambda: [1.0] * 3)
    ratio: float | None = None
    direction: float | None = None

    def fork(self) -> VelocityStream:
        return replace(self, scales=self.scales.copy(), feedback=self.feedback.copy())


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def fit(target: torch.Tensor, basis: torch.Tensor, default: float, low: float, high: float) -> float:
    denom = float(basis.square().sum())
    if not math.isfinite(denom) or denom <= 1e-8:
        return default
    fitted = float((target * basis).sum()) / denom
    return clamp(fitted, low, high) if math.isfinite(fitted) else default


class FlowRuntime:
    def __init__(self, config: AlignCacheConfig):
        if config.policy not in ("qwen_flow_v1", "minimax_h3_flow_v1"):
            raise ValueError("align_cache: FlowRuntime requires a flow policy")
        self.config, self.mode = config, config.mode
        self.qwen = config.policy == "qwen_flow_v1"
        self.active = False
        self.last_summary: dict[str, object] = {}
        self.reset()

    def reset(self) -> None:
        if self.active:
            raise RuntimeError("align_cache: request already active")
        self.streams: list[VelocityStream] = []
        self.schedule: tuple[tuple[float, ...], ...] | None = None
        self.identity: tuple[object, ...] | None = None
        self.input_contract: TensorContract | None = None
        self.output_contract: TensorContract | None = None
        self.index = 0
        self.transfer = [1.0] * 3
        self.calls = 0
        self.counts: Counter[str] = Counter()
        self.peak_bytes = 0
        self.disabled_reason: str | None = None

    @contextmanager
    def request_context(self) -> Iterator[FlowRuntime]:
        self.reset()
        self.active = True
        status = "failed"
        try:
            yield self
            status = "completed"
        finally:
            self.last_summary = {
                "backend": "align_cache",
                "policy": self.config.policy,
                "mode": self.mode,
                "status": status,
                "completed_evaluations": self.index,
                "model_forward_calls": self.calls,
                "reasons": dict(self.counts),
                "peak_cache_bytes": self.peak_bytes,
            }
            self.streams = []
            self.schedule = self.identity = self.input_contract = None
            self.output_contract = None
            self.active = False
            if self.config.diagnostics != "off":
                try:
                    logger.info("AlignCache flow request: %s", self.last_summary)
                except Exception:
                    pass

    def bind(self, schedules: Sequence[Sequence[float]], identity: tuple) -> None:
        if not self.active or self.schedule is not None:
            raise RuntimeError("align_cache: schedule requires a fresh active request")
        expected_counts = (1, 2) if self.qwen else (2,)
        if len(schedules) not in expected_counts:
            raise ValueError(f"align_cache: policy requires stream count in {expected_counts}")
        plans = tuple(tuple(float(s) for s in values) for values in schedules)
        if not plans or len({len(p) for p in plans}) != 1 or len(plans[0]) < 2:
            raise ValueError("align_cache: aligned sigma schedules with terminal sigma required")
        for p in plans:
            if any(not math.isfinite(s) or s < 0 for s in p) or any(a <= b for a, b in zip(p, p[1:])):
                raise ValueError("align_cache: finite descending nonnegative sigmas required")
        self.schedule, self.identity = plans, identity
        self.streams = [VelocityStream() for _ in plans]

    def _input(self, x: torch.Tensor) -> torch.Tensor:
        return sample_velocity(x, True) if self.qwen else x.detach().float().clone()

    def _prediction(self, state: VelocityStream, phase: int) -> torch.Tensor:
        b, a = state.anchors
        current, previous = a.output.float(), b.output.float()
        sa, sb = sample_velocity(current, self.qwen), sample_velocity(previous, self.qwen)
        if self.qwen:
            na, nb = sa.norm(), sb.norm()
            gate = float(
                (torch.dot(sa, sb) / (na * nb).clamp_min(1e-12)).clamp(0, 1)
                * torch.minimum(na, nb)
                / torch.maximum(na, nb).clamp_min(1e-12)
            )
        else:
            tolerance = (0.0, 0.30, 0.18)[phase]
            minimum = (1.0, 0.80, 0.88)[phase]
            ratio = 0 if state.ratio is None else max(0, 1 - abs(state.ratio - 1) / max(tolerance, 1e-6))
            direction = (
                0 if state.direction is None else clamp((state.direction - minimum) / max(1 - minimum, 1e-6), 0, 1)
            )
            gate = min(ratio, direction)
            if gate <= 1e-4:
                gate = 0.0
        beta = self.config.predictor.beta[("early", "mid", "late")[phase]]
        return (state.scales[phase] * current + beta * gate * (current - previous)).to(a.output.dtype)

    def evaluate(
        self,
        index: int,
        inputs: Sequence[torch.Tensor],
        identity: tuple,
        exact: Callable[[Callable[[], None]], Sequence[torch.Tensor]],
        combine: Callable[[Sequence[torch.Tensor]], torch.Tensor] | None = None,
        force_exact: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        if not self.active or self.schedule is None or identity != self.identity or index != self.index:
            raise ValueError("align_cache: missing plan, changed identity or out-of-order evaluation")
        total = len(self.schedule[0]) - 1
        if index >= total or len(inputs) != len(self.streams):
            raise ValueError("align_cache: evaluation count/stream mismatch")
        contract = tuple((tuple(x.shape), x.dtype, x.device) for x in inputs)
        if any(x.numel() == 0 for x in inputs):
            raise ValueError("align_cache: empty velocity stream is unsupported")
        if self.input_contract is not None and self.input_contract != contract:
            raise ValueError("align_cache: input shape/dtype/device changed")
        self.input_contract = contract
        for x, state in zip(inputs, self.streams):
            if state.anchors and (x.shape != state.anchors[-1].input_sample.shape and not self.qwen):
                raise ValueError("align_cache: input shape changed")
            if state.anchors and (
                x.shape != state.anchors[-1].output.shape or x.device != state.anchors[-1].output.device
            ):
                raise ValueError("align_cache: velocity shape/device changed")
        ledger = TensorLedger(self.streams)
        try:
            with ledger:
                return self._evaluate(index, total, inputs, exact, combine, force_exact, ledger)
        finally:
            self.peak_bytes = max(self.peak_bytes, ledger.peak)

    def _valid_predictions(self, outputs: Sequence[torch.Tensor], inputs: Sequence[torch.Tensor]) -> bool:
        """Validate both branches before any native model side effects."""
        return len(outputs) == len(inputs) and all(
            isinstance(output, torch.Tensor)
            and output.shape == x.shape
            and output.device == x.device
            and output.dtype == state.anchors[-1].output.dtype
            and bool(torch.isfinite(output).all())
            for output, x, state in zip(outputs, inputs, self.streams)
        )

    def _evaluate(self, index, total, inputs, exact, combine, force_exact, ledger):
        assert self.schedule is not None
        ratio = index / max(total - 1, 1)
        phase = 0 if ratio < self.config.phase.early_end else 1 if ratio < self.config.phase.late_start else 2
        name = ("early", "mid", "late")[phase]
        budgets = [s.budget for s in self.streams]
        thresholds = [
            self.config.base_threshold if j == 0 or self.qwen else self.config.flow.audio_threshold
            for j in range(len(inputs))
        ]
        thresholds = [
            t
            * self.config.phase.scales[name]
            * clamp(1 / s.feedback[phase], self.config.feedback.scale_low[name], self.config.feedback.scale_high[name])
            for t, s in zip(thresholds, self.streams)
        ]
        if sum(x.numel() for x in inputs) * 4 * 50 > self.config.max_cache_bytes:
            self.disabled_reason = self.disabled_reason or "capacity_exceeded"
            self.streams = [VelocityStream() for _ in inputs]
        reason = None
        if self.disabled_reason is not None:
            reason = self.disabled_reason
        elif self.mode in ("force_exact", "passthrough") or force_exact:
            reason = "force_exact"
        elif any(t <= 0 for t in thresholds) or (self.qwen and self.config.flow.max_consecutive_skips == 0):
            reason = "threshold_disabled"
        elif index < self.config.warmup_steps or index >= total - self.config.tail_steps:
            reason = "warmup_tail"
        elif any(len(s.anchors) < 2 for s in self.streams):
            reason = "bootstrap"
        else:
            age = index - self.streams[0].anchors[-1].index
            if (
                (age - 1 >= self.config.feedback.age_caps[name])
                if self.qwen
                else (age > self.config.feedback.age_caps[name])
            ):
                reason = "max_age"
            if self.qwen and age - 1 >= self.config.flow.max_consecutive_skips:
                reason = reason or "max_age"
            for j, (x, state) in enumerate(zip(inputs, self.streams)):
                if self.qwen and j > 0:
                    continue  # Qwen routes on the conditional branch risk.
                b, a = state.anchors
                ya, yb = (
                    (sample_velocity(a.output, True), sample_velocity(b.output, True))
                    if self.qwen
                    else (a.output, b.output)
                )
                dx, norm = mean_abs(a.input_sample - b.input_sample), mean_abs(ya)
                if dx <= 1e-8 or norm <= 1e-8:
                    reason = reason or "unreliable_gain"
                elif self.qwen and abs(self.schedule[j][index] - a.sigma) > self.config.flow.max_time_gap:
                    reason = reason or "time_gap"
                else:
                    budgets[j] += (
                        mean_abs(ya - yb)
                        / max(dx, 1e-12 if not self.qwen else 1e-8)
                        * mean_abs(self._input(x) - state.previous_input)
                        / max(norm, 1e-12 if not self.qwen else 1e-8)
                    )
                    if not math.isfinite(budgets[j]):
                        reason = reason or "nonfinite_risk"
            if reason is None and any(b >= t for b, t in zip(budgets[:1] if self.qwen else budgets, thresholds)):
                reason = "budget"
        drafts = None
        if reason in (None, "budget"):
            drafts = tuple(self._prediction(s, phase) for s in self.streams)
            if not self._valid_predictions(drafts, inputs):
                reason, drafts = "invalid_prediction", None
            if drafts is not None and self.qwen and len(drafts) == 2:
                c, u = drafts
                ca, ua = (s.anchors[-1].output.float() for s in self.streams)
                candidate = ua + self.transfer[phase] * (c.float() - ca)
                age = index - self.streams[0].anchors[-1].index - 1
                weight = self.config.anchor.blend[name] * min(
                    max(0, 1 - age / max(self.config.feedback.age_caps[name], 1)),
                    max(0, 1 - relative_mse(c, ca) / max(self.config.anchor.cond_cap[name], 1e-12)),
                    max(0, 1 - relative_mse(candidate, u) / max(self.config.anchor.disagreement_cap[name], 1e-12)),
                )
                drafts = (c, ((1 - weight) * u.float() + weight * candidate).to(u.dtype))
            if drafts is not None and not self._valid_predictions(drafts, inputs):
                reason, drafts = "invalid_prediction", None
        outputs = drafts
        is_exact = reason is not None
        if is_exact or self.mode == "trace":
            with ledger.native():

                def count_call():
                    self.calls += 1

                outputs = tuple(exact(count_call))
            if len(outputs) != len(inputs) or any(
                not isinstance(y, torch.Tensor)
                or y.shape != x.shape
                or y.device != x.device
                or not y.is_floating_point()
                for x, y in zip(inputs, outputs)
            ):
                raise ValueError("align_cache: native velocity output contract violated")
            if not all(bool(torch.isfinite(y).all()) for y in outputs):
                self.disabled_reason = "nonfinite_exact"
                reason = self.disabled_reason
                is_exact = True
            if any(s.anchors and y.dtype != s.anchors[-1].output.dtype for s, y in zip(self.streams, outputs)):
                raise ValueError("align_cache: output dtype changed")
        assert outputs is not None
        contract = tuple((tuple(y.shape), y.dtype, y.device) for y in outputs)
        if self.output_contract is not None and contract != self.output_contract:
            raise ValueError("align_cache: native output contract changed")
        if self.qwen and len(outputs) == 2 and outputs[0].dtype != outputs[1].dtype:
            raise ValueError("align_cache: CFG branch output dtype differs")
        states = [s.fork() for s in self.streams]
        transfer = self.transfer.copy()
        if is_exact and self.disabled_reason is None and self.mode != "passthrough":
            if reason == "budget" and drafts is not None:
                pairs = [(combine(drafts), combine(outputs))] if self.qwen else list(zip(drafts, outputs))
                for j, (draft, output) in enumerate(pairs):
                    error = relative_mse(draft, output)
                    if not self.qwen:
                        error = math.sqrt(error)
                    if math.isfinite(error):
                        fb = self.config.feedback
                        value = clamp(error / max(budgets[j], fb.eps), fb.ratio_low, fb.ratio_high)
                        states[j].feedback[phase] = (
                            fb.ratio_ema * states[j].feedback[phase] + (1 - fb.ratio_ema) * value
                        )
            if self.qwen and len(states) == 2 and states[0].anchors:
                dc = sample_velocity(outputs[0], True) - sample_velocity(states[0].anchors[-1].output, True)
                du = sample_velocity(outputs[1], True) - sample_velocity(states[1].anchors[-1].output, True)
                transfer[phase] = 0.9 * transfer[phase] + 0.1 * fit(du, dc, transfer[phase], 0, 2)
            for j, (state, x, y) in enumerate(zip(states, inputs, outputs)):
                if state.anchors:
                    old = state.anchors[-1].output.float()
                    target, basis = (
                        (sample_velocity(y, True), sample_velocity(old, True)) if self.qwen else (y.float(), old)
                    )
                    low, high, decay = (0.9, 1.1, 0.9) if self.qwen else (0.85, 1.18, 0.85)
                    state.scales[phase] = decay * state.scales[phase] + (1 - decay) * fit(
                        target, basis, state.scales[phase], low, high
                    )
                    current_sample = sample_velocity(y, self.qwen)
                    previous_sample = sample_velocity(old, self.qwen)
                    na, nb = float(current_sample.norm()), float(previous_sample.norm())
                    if nb > 1e-12:
                        val = na / nb
                        state.ratio = val if state.ratio is None else 0.8 * state.ratio + 0.2 * val
                    if na * nb > 1e-12:
                        val = clamp(float(torch.dot(current_sample, previous_sample)) / (na * nb), -1, 1)
                        state.direction = val if state.direction is None else 0.8 * state.direction + 0.2 * val
                state.anchors = (
                    *state.anchors[-1:],
                    VelocityAnchor(self._input(x), y.detach().clone(), self.schedule[j][index], index),
                )
        for state, x, budget in zip(states, inputs, budgets):
            state.previous_input = (
                self._input(x) if self.disabled_reason is None and self.mode != "passthrough" else None
            )
            state.budget = 0.0 if is_exact else budget
        self.streams = states
        if self.disabled_reason is not None:
            self.streams = [VelocityStream() for _ in inputs]
        self.output_contract = contract
        self.transfer = transfer
        self.index += 1
        self.counts[reason or "fast"] += 1
        self.counts["actual_exact" if is_exact or self.mode == "trace" else "actual_fast"] += 1
        return outputs
