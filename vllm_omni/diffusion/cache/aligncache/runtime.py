# SPDX-License-Identifier: Apache-2.0
"""Atomic pair execution; real forward failures are never retried."""

from __future__ import annotations

import logging
import math
import time
from collections import Counter, deque
from contextlib import contextmanager
from dataclasses import asdict, replace

import torch

from .controller import decide, update_feedback, update_gain
from .memory import TensorLedger
from .predictor import correct_anchor, predict, update_alpha, write_branch
from .state import ExactAnchor, PairTransaction, RequestState, storage_bytes

logger = logging.getLogger(__name__)


class AlignCacheRuntime:
    def __init__(self, config):
        self.config = config
        self.mode = config.mode
        self.emit_logs = True
        self.epoch = 0
        self.active = False
        self.state = None
        self.plan = None
        self.signature = None
        self.last_summary = {}
        self.trace = deque(maxlen=256)
        self.counts = Counter()
        self.reasons = Counter()
        self.capacity_exceeded = False
        self.peak_cache_bytes = 0
        self.ledger = None
        self.disabled_reason = None
        self.controller_seconds = 0.0
        self.native_output_dtype = None

    def reset(self):
        if self.active:
            raise RuntimeError("align_cache: request already active; refresh/reentry forbidden")
        self.state = self.plan = self.signature = None
        self.trace.clear()

    @contextmanager
    def request_context(self):
        self.reset()
        self.active = True
        self.epoch += 1
        self.state = RequestState.create(self.epoch, self.config)
        self.counts.clear()
        self.reasons.clear()
        self.capacity_exceeded = False
        self.peak_cache_bytes = 0
        self.disabled_reason = None
        self.controller_seconds = 0.0
        self.native_output_dtype = None
        status = "failed"
        try:
            yield self
            status = "completed"
        finally:
            device_peak = None
            try:
                if self.signature is not None and self.signature.device.type == "cuda":
                    device_peak = torch.accelerator.max_memory_allocated(self.signature.device)
            except Exception:
                pass
            self.last_summary = {
                "backend": "align_cache",
                "policy": self.config.policy,
                "profile": self.config.profile,
                "mode": self.mode,
                "config_hash": self.config.config_hash,
                "epoch": self.epoch,
                "configured": True,
                "installed": True,
                "request_eligible": self.signature is not None,
                "fast_used": bool(self.counts["fast_pairs"]),
                "planned_evaluations": self.plan.evaluations if self.plan else 0,
                "completed_pairs": self.counts["completed_pairs"],
                "exact_pairs": self.counts["exact_pairs"],
                "fast_pairs": self.counts["fast_pairs"],
                "model_forward_calls": self.counts["model_forward_calls"],
                "shadow_exact_pairs": self.counts["shadow_exact_pairs"],
                "shadow_fast_pairs": self.counts["shadow_fast_pairs"],
                "feedback_updates": self.state.feedback_updates if self.state else 0,
                "fallback_counts": dict(self.reasons),
                "peak_cache_bytes": self.peak_cache_bytes,
                "active_cache_bytes": 0,
                "device_peak_allocated_bytes": device_peak,
                "controller_seconds": self.controller_seconds,
                "status": status,
            }
            self.state = self.plan = self.signature = None
            self.active = False
            self.native_output_dtype = None
            if self.emit_logs and self.config.diagnostics != "off":
                try:
                    logger.info("AlignCache request: %s", self.last_summary)
                    if self.config.diagnostics == "trace":
                        logger.info("AlignCache bounded trace (last 256): %s", list(self.trace))
                except Exception:
                    # Diagnostics must never alter generation or mask its original error.
                    pass

    def bind_schedule(self, plan, signature):
        if not self.active or self.plan is not None:
            raise RuntimeError("align_cache: schedule requires a fresh active request")
        if not plan.timesteps or not all(math.isfinite(t) for t in plan.timesteps):
            raise ValueError("align_cache.schedule: finite nonempty timesteps required")
        if any(a <= b for a, b in zip(plan.timesteps, plan.timesteps[1:])):
            raise ValueError("align_cache.schedule: strictly descending native time grid required")
        if (
            signature.epoch != self.epoch
            or signature.guidance_scale <= 1
            or not math.isfinite(signature.guidance_scale)
        ):
            raise ValueError("align_cache.signature: epoch and serial CFG > 1 required")
        self.plan, self.signature = plan, signature
        self.state.guidance = signature.guidance_scale

    @staticmethod
    def estimate_capacity(x, trace=False):
        # Live-tensor bound: 48 full FP32 buffers (50 in trace) plus sampled and scalar reductions.
        full = (50 if trace else 48) * x.numel() * 4
        sampled = 6 * ((x.numel() + 63) // 64) * 4
        return full + sampled + 4096

    def _validate_step(self, step, x, conditions):
        sig = self.signature
        if not self.active or self.plan is None or sig is None:
            raise RuntimeError("align_cache: missing schedule bind")
        if step.evaluation_index != self.counts["completed_pairs"] or step.evaluation_index >= self.plan.evaluations:
            raise ValueError("align_cache: repeated/out-of-order evaluation_index")
        if (
            step.component != sig.component
            or step.guidance_scale != sig.guidance_scale
            or tuple(x.shape) != sig.shape
            or x.dtype != sig.dtype
            or x.device != sig.device
            or conditions != sig.condition
            or step.timestep != self.plan.timesteps[step.evaluation_index]
        ):
            raise ValueError("align_cache: component, condition, CFG, tensor or schedule identity changed")

    def _exact(self, callback, branch, x):
        self.counts["model_forward_calls"] += 1
        started = time.perf_counter()
        try:
            if self.ledger is None:
                output = callback(branch)
            else:
                with self.ledger.native():
                    output = callback(branch)
        finally:
            self.controller_seconds -= time.perf_counter() - started
        if not isinstance(output, torch.Tensor) or output.shape != x.shape or output.device != x.device:
            raise ValueError("align_cache: native output must be a Tensor with input shape/device")
        if not output.is_floating_point():
            raise ValueError("align_cache: native output must have floating dtype")
        if self.native_output_dtype is not None and output.dtype != self.native_output_dtype:
            raise ValueError("align_cache: native output dtype changed within request")
        return output

    @staticmethod
    def _validate_pair_dtype(cond, uncond):
        if cond.dtype != uncond.dtype:
            raise ValueError("align_cache: conditional and unconditional native output dtype differ")

    def predict_pair(self, step, x, conditions, exact):
        started = time.perf_counter()
        ledger = TensorLedger(self.state)
        self.ledger = ledger
        try:
            with ledger:
                outputs = self._predict_pair(step, x, conditions, exact)
                self.native_output_dtype = outputs[0].dtype
                return outputs
        finally:
            self.controller_seconds += time.perf_counter() - started
            self.peak_cache_bytes = max(self.peak_cache_bytes, ledger.peak)
            self.ledger = None

    def _predict_pair(self, step, x, conditions, exact):
        self._validate_step(step, x, conditions)
        if self.mode == "passthrough":
            raise RuntimeError("align_cache: passthrough must use the native CFG entrypoint")
        if not self.capacity_exceeded and self.estimate_capacity(x, self.mode == "trace") > self.config.max_cache_bytes:
            self.capacity_exceeded = True
            self.disabled_reason = "capacity_exceeded"
            self.state = None
        if self.disabled_reason:
            cond, uncond = self._exact(exact, "cond", x), self._exact(exact, "uncond", x)
            self._validate_pair_dtype(cond, uncond)
            self.counts["exact_pairs"] += 1
            self.counts["completed_pairs"] += 1
            self.reasons[self.disabled_reason] += 1
            return cond, uncond
        decision = decide(self.state, x, step.evaluation_index, self.plan.evaluations, self.config)
        tx = PairTransaction.begin(self.state, decision)
        candidate = None
        if not decision.exact:
            candidate = self._fast(tx, x, step)
            if not all(
                v is not None
                and v.dtype == self.state.anchor.output_dtype
                and v.shape == x.shape
                and v.device == x.device
                and bool(torch.isfinite(v).all())
                for v in candidate
            ):
                decision = replace(decision, exact=True, reason="invalid_prediction")
                tx = PairTransaction.begin(self.state, decision)
                candidate = None
        if decision.exact:
            outputs = self._compute_exact(tx, x, step, exact)
            if self.disabled_reason:
                self.state = None
                self.counts["completed_pairs"] += 1
                self.counts["exact_pairs"] += 1
                self.reasons[self.disabled_reason] += 1
                return outputs
        elif self.mode == "trace":
            outputs = self._exact(exact, "cond", x), self._exact(exact, "uncond", x)
            self._validate_pair_dtype(*outputs)
        else:
            outputs = candidate
        tx.state.completed_pairs += 1
        tx.state.previous_input = x.detach().float().clone()
        tx.state.budget = 0.0 if decision.exact else decision.budget
        self.peak_cache_bytes = max(self.peak_cache_bytes, storage_bytes(self.state, tx, candidate))
        self.state = tx.state
        self.counts["completed_pairs"] += 1
        self.counts["exact_pairs" if decision.exact or self.mode == "trace" else "fast_pairs"] += 1
        if self.mode == "trace":
            self.counts["shadow_exact_pairs" if decision.exact else "shadow_fast_pairs"] += 1
        self.reasons[decision.reason] += 1
        if self.config.diagnostics == "trace":
            self.trace.append(
                {
                    **asdict(decision),
                    "index": step.evaluation_index,
                    "feedback_scale": self.state.feedback_scale[decision.phase],
                    "anchor_reason": self.state.anchor_reason,
                }
            )
        return outputs

    def _fast(self, tx, x, step):
        state, phase = tx.state, tx.decision.phase
        cond = predict(state, x, "cond", phase, self.config)
        if cond is None:
            return None, None
        cond = cond.to(state.anchor.output_dtype)
        write_branch(state, x, cond, "cond", phase, False, self.config)
        uncond = predict(state, x, "uncond", phase, self.config)
        if uncond is None:
            return cond, None
        uncond, state.anchor_reason = correct_anchor(state, cond, uncond, step.evaluation_index, phase, self.config)
        uncond = uncond.to(state.anchor.output_dtype)
        write_branch(state, x, uncond, "uncond", phase, False, self.config)
        return cond, uncond

    def _compute_exact(self, tx, x, step, exact):
        state, phase = tx.state, tx.decision.phase
        feedback = tx.decision.reason == "budget_exact"
        if feedback:
            tx.cond_draft = predict(state, x, "cond", phase, self.config)
            if tx.cond_draft is not None:
                tx.cond_draft = tx.cond_draft.to(state.anchor.output_dtype)
        cond = self._exact(exact, "cond", x)
        if not bool(torch.isfinite(cond).all()) or not bool(torch.isfinite(x).all()):
            uncond = self._exact(exact, "uncond", x)
            self._validate_pair_dtype(cond, uncond)
            self.disabled_reason = "nonfinite_exact"
            return cond, uncond
        # Draft for uncond sees the new cond transaction, but the old exact anchor.
        write_branch(state, x, cond, "cond", phase, True, self.config)
        if feedback:
            tx.uncond_draft = predict(state, x, "uncond", phase, self.config)
            if tx.uncond_draft is not None:
                tx.uncond_draft, _ = correct_anchor(
                    state, tx.cond_draft, tx.uncond_draft, step.evaluation_index, phase, self.config
                )
                tx.uncond_draft = tx.uncond_draft.to(state.anchor.output_dtype)
        uncond = self._exact(exact, "uncond", x)
        self._validate_pair_dtype(cond, uncond)
        if not bool(torch.isfinite(uncond).all()):
            self.disabled_reason = "nonfinite_exact"
            return cond, uncond
        write_branch(state, x, uncond, "uncond", phase, True, self.config)
        update_gain(state, x, cond, self.config)
        update_alpha(state, cond, uncond, phase, self.config)
        update_feedback(
            state, tx.decision, (tx.cond_draft, tx.uncond_draft), (cond, uncond), step.guidance_scale, self.config
        )
        state.anchor = ExactAnchor(
            step.evaluation_index,
            x.detach().float().clone(),
            cond.detach().float().clone(),
            uncond.detach().float().clone(),
            state.cond.last,
            state.uncond.last,
            cond.dtype,
        )
        return cond, uncond

    def record_passthrough(self):
        self.counts["completed_pairs"] += 1
        self.counts["exact_pairs"] += 1
        self.reasons["passthrough"] += 1
