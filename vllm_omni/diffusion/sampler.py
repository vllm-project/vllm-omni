# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request-local deterministic samplers operating on denoised (x0) predictions.

KSAMPLER wraps an algorithm function that owns model evaluation and integration.
Models supply an x0 callable; schedules and history live in request-local states.
The legacy precomputed-x0 step helpers remain available for existing callers.

RES coefficients follow ComfyUI_RH_MinMaxH3 sampler_core.py at
6abdf97ae8cdce548c751ce850adc5010b583b7c. Euler preserves the original
floating-point operation order.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, nullcontext
from copy import copy
from dataclasses import dataclass, field

import torch


def _require_finite_tensor(tensor: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{name} must be finite")


def _validate_sigma(value: float, name: str) -> float:
    sigma = float(value)
    if not math.isfinite(sigma):
        raise ValueError(f"{name} must be finite")
    if sigma < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return sigma


def resolve_sampler_name(name: str | None) -> str:
    if name is None:
        return "euler"
    if name not in ("euler", "res_multistep", "heun"):
        raise ValueError(f"Unsupported sampler: {name!r}; expected euler, res_multistep or heun")
    return name


def _validate_sigmas(sigmas: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(s) for s in sigmas)
    if len(values) < 2 or any(not math.isfinite(s) or not 0 <= s <= 1 for s in values):
        raise ValueError("Sampler sigmas must contain at least two finite values in [0, 1]")
    if any(a <= b for a, b in zip(values, values[1:])):
        raise ValueError("Sampler sigmas must be strictly decreasing")
    return values


class BaseSampler(ABC):
    """Legacy update-only interface for callers supplying precomputed x0."""

    def __init__(self, sigmas: Sequence[float]) -> None:
        self.sigmas = _validate_sigmas(sigmas)
        self.step_index = 0

    def step(self, state: torch.Tensor, denoised: torch.Tensor, step: int) -> torch.Tensor:
        if step != self.step_index or not 0 <= step < len(self.sigmas) - 1:
            raise ValueError("Sampler steps must be consumed once, in order")
        out = self._step(state, denoised, step)
        self.step_index += 1
        return out

    @abstractmethod
    def _step(self, state: torch.Tensor, denoised: torch.Tensor, step: int) -> torch.Tensor:
        """Apply the algorithm without advancing the shared step index."""
        ...


class EulerSampler(BaseSampler):
    """Deterministic Euler updates from denoised predictions."""

    def _step(self, state: torch.Tensor, denoised: torch.Tensor, step: int) -> torch.Tensor:
        curr, nxt = self.sigmas[step : step + 2]
        return self.step_denoised(state, denoised, sigma_curr=curr, sigma_next=nxt)

    @staticmethod
    def step_denoised(
        state: torch.Tensor,
        denoised: torch.Tensor,
        *,
        sigma_curr: float,
        sigma_next: float,
    ) -> torch.Tensor:
        if state.shape != denoised.shape:
            raise ValueError(f"state and denoised shapes must match, got {state.shape} vs {denoised.shape}")
        if not torch.is_floating_point(state):
            raise ValueError("state must be a floating point tensor")
        if not torch.is_floating_point(denoised):
            raise ValueError("denoised must be a floating point tensor")
        _require_finite_tensor(state, "state")
        _require_finite_tensor(denoised, "denoised")
        sigma_curr = _validate_sigma(sigma_curr, "sigma_curr")
        sigma_next = _validate_sigma(sigma_next, "sigma_next")
        if sigma_curr == 0.0:
            if sigma_next != 0.0:
                raise ValueError("sigma_next must be 0 when sigma_curr is 0")
            return state
        compute_dtype = torch.float32
        if state.dtype not in (torch.float16, torch.bfloat16):
            compute_dtype = state.dtype
        sigma_curr_t = state.new_tensor(sigma_curr, dtype=compute_dtype)
        sigma_next_t = state.new_tensor(sigma_next, dtype=compute_dtype)
        sigma_ratio = sigma_next_t / sigma_curr_t
        out = sigma_ratio * state.to(dtype=compute_dtype) + (1.0 - sigma_ratio) * denoised.to(dtype=compute_dtype)
        out = out.to(dtype=state.dtype)
        _require_finite_tensor(out, "euler_eta0_step output")
        return out


class ResMultistepSampler(BaseSampler):
    """RES updates owning x0 history, with Euler at the first/zero-sigma step."""

    def __init__(self, sigmas: Sequence[float]) -> None:
        super().__init__(sigmas)
        self.coeffs = self.compute_coefficients(self.sigmas)
        self.old_denoised: torch.Tensor | None = None

    def _step(self, state: torch.Tensor, denoised: torch.Tensor, step: int) -> torch.Tensor:
        coeff = self.coeffs[step]
        curr, nxt = self.sigmas[step : step + 2]
        if coeff is None:
            out = EulerSampler.step_denoised(state, denoised, sigma_curr=curr, sigma_next=nxt)
        else:
            old = self.old_denoised
            if old is None or state.shape != denoised.shape or old.shape != state.shape:
                raise ValueError("Sampler state and denoised history shapes must match")
            if not torch.is_floating_point(state) or not torch.is_floating_point(denoised):
                raise ValueError("Sampler requires floating point tensors")
            if not torch.isfinite(state).all() or not torch.isfinite(denoised).all():
                raise ValueError("Sampler inputs must be finite")
            dtype = torch.float32 if state.dtype in (torch.float16, torch.bfloat16) else state.dtype
            ratio = state.new_tensor(nxt / curr, dtype=dtype)
            hb1, hb2 = coeff
            out = (ratio * state.to(dtype) + hb1 * denoised.to(dtype) + hb2 * old.to(dtype)).to(state.dtype)
            if not torch.isfinite(out).all():
                raise ValueError("Sampler output must be finite")
        self.old_denoised = denoised.detach().clone() if step + 1 < len(self.coeffs) else None
        return out

    @staticmethod
    def compute_coefficients(sigmas: Sequence[float]) -> list[tuple[float, float] | None]:
        values = _validate_sigmas(sigmas)
        coeffs: list[tuple[float, float] | None] = []
        for i, (curr, nxt) in enumerate(zip(values, values[1:])):
            if i == 0 or nxt == 0:
                coeffs.append(None)
                continue
            h = math.log(curr / nxt)
            c2 = math.log(curr / values[i - 1]) / h
            # Stable evaluation near h=0, where subtracting phi1-1 loses precision.
            phi1 = math.expm1(-h) / (-h)
            phi2 = (phi1 - 1) / (-h) if abs(h) >= 1e-5 else 0.5 - h / 6 + h * h / 24 - h**3 / 120
            coeffs.append((h * (phi1 - phi2 / c2), h * phi2 / c2))
        return coeffs


Latents = tuple[torch.Tensor, ...]


@dataclass(eq=False)
class SamplingState:
    """Keep latents, sigmas, step_index, and RES history for one request.

    The sampler updates this state after each step and uses it to continue
    sampling on the next call.
    """

    latents: Latents
    sigmas: tuple[Sequence[float], ...]
    step_index: int = 0
    history: list[ResMultistepSampler] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.sigmas = tuple(_validate_sigmas(s) for s in self.sigmas)
        if not self.latents or len(self.latents) != len(self.sigmas):
            raise ValueError("Each latent stream requires one sigma schedule")
        if any(len(s) != len(self.sigmas[0]) for s in self.sigmas):
            raise ValueError("Coupled sigma schedules must have equal lengths")
        if self.step_index != 0:
            raise ValueError("New sampling states must start at step zero")
        self.history = [ResMultistepSampler(s) for s in self.sigmas]

    @property
    def finished(self) -> bool:
        return self.step_index == len(self.sigmas[0]) - 1


@dataclass
class ModelEvaluation:
    """One prediction, which may evaluate a temporary state within a solver step."""

    state: SamplingState
    latents: Latents
    sigmas: tuple[float, ...]


DenoiseModel = Callable[[Sequence[ModelEvaluation]], Sequence[Latents]]
SampleCallback = Callable[[SamplingState], None]
StepContext = Callable[[Sequence[SamplingState]], AbstractContextManager]


def _evaluate(model: DenoiseModel, evaluations: list[ModelEvaluation]) -> Sequence[Latents]:
    for evaluation in evaluations:
        for latent in evaluation.latents:
            if not torch.is_floating_point(latent):
                raise ValueError("Latents must be floating point")
            _require_finite_tensor(latent, "Model input")
    predictions = model(evaluations)
    if len(predictions) != len(evaluations):
        raise ValueError("Model must return one prediction per request")
    for evaluation, prediction in zip(evaluations, predictions, strict=True):
        if len(prediction) != len(evaluation.latents):
            raise ValueError("Model must return one x0 per latent stream")
        for x, x0 in zip(evaluation.latents, prediction, strict=True):
            if x.shape != x0.shape or not torch.is_floating_point(x0):
                raise ValueError("Model predictions must match latent shapes and be floating point")
            _require_finite_tensor(x0, "Model prediction")
    return predictions


def _sampling_steps(states: Sequence[SamplingState], num_steps: int | None) -> Iterator[list[SamplingState]]:
    if num_steps is not None and num_steps < 1:
        raise ValueError("num_steps must be positive")
    if len({id(state) for state in states}) != len(states):
        raise ValueError("Sampling states must be unique within a batch")
    count = 0
    while active := [s for s in states if not s.finished]:
        if num_steps is not None and count >= num_steps:
            break
        yield active
        count += 1


def _current_evaluations(states: Sequence[SamplingState]) -> list[ModelEvaluation]:
    return [ModelEvaluation(s, s.latents, tuple(v[s.step_index] for v in s.sigmas)) for s in states]


def _commit(states: Sequence[SamplingState], outputs: Sequence[Latents], callback: SampleCallback | None) -> None:
    # Validate the whole batch before publishing any of its new latent states.
    for output in outputs:
        for tensor in output:
            _require_finite_tensor(tensor, "Sampler output")
    for state, output in zip(states, outputs, strict=True):
        state.latents = tuple(output)
        state.step_index += 1
    if callback:
        for state in states:
            callback(state)


def sample_euler(
    model: DenoiseModel,
    states: Sequence[SamplingState],
    *,
    num_steps: int | None = None,
    callback: SampleCallback | None = None,
    step_context: StepContext | None = None,
) -> None:
    """Euler integration; the algorithm owns every model call."""
    for active in _sampling_steps(states, num_steps):
        with step_context(active) if step_context else nullcontext():
            predictions = _evaluate(model, _current_evaluations(active))
            outputs = [
                tuple(
                    EulerSampler.step_denoised(
                        x, x0, sigma_curr=s[state.step_index], sigma_next=s[state.step_index + 1]
                    )
                    for x, x0, s in zip(state.latents, prediction, state.sigmas, strict=True)
                )
                for state, prediction in zip(active, predictions, strict=True)
            ]
            _commit(active, outputs, callback)


def sample_res_multistep(
    model: DenoiseModel,
    states: Sequence[SamplingState],
    *,
    num_steps: int | None = None,
    callback: SampleCallback | None = None,
    step_context: StepContext | None = None,
) -> None:
    """RES integration with request-local, transactional prediction history."""
    for active in _sampling_steps(states, num_steps):
        with step_context(active) if step_context else nullcontext():
            predictions = _evaluate(model, _current_evaluations(active))
            histories = [[copy(h) for h in state.history] for state in active]
            outputs = [
                tuple(
                    h.step(x, x0, state.step_index) for h, x, x0 in zip(history, state.latents, prediction, strict=True)
                )
                for state, history, prediction in zip(active, histories, predictions, strict=True)
            ]
            for state, history in zip(active, histories, strict=True):
                state.history = history
            _commit(active, outputs, callback)


def sample_heun(
    model: DenoiseModel,
    states: Sequence[SamplingState],
    *,
    num_steps: int | None = None,
    callback: SampleCallback | None = None,
    step_context: StepContext | None = None,
) -> None:
    """Predict, evaluate the Euler proposal, then combine both derivatives."""
    for active in _sampling_steps(states, num_steps):
        with step_context(active) if step_context else nullcontext():
            predictions = _evaluate(model, _current_evaluations(active))
            derivatives, proposals, corrections = [], [], []
            for state, prediction in zip(active, predictions, strict=True):
                ds, trial = [], []
                for x, x0, schedule in zip(state.latents, prediction, state.sigmas, strict=True):
                    curr, nxt = schedule[state.step_index : state.step_index + 2]
                    dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
                    d = (x.to(dtype) - x0.to(dtype)) / curr
                    ds.append(d)
                    trial.append((x.to(dtype) + (nxt - curr) * d).to(x.dtype))
                derivatives.append(ds)
                proposals.append(tuple(trial))
                next_sigmas = tuple(s[state.step_index + 1] for s in state.sigmas)
                if any(s > 0 for s in next_sigmas):
                    corrections.append(ModelEvaluation(state, tuple(trial), next_sigmas))
            corrected = _evaluate(model, corrections) if corrections else []
            by_state = {evaluation.state: pred for evaluation, pred in zip(corrections, corrected, strict=True)}
            outputs = []
            for state, ds, trial in zip(active, derivatives, proposals, strict=True):
                output = []
                for j, (x, d, proposal, schedule) in enumerate(
                    zip(state.latents, ds, trial, state.sigmas, strict=True)
                ):
                    curr, nxt = schedule[state.step_index : state.step_index + 2]
                    if nxt == 0:
                        output.append(proposal)
                    else:
                        d2 = (proposal.to(d.dtype) - by_state[state][j].to(d.dtype)) / nxt
                        output.append((x.to(d.dtype) + (nxt - curr) * (d + d2) / 2).to(x.dtype))
                outputs.append(tuple(output))
            _commit(active, outputs, callback)


class Sampler(ABC):
    """ComfyUI-style sampler interface; models expose prediction, not sampling."""

    @abstractmethod
    def sample(
        self,
        model: DenoiseModel,
        states: Sequence[SamplingState],
        *,
        num_steps: int | None = None,
        callback: SampleCallback | None = None,
        step_context: StepContext | None = None,
    ) -> None: ...


class KSAMPLER(Sampler):
    """Wrap an algorithm function independently of any model or request.

    sample() updates the supplied request states. num_steps=None runs each to
    completion; num_steps=1 yields control to the serving scheduler after one
    complete algorithm step, including all of its model evaluations. Callbacks
    observe committed states; step_context encloses prediction and update.
    """

    def __init__(self, sampler_function: Callable) -> None:
        self.sampler_function = sampler_function

    def sample(
        self,
        model: DenoiseModel,
        states: Sequence[SamplingState],
        *,
        num_steps: int | None = None,
        callback: SampleCallback | None = None,
        step_context: StepContext | None = None,
    ) -> None:
        self.sampler_function(model, states, num_steps=num_steps, callback=callback, step_context=step_context)


def sampler_object(name: str | None = None) -> KSAMPLER:
    functions = {"euler": sample_euler, "res_multistep": sample_res_multistep, "heun": sample_heun}
    return KSAMPLER(functions[resolve_sampler_name(name)])
