# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request-local deterministic samplers operating on denoised (x0) predictions.

KSAMPLER wraps an algorithm function that owns model evaluation and integration.
Models supply an x0 callable; schedules and history live in request-local states.

RES coefficients follow ComfyUI_RH_MinMaxH3 sampler_core.py at
6abdf97ae8cdce548c751ce850adc5010b583b7c. Euler preserves the original
floating-point operation order.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass

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
    if name not in ("euler", "res_multistep"):
        raise ValueError(f"Unsupported sampler: {name!r}; expected euler or res_multistep")
    return name


def _validate_sigmas(sigmas: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(s) for s in sigmas)
    if len(values) < 2 or any(not math.isfinite(s) or not 0 <= s <= 1 for s in values):
        raise ValueError("Sampler sigmas must contain at least two finite values in [0, 1]")
    if any(a <= b for a, b in zip(values, values[1:])):
        raise ValueError("Sampler sigmas must be strictly decreasing")
    return values


def euler_eta0_step(
    state: torch.Tensor,
    denoised: torch.Tensor,
    *,
    sigma_curr: float,
    sigma_next: float,
) -> torch.Tensor:
    """Deterministic Euler update from a denoised (x0) prediction."""
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


def res_multistep_coefficients(sigmas: Sequence[float]) -> list[tuple[float, float] | None]:
    """Per-step (h*b1, h*b2) RES weights; None where the step falls back to Euler."""
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


def _res_multistep_step(
    state: torch.Tensor,
    denoised: torch.Tensor,
    old_denoised: torch.Tensor | None,
    coeff: tuple[float, float] | None,
    sigma_curr: float,
    sigma_next: float,
) -> torch.Tensor:
    """One RES update for one latent stream; Euler at the first/zero-sigma step."""
    if coeff is None:
        return euler_eta0_step(state, denoised, sigma_curr=sigma_curr, sigma_next=sigma_next)
    if old_denoised is None or state.shape != denoised.shape or old_denoised.shape != state.shape:
        raise ValueError("Sampler state and denoised history shapes must match")
    if not torch.is_floating_point(state) or not torch.is_floating_point(denoised):
        raise ValueError("Sampler requires floating point tensors")
    if not torch.isfinite(state).all() or not torch.isfinite(denoised).all():
        raise ValueError("Sampler inputs must be finite")
    dtype = torch.float32 if state.dtype in (torch.float16, torch.bfloat16) else state.dtype
    ratio = state.new_tensor(sigma_next / sigma_curr, dtype=dtype)
    hb1, hb2 = coeff
    out = (ratio * state.to(dtype) + hb1 * denoised.to(dtype) + hb2 * old_denoised.to(dtype)).to(state.dtype)
    if not torch.isfinite(out).all():
        raise ValueError("Sampler output must be finite")
    return out


Latents = tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class ResMultistepHistory:
    """Immutable per-stream RES coefficients and previous x0; each step builds a new one."""

    coeffs: tuple[list[tuple[float, float] | None], ...]
    old_denoised: tuple[torch.Tensor | None, ...]


@dataclass(eq=False)
class SamplingState:
    """Keep latents, sigmas, step_index, and algorithm history for one request.

    The sampler updates this state after each step and uses it to continue
    sampling on the next call. Algorithms needing history create their own
    history object; Euler leaves it as None. Build updated history
    separately and replace it after validation, so a failed step can be retried.
    """

    latents: Latents
    sigmas: tuple[Sequence[float], ...]
    step_index: int = 0
    history: object | None = None

    def __post_init__(self) -> None:
        self.sigmas = tuple(_validate_sigmas(s) for s in self.sigmas)
        if not self.latents or len(self.latents) != len(self.sigmas):
            raise ValueError("Each latent stream requires one sigma schedule")
        if any(len(s) != len(self.sigmas[0]) for s in self.sigmas):
            raise ValueError("Coupled sigma schedules must have equal lengths")
        if self.step_index != 0:
            raise ValueError("New sampling states must start at step zero")

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


def _commit(
    states: Sequence[SamplingState],
    outputs: Sequence[Latents],
    callback: SampleCallback | None,
    *,
    histories: Sequence[object] | None = None,
) -> None:
    # Validate the whole batch before publishing any of its new latent states.
    if len(outputs) != len(states) or (histories is not None and len(histories) != len(states)):
        raise ValueError("Each sampling state requires one output and, when supplied, one history")
    for output in outputs:
        for tensor in output:
            _require_finite_tensor(tensor, "Sampler output")
    for index, (state, output) in enumerate(zip(states, outputs, strict=True)):
        state.latents = tuple(output)
        state.step_index += 1
        if histories is not None:
            state.history = histories[index]
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
                    euler_eta0_step(x, x0, sigma_curr=s[state.step_index], sigma_next=s[state.step_index + 1])
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
            histories = []
            for state in active:
                history = state.history
                if history is None:
                    if state.step_index != 0:
                        raise ValueError("RES history is required to resume sampling")
                    history = ResMultistepHistory(
                        tuple(res_multistep_coefficients(s) for s in state.sigmas), (None,) * len(state.sigmas)
                    )
                elif not isinstance(history, ResMultistepHistory):
                    raise ValueError("RES sampling requires ResMultistepHistory")
                histories.append(history)
            predictions = _evaluate(model, _current_evaluations(active))
            outputs, new_histories = [], []
            for state, history, prediction in zip(active, histories, predictions, strict=True):
                i = state.step_index
                outputs.append(
                    tuple(
                        _res_multistep_step(x, x0, old, coeffs[i], s[i], s[i + 1])
                        for x, x0, old, coeffs, s in zip(
                            state.latents, prediction, history.old_denoised, history.coeffs, state.sigmas, strict=True
                        )
                    )
                )
                # A new history object; the committed one stays intact if this step fails.
                keep = i + 1 < len(state.sigmas[0]) - 1
                new_histories.append(
                    ResMultistepHistory(
                        history.coeffs, tuple(x0.detach().clone() if keep else None for x0 in prediction)
                    )
                )
            _commit(active, outputs, callback, histories=new_histories)


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
    functions = {"euler": sample_euler, "res_multistep": sample_res_multistep}
    return KSAMPLER(functions[resolve_sampler_name(name)])
