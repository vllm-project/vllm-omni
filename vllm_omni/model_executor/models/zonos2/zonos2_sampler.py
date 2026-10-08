# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request-local ZONOS2 sampling and reconstructible nine-codebook history."""

from __future__ import annotations

import math
import secrets
from dataclasses import dataclass
from typing import Any

import torch

CONTINUE_TOKEN = 0
STOP_TOKEN = 1


@dataclass(frozen=True)
class Zonos2SamplingParams:
    temperature: float = 1.15
    top_k: int = 106
    top_p: float = 1.0
    min_p: float = 0.18
    repetition_window: int = 50
    repetition_penalty: float = 1.2
    repetition_codebooks: int = 8
    seed: int | None = 42
    max_tokens: int = 1024
    ignore_eos: bool = False

    def __post_init__(self):
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        for key in ("top_k", "repetition_window", "repetition_codebooks", "max_tokens"):
            if not isinstance(getattr(self, key), int) or isinstance(getattr(self, key), bool):
                raise ValueError(f"{key} must be an integer")
        if self.seed is not None and not isinstance(self.seed, int):
            raise ValueError("seed must be an integer or None")
        if self.top_k < -1:
            raise ValueError("top_k must be -1, 0 or positive")
        if not 0 <= self.top_p <= 1 or not 0 <= self.min_p <= 1:
            raise ValueError("top_p and min_p must be in [0,1]")
        if self.repetition_window < 0 or not math.isfinite(self.repetition_penalty) or self.repetition_penalty < 1:
            raise ValueError("repetition_window must be nonnegative and repetition_penalty >=1")
        if not 0 <= self.repetition_codebooks <= 8:
            raise ValueError("repetition_codebooks must be in [0,8]; CB8 is never penalized")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")

    @classmethod
    def from_runtime(cls, values: dict[str, Any]):
        extra = values.get("extra_args") or {}
        args = {
            key: values[key]
            for key in (
                "temperature",
                "top_k",
                "top_p",
                "min_p",
                "seed",
                "max_tokens",
                "ignore_eos",
                "repetition_penalty",
            )
            if values.get(key) is not None
        }
        if "seed" in values:
            args["seed"] = values["seed"]
        for key in (
            "temperature",
            "top_k",
            "top_p",
            "min_p",
            "repetition_window",
            "repetition_penalty",
            "repetition_codebooks",
        ):
            if key in extra:
                args[key] = extra[key]
        return cls(**args)


def frame_seed(seed: int, step: int) -> int:
    """Counter-derived RNG: reproducible after replay and batch reordering.

    This stream intentionally has no global generator and is not promised to
    reproduce the reference's version-dependent torch.multinomial bitstream.
    """
    mask = (1 << 64) - 1
    value = ((seed & mask) + (step + 1) * 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return (value ^ (value >> 31)) & ((1 << 63) - 1)


def repetition_logits(logits: torch.Tensor, history: torch.Tensor, params: Zonos2SamplingParams) -> torch.Tensor:
    adjusted = logits.float()
    if (
        params.repetition_window == 0
        or params.repetition_penalty == 1
        or not len(history)
        or params.repetition_codebooks == 0
    ):
        return adjusted
    tail = history[-params.repetition_window :].transpose(0, 1).long()
    valid = (tail >= 0) & (tail < 1024)
    valid[params.repetition_codebooks :] = False
    counts = torch.zeros_like(adjusted, dtype=torch.int32)
    counts.scatter_add_(1, tail.clamp(0, adjusted.shape[-1] - 1), valid.int())
    scaled = torch.where(adjusted > 0, adjusted / params.repetition_penalty, adjusted * params.repetition_penalty)
    return torch.where(counts > 0, scaled, adjusted)


def frame_probabilities(logits: torch.Tensor, history: torch.Tensor, params: Zonos2SamplingParams) -> torch.Tensor:
    adjusted = repetition_logits(logits, history, params)
    scaled = adjusted / max(params.temperature, 1e-8)
    if 0 < params.top_k < scaled.shape[-1]:
        cutoff = scaled.topk(params.top_k, dim=-1).values[:, -1:]
        scaled = scaled.masked_fill(scaled < cutoff, float("-inf"))
    probabilities = scaled.softmax(dim=-1)
    if 0 < params.top_p < 1:
        sorted_probs, order = probabilities.sort(dim=-1, descending=True)
        keep = sorted_probs.cumsum(-1) - sorted_probs <= params.top_p
        sorted_probs = sorted_probs * keep
        probabilities = torch.zeros_like(probabilities).scatter(1, order, sorted_probs)
        probabilities = probabilities / probabilities.sum(-1, keepdim=True).clamp_min(1e-8)
    if params.min_p > 0:
        probabilities = probabilities * (probabilities >= probabilities.amax(-1, keepdim=True) * params.min_p)
        probabilities = probabilities / probabilities.sum(-1, keepdim=True).clamp_min(1e-8)
    fallback = torch.zeros_like(probabilities).scatter(1, adjusted.argmax(-1, keepdim=True), 1)
    return torch.where(probabilities.sum(-1, keepdim=True) > 0, probabilities, fallback)


def sample_frame(
    logits: torch.Tensor, history: torch.Tensor, params: Zonos2SamplingParams, seed: int, step: int
) -> torch.Tensor:
    if params.temperature == 0:
        return repetition_logits(logits, history, params).argmax(-1).to(torch.int32)
    generator = torch.Generator(device=logits.device).manual_seed(frame_seed(seed, step))
    probabilities = frame_probabilities(logits, history, params)
    return torch.multinomial(probabilities, num_samples=1, generator=generator).reshape(9).to(torch.int32)


@dataclass
class Zonos2RequestState:
    request_id: str
    params: Zonos2SamplingParams
    seed: int
    history: torch.Tensor
    eos_frame: torch.Tensor
    countdown: torch.Tensor
    stopped: torch.Tensor

    @classmethod
    def rebuild(cls, request_id: str, params: Zonos2SamplingParams, history: torch.Tensor, seed: int | None = None):
        if history.ndim != 2 or history.shape[1] != 9 or history.dtype not in (torch.int32, torch.int64):
            raise ValueError("ZONOS2 history must be integer [T,9]")
        owned = history.detach().clone().to(torch.int32)
        device = owned.device
        eos = torch.tensor(-1, device=device, dtype=torch.long)
        remaining = torch.tensor(-1, device=device, dtype=torch.long)
        if len(owned) and not params.ignore_eos:
            mask = owned == 1024
            first = torch.where(mask.any(-1), torch.arange(len(owned), device=device), len(owned)).amin()
            valid = first < len(owned)
            last_cb = torch.where(mask, torch.arange(9, device=device), -1).amax(-1)
            aligned = (first - last_cb[first.clamp(max=len(owned) - 1)]).clamp_min(0)
            eos = torch.where(valid, aligned, eos)
            remaining = torch.where(valid, (10 - (len(owned) - first)).clamp_min(0), remaining)
        stopped = ((eos >= 0) & (remaining == 0)) | (len(owned) >= params.max_tokens)
        base = seed if seed is not None else params.seed
        return cls(request_id, params, secrets.randbits(63) if base is None else base, owned, eos, remaining, stopped)

    def append(self, codes: torch.Tensor) -> torch.Tensor:
        step = len(self.history)
        if not self.params.ignore_eos:
            mask = codes == 1024
            first = (self.eos_frame < 0) & mask.any()
            last_cb = torch.where(mask, torch.arange(9, device=codes.device), -1).amax()
            self.eos_frame = torch.where(first, (step - last_cb).clamp_min(0), self.eos_frame)
            self.countdown = torch.where(first, 10, self.countdown)
            self.countdown = torch.where(self.countdown > 0, self.countdown - 1, self.countdown)
        self.history = torch.cat((self.history, codes.reshape(1, 9)), dim=0)
        self.stopped = ((self.eos_frame >= 0) & (self.countdown == 0)) | (len(self.history) >= self.params.max_tokens)
        return torch.where(self.stopped, STOP_TOKEN, CONTINUE_TOKEN).to(torch.int64)

    def payload(self) -> dict[str, Any]:
        return {
            "history": self.history,
            "seed": self.seed,
            "eos_frame": self.eos_frame,
            "countdown": self.countdown,
            "stopped": self.stopped,
        }
