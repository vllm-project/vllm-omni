"""Framework boundaries for Talker codec sampling.

Production stochastic sampling uses a logits-filter boundary implemented in
graph-capturable tensor ops: history counting, temperature, repetition
penalty, EOS mask, top-p and top-k, while native NPU operators keep softmax
and ``torch.multinomial`` semantics (including Generator state) unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

VOCAB_SIZE = 6562
HISTORY_WINDOW = 16
MIN_TOKENS_TO_KEEP = 3


@dataclass(frozen=True)
class TalkerCodecDeviceState:
    history: torch.Tensor
    history_len: torch.Tensor
    step: torch.Tensor
    max_tokens: torch.Tensor
    finished: torch.Tensor


@dataclass(frozen=True)
class TalkerCodecSampleResult:
    sampled_token: torch.Tensor
    state: TalkerCodecDeviceState
    emit: torch.Tensor


def make_device_state(
    codes: torch.Tensor,
    *,
    step: int,
    max_tokens: int,
    finished: bool,
) -> TalkerCodecDeviceState:
    """Create one persistent B=1 state at request/segment setup time."""
    recent = codes.reshape(-1)[-HISTORY_WINDOW:].to(dtype=torch.int32)
    history = torch.zeros((1, HISTORY_WINDOW), dtype=torch.int32, device=codes.device)
    if recent.numel():
        history[0, : recent.numel()].copy_(recent)
    return TalkerCodecDeviceState(
        history=history,
        history_len=torch.tensor([recent.numel()], dtype=torch.int32, device=codes.device),
        step=torch.tensor([step], dtype=torch.int32, device=codes.device),
        max_tokens=torch.tensor([max_tokens], dtype=torch.int32, device=codes.device),
        finished=torch.tensor([finished], dtype=torch.bool, device=codes.device),
    )


def prepare_codec_logits(
    raw_logits: torch.Tensor,
    state: TalkerCodecDeviceState,
    min_tokens: torch.Tensor,
    temperature: torch.Tensor,
    repetition_penalty: torch.Tensor,
    *,
    eos_token_id: int,
    top_k: int,
    top_p: float,
    min_tokens_to_keep: int = MIN_TOKENS_TO_KEEP,
) -> torch.Tensor:
    """Prepare and filter logits without changing native NPU RNG semantics."""
    logits = raw_logits.float() / temperature.reshape(1, 1)
    positions = torch.arange(HISTORY_WINDOW, dtype=torch.int32, device=raw_logits.device).reshape(1, -1)
    valid = positions < state.history_len.reshape(1, 1)
    safe_tokens = torch.where(valid, state.history, torch.zeros_like(state.history)).to(torch.long)
    counts = torch.zeros((1, VOCAB_SIZE), dtype=torch.float32, device=raw_logits.device)
    counts.scatter_add_(1, safe_tokens, valid.to(torch.float32))
    alpha = torch.pow(repetition_penalty.reshape(1, 1), counts)
    logits = torch.where(logits < 0, logits * alpha, logits / alpha)
    eos = logits[..., eos_token_id : eos_token_id + 1]
    eos = torch.where(
        (state.step < min_tokens).reshape(1, 1),
        torch.full_like(eos, float("-inf")),
        eos,
    )
    logits = torch.cat([logits[..., :eos_token_id], eos, logits[..., eos_token_id + 1 :]], dim=-1)
    if 0.0 < float(top_p) < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=False, dim=-1)
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cumulative_probs <= (1.0 - float(top_p))
        remove[..., -min_tokens_to_keep:] = False
        remove = remove.scatter(-1, sorted_indices, remove)
        logits.masked_fill_(remove, float("-inf"))
    if top_k > 0:
        keep = min(VOCAB_SIZE, max(int(top_k), int(min_tokens_to_keep)))
        threshold = torch.topk(logits, keep, dim=-1).values[..., -1, None]
        logits.masked_fill_(logits < threshold, float("-inf"))
    return logits


def advance_codec_device_state(
    state: TalkerCodecDeviceState,
    sampled_token: torch.Tensor,
    *,
    eos_token_id: int,
) -> TalkerCodecDeviceState:
    """Advance persistent stochastic state with graph-capturable tensor ops."""
    sampled = sampled_token.reshape(1).to(torch.int32)
    active = ~state.finished
    next_step = torch.where(active, state.step + 1, state.step)
    is_eos = sampled == int(eos_token_id)
    reached_limit = active & (next_step >= state.max_tokens)
    finished = state.finished | is_eos | reached_limit
    emit = active & (~is_eos) & (~reached_limit)
    shifted = torch.cat([state.history[:, 1:], sampled.reshape(1, 1)], dim=1)
    append_at = state.history_len.clamp(min=0, max=HISTORY_WINDOW - 1).to(torch.long).reshape(1, 1)
    appended = state.history.scatter(1, append_at, sampled.reshape(1, 1))
    candidate = torch.where((state.history_len >= HISTORY_WINDOW).reshape(1, 1), shifted, appended)
    history = torch.where(emit.reshape(1, 1), candidate, state.history)
    history_len = torch.clamp(state.history_len + emit.to(torch.int32), max=HISTORY_WINDOW)
    return TalkerCodecDeviceState(history, history_len, next_step, state.max_tokens, finished)


def codec_sample_result(
    state: TalkerCodecDeviceState,
    sampled_token: torch.Tensor,
    *,
    eos_token_id: int,
) -> TalkerCodecSampleResult:
    """Advance device state and retain the device-side emit decision."""
    sampled = sampled_token.reshape(1).to(torch.int32)
    active = ~state.finished
    next_step = torch.where(active, state.step + 1, state.step)
    emit = active & (sampled != int(eos_token_id)) & (next_step < state.max_tokens)
    next_state = advance_codec_device_state(state, sampled, eos_token_id=eos_token_id)
    return TalkerCodecSampleResult(sampled_token=sampled, state=next_state, emit=emit)


def greedy_codec_sample(
    raw_logits: torch.Tensor,
    state: TalkerCodecDeviceState,
    min_tokens: torch.Tensor,
    repetition_penalty: torch.Tensor,
    *,
    top_k: int,
    eos_token_id: int,
) -> TalkerCodecSampleResult:
    """Greedy codec sample in graph-capturable tensor ops."""
    positions = torch.arange(HISTORY_WINDOW, dtype=torch.int32, device=raw_logits.device).reshape(1, -1)
    valid = positions < state.history_len.reshape(1, 1)
    safe_tokens = torch.where(valid, state.history, torch.zeros_like(state.history)).to(torch.long)
    counts = torch.zeros((1, VOCAB_SIZE), dtype=torch.float32, device=raw_logits.device)
    counts.scatter_add_(1, safe_tokens, valid.to(torch.float32))
    alpha = torch.pow(repetition_penalty.reshape(1, 1), counts)
    penalized = torch.where(raw_logits < 0, raw_logits * alpha, raw_logits / alpha)

    eos = penalized[..., eos_token_id : eos_token_id + 1]
    eos = torch.where(
        (state.step < min_tokens).reshape(1, 1),
        torch.full_like(eos, float("-inf")),
        eos,
    )
    penalized = torch.cat([penalized[..., :eos_token_id], eos, penalized[..., eos_token_id + 1 :]], dim=-1)
    if top_k > 0:
        keep = min(VOCAB_SIZE, max(int(top_k), MIN_TOKENS_TO_KEEP))
        threshold = torch.topk(penalized, keep, dim=-1).values[..., -1, None]
        penalized = penalized.masked_fill(penalized < threshold, float("-inf"))
    sampled = torch.argmax(penalized, dim=-1).to(torch.int32)

    active = ~state.finished
    next_step = torch.where(active, state.step + 1, state.step)
    is_eos = sampled.to(torch.int64) == int(eos_token_id)
    reached_limit = active & (next_step >= state.max_tokens)
    finished = state.finished | is_eos | reached_limit
    emit = active & (~is_eos) & (~reached_limit)

    shifted = torch.cat([state.history[:, 1:], sampled.to(torch.int32).unsqueeze(1)], dim=1)
    append_at = state.history_len.clamp(min=0, max=HISTORY_WINDOW - 1).to(torch.long).unsqueeze(1)
    appended = state.history.scatter(1, append_at, sampled.to(torch.int32).unsqueeze(1))
    candidate = torch.where((state.history_len >= HISTORY_WINDOW).unsqueeze(1), shifted, appended)
    history = torch.where(emit.unsqueeze(1), candidate, state.history)
    history_len = torch.clamp(state.history_len + emit.to(torch.int32), max=HISTORY_WINDOW)
    return TalkerCodecSampleResult(
        sampled_token=sampled,
        state=TalkerCodecDeviceState(history, history_len, next_step, state.max_tokens, finished),
        emit=emit,
    )
