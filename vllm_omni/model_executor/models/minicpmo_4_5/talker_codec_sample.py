# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Framework boundaries for Talker codec sampling.

Production stochastic sampling uses a logits-filter boundary implemented in
graph-capturable tensor ops: stop-token censoring, whitelist censoring,
per-id bias, repetition penalty, frequency/presence penalties, temperature,
min-p, top-k, then top-p over the top-k-filtered distribution keeping at least
one candidate, while native NPU operators keep softmax and ``torch.multinomial``
semantics (including Generator state) unchanged.

Order and defaults follow the engine's single-frame path so that enabling the
multi-frame (K-step) decode does not change the sampling semantics:
``vllm/v1/sample/sampler.py`` for the temperature/min-p/top-k/top-p order,
``logits_processor/builtin.py`` for the censoring, bias and min-p processors and
``model_executor/layers/utils.py`` for the penalties. Every added knob is a
no-op at its default value.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

VOCAB_SIZE = 6562
HISTORY_WINDOW = 16


@dataclass(frozen=True)
class TalkerCodecDeviceState:
    history: torch.Tensor
    history_len: torch.Tensor
    step: torch.Tensor
    max_tokens: torch.Tensor
    finished: torch.Tensor
    # Whole-stream occurrence counts, one entry per codec id, or ``None`` when
    # the request set neither frequency_penalty nor presence_penalty. The engine
    # derives both penalties from the request's entire prompt+output sequence
    # (``model_executor/layers/utils.py:73-78``), which the 16-frame ``history``
    # window cannot express; this tensor is rebuilt from the host-side record at
    # every chunk boundary and advanced in place per frame. It is mutated rather
    # than re-allocated because a fresh VOCAB_SIZE-wide tensor per frame would
    # be an allocation per decode step.
    full_bins: torch.Tensor | None = None


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
    full_bins: torch.Tensor | None = None,
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
        full_bins=full_bins,
    )


def _stop_id_tuple(eos_token_id: int, min_tokens_stop_ids: tuple[int, ...]) -> tuple[int, ...]:
    """The censored stop set, codec EOS first and de-duplicated in a stable order.

    Taken from the engine's ``all_stop_token_ids`` (``sampling_params.py:537``,
    ``:676``, ``:688``). An empty ``min_tokens_stop_ids`` keeps the
    codec-EOS-only behaviour, which is what the K-step path used before this
    knob existed.
    """
    return tuple(dict.fromkeys((int(eos_token_id), *min_tokens_stop_ids)))


def _censor_stop_logits(
    logits: torch.Tensor,
    state: TalkerCodecDeviceState,
    min_tokens: torch.Tensor,
    *,
    stop_ids: tuple[int, ...],
    eos_window_masked: bool,
) -> torch.Tensor:
    """Censor every stop id while the request is below its ``min_tokens`` floor.

    The engine's ``MinTokensLogitsProcessor`` masks ``all_stop_token_ids``
    (eos + ``stop_token_ids`` + extra eos ids) and lifts the mask by a spec-aware
    counter (``builtin.py:196-207`` / ``:296-373``); here the counter is the
    state's frame step, which the single-frame path uses as well
    (``minicpmo_4_5_omni_tts.py``'s ``mask_eos_rows``). ``eos_window_masked``
    overrides the floor for the duplex turn-end drain window, which the
    ``step < min_tokens`` criterion cannot express.
    """
    stop_index = torch.tensor(stop_ids, dtype=torch.long, device=logits.device)
    stop_logits = logits.index_select(-1, stop_index)
    if eos_window_masked:
        censored = torch.full_like(stop_logits, float("-inf"))
    else:
        censored = torch.where(
            (state.step < min_tokens).reshape(1, 1),
            torch.full_like(stop_logits, float("-inf")),
            stop_logits,
        )
    return logits.index_copy(-1, stop_index, censored)


def _censor_disallowed_logits(
    logits: torch.Tensor,
    allowed_token_ids_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Blank every id the request's whitelist excludes.

    The engine precomputes a boolean mask whose True entries are the ids to
    censor (``v1/worker/gpu_input_batch.py:463-466``) and applies it as
    ``logits.masked_fill_(mask, -inf)`` (``sampler.py:393-394``). ``None`` means
    the request set no whitelist, which is the existing behaviour.
    """
    if allowed_token_ids_mask is None:
        return logits
    return logits.masked_fill(allowed_token_ids_mask, float("-inf"))


def _apply_logit_bias(
    logits: torch.Tensor,
    logit_bias_ids: tuple[int, ...],
    logit_bias_values: tuple[float, ...],
) -> torch.Tensor:
    """Add the request's per-id biases.

    Mirrors ``LogitBiasLogitsProcessor.apply`` (``builtin.py:161-164``), which
    the engine runs after the censoring masks and before the penalties, so a
    censored id stays censored. Empty ids means the request set no bias.
    """
    if not logit_bias_ids:
        return logits
    bias_index = torch.tensor(logit_bias_ids, dtype=torch.long, device=logits.device)
    bias_values = torch.tensor(logit_bias_values, dtype=logits.dtype, device=logits.device)
    current = logits.index_select(-1, bias_index)
    return logits.index_copy(-1, bias_index, current + bias_values)


def _filter_codec_logits(
    logits: torch.Tensor,
    state: TalkerCodecDeviceState,
    min_tokens: torch.Tensor,
    *,
    stop_ids: tuple[int, ...],
    allowed_token_ids_mask: torch.Tensor | None,
    logit_bias_ids: tuple[int, ...],
    logit_bias_values: tuple[float, ...],
    eos_window_masked: bool,
) -> torch.Tensor:
    """The masks both sampling paths share, in the engine's order.

    ``allowed ids -> stop censoring -> logit bias`` are all applied by the engine
    before its penalties (``sampler.py:393-405``); the bias is last because it is
    additive while the other two write ``-inf``, so a censored id must not be
    revived by a bias.
    """
    logits = _censor_disallowed_logits(logits, allowed_token_ids_mask)
    logits = _censor_stop_logits(
        logits,
        state,
        min_tokens,
        stop_ids=stop_ids,
        eos_window_masked=eos_window_masked,
    )
    return _apply_logit_bias(logits, logit_bias_ids, logit_bias_values)


def _repetition_penalty_scaling(
    logits: torch.Tensor,
    state: TalkerCodecDeviceState,
    repetition_penalty: torch.Tensor,
) -> torch.Tensor:
    """Repetition penalty over the codec history window (engine step 6a).

    ``alpha = penalty ** frequency`` with the sign-dependent branch, matching the
    model's single-frame ``_apply_batched_repetition_penalty``.
    """
    positions = torch.arange(HISTORY_WINDOW, dtype=torch.int32, device=logits.device).reshape(1, -1)
    valid = positions < state.history_len.reshape(1, 1)
    safe_tokens = torch.where(valid, state.history, torch.zeros_like(state.history)).to(torch.long)
    counts = torch.zeros((1, VOCAB_SIZE), dtype=torch.float32, device=logits.device)
    counts.scatter_add_(1, safe_tokens, valid.to(torch.float32))
    alpha = torch.pow(repetition_penalty.reshape(1, 1), counts)
    return torch.where(logits < 0, logits * alpha, logits / alpha)


def _apply_stream_penalties(
    logits: torch.Tensor,
    state: TalkerCodecDeviceState,
    frequency_penalty: float,
    presence_penalty: float,
) -> torch.Tensor:
    """Whole-stream frequency and presence penalties (engine steps 6b/6c).

    Straight from ``model_executor/layers/utils.py:86-87``:

        logits -= frequency_penalties * output_bin_counts
        logits -= presence_penalties * output_mask

    ``output_bin_counts`` is the occurrence count over the whole prompt+output
    sequence and ``output_mask`` is its ``> 0`` image; both come from the
    request's stream record. Skipped entirely when the request set neither, or
    when the record is absent (nothing to subtract from).
    """
    if state.full_bins is None:
        return logits
    if not frequency_penalty and not presence_penalty:
        return logits
    bins = state.full_bins.reshape(1, -1)
    if frequency_penalty:
        logits = logits - bins * frequency_penalty
    if presence_penalty:
        logits = logits - (bins > 0).to(logits.dtype) * presence_penalty
    return logits


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
    min_p: float = 0.0,
    min_tokens_stop_ids: tuple[int, ...] = (),
    allowed_token_ids_mask: torch.Tensor | None = None,
    logit_bias_ids: tuple[int, ...] = (),
    logit_bias_values: tuple[float, ...] = (),
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    eos_window_masked: bool = False,
) -> torch.Tensor:
    """Prepare and filter logits without changing native NPU RNG semantics.

    ``min_p`` mirrors the engine's argmax-invariant ``MinPLogitsProcessor``,
    which runs after temperature and before top-k/top-p (``builtin.py:102-113``).
    ``min_tokens_stop_ids`` widens the censored stop set beyond the codec EOS;
    ``allowed_token_ids_mask`` applies the request's whitelist;
    ``logit_bias_ids``/``logit_bias_values`` add the request's per-id biases;
    ``frequency_penalty``/``presence_penalty`` subtract the whole-stream counts.
    Every one of them is a no-op at its default value.
    """
    stop_ids = _stop_id_tuple(eos_token_id, min_tokens_stop_ids)
    logits = raw_logits.float()
    # Engine steps 5a / 3-4 / 5b: whitelist, stop censoring and bias, in the
    # order the engine applies them, before any penalty.
    logits = _filter_codec_logits(
        logits,
        state,
        min_tokens,
        stop_ids=stop_ids,
        allowed_token_ids_mask=allowed_token_ids_mask,
        logit_bias_ids=logit_bias_ids,
        logit_bias_values=logit_bias_values,
        eos_window_masked=eos_window_masked,
    )
    # Engine step 6a: repetition penalty over the codec history window.
    logits = _repetition_penalty_scaling(logits, state, repetition_penalty)
    # Engine steps 6b/6c: frequency and presence penalties over the whole stream.
    logits = _apply_stream_penalties(logits, state, frequency_penalty, presence_penalty)
    # Engine step 7b: temperature is applied after the penalties. The division
    # also gives the in-place masks below a tensor of their own.
    logits = logits / temperature.reshape(1, 1)
    # Engine step 7c: min_p. The engine runs it as an argmax-invariant
    # processor, i.e. after temperature and before top-k/top-p, and the greedy
    # branch never sees it -- which is why greedy_codec_sample has no min_p.
    if min_p > 0.0:
        probabilities = torch.softmax(logits, dim=-1)
        min_p_threshold = probabilities.amax(dim=-1, keepdim=True) * min_p
        logits.masked_fill_(probabilities < min_p_threshold, float("-inf"))
    # Engine step 7d: top-k first (topk_topp_sampler.py:392).
    if top_k > 0:
        keep = min(VOCAB_SIZE, int(top_k))
        threshold = torch.topk(logits, keep, dim=-1).values[..., -1, None]
        logits.masked_fill_(logits < threshold, float("-inf"))
    # Engine step 7d: top-p runs over the top-k-filtered distribution and keeps
    # at least one candidate (topk_topp_sampler.py:404/:415).
    if 0.0 < float(top_p) < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=False, dim=-1)
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cumulative_probs <= (1.0 - float(top_p))
        remove[..., -1:] = False
        remove = remove.scatter(-1, sorted_indices, remove)
        logits.masked_fill_(remove, float("-inf"))
    return logits


def _advance_full_bins(
    state: TalkerCodecDeviceState,
    sampled: torch.Tensor,
    emit: torch.Tensor,
) -> None:
    """Add one occurrence for the emitted id, in place.

    In-place on purpose: re-allocating a VOCAB_SIZE-wide tensor every frame would
    be an allocation per decode step. The tensor belongs to the request's state
    record, which is replaced every frame, so mutating it cannot be observed by
    an earlier state.
    """
    if state.full_bins is None:
        return
    state.full_bins.scatter_add_(
        0,
        sampled.reshape(-1).to(torch.long),
        emit.reshape(-1).to(state.full_bins.dtype),
    )


def advance_codec_device_state(
    state: TalkerCodecDeviceState,
    sampled_token: torch.Tensor,
    *,
    eos_token_id: int,
    ignore_eos: bool = False,
) -> TalkerCodecDeviceState:
    """Advance persistent stochastic state with graph-capturable tensor ops."""
    sampled = sampled_token.reshape(1).to(torch.int32)
    active = ~state.finished
    next_step = torch.where(active, state.step + 1, state.step)
    if ignore_eos:
        # ``ignore_eos`` blanks the engine's ``_eos_token_id``
        # (vllm/sampling_params.py:670-671), so a sampled codec EOS is just
        # another frame and must not finish the request.
        is_eos = torch.zeros_like(state.finished)
    else:
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
    _advance_full_bins(state, sampled, emit)
    return TalkerCodecDeviceState(
        history, history_len, next_step, state.max_tokens, finished, state.full_bins
    )


def codec_sample_result(
    state: TalkerCodecDeviceState,
    sampled_token: torch.Tensor,
    *,
    eos_token_id: int,
    ignore_eos: bool = False,
) -> TalkerCodecSampleResult:
    """Advance device state and retain the device-side emit decision."""
    sampled = sampled_token.reshape(1).to(torch.int32)
    active = ~state.finished
    next_step = torch.where(active, state.step + 1, state.step)
    if ignore_eos:
        is_eos = torch.zeros_like(state.finished)
    else:
        is_eos = sampled == int(eos_token_id)
    emit = active & (~is_eos) & (next_step < state.max_tokens)
    next_state = advance_codec_device_state(
        state,
        sampled,
        eos_token_id=eos_token_id,
        ignore_eos=ignore_eos,
    )
    return TalkerCodecSampleResult(sampled_token=sampled, state=next_state, emit=emit)


def greedy_codec_sample(
    raw_logits: torch.Tensor,
    state: TalkerCodecDeviceState,
    min_tokens: torch.Tensor,
    repetition_penalty: torch.Tensor,
    *,
    eos_token_id: int,
    min_tokens_stop_ids: tuple[int, ...] = (),
    allowed_token_ids_mask: torch.Tensor | None = None,
    logit_bias_ids: tuple[int, ...] = (),
    logit_bias_values: tuple[float, ...] = (),
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    ignore_eos: bool = False,
    eos_window_masked: bool = False,
) -> TalkerCodecSampleResult:
    """Greedy codec sample in graph-capturable tensor ops.

    The engine's greedy branch returns the argmax after the censoring masks, the
    logit bias and the penalties, and skips temperature, min-p, top-k and top-p
    entirely (``vllm/v1/sample/sampler.py:30-60``), so no top-k mask belongs on
    this path -- masking candidates below the argmax cannot change the argmax
    itself. Every other filter is the same one the random path uses.

    ``eos_window_masked`` masks the codec EOS for this frame regardless of the
    step counter -- see prepare_codec_logits.
    """
    stop_ids = _stop_id_tuple(eos_token_id, min_tokens_stop_ids)
    penalized = raw_logits.float()
    # Engine steps 5a / 3-4 / 5b: same masks as the random path.
    penalized = _filter_codec_logits(
        penalized,
        state,
        min_tokens,
        stop_ids=stop_ids,
        allowed_token_ids_mask=allowed_token_ids_mask,
        logit_bias_ids=logit_bias_ids,
        logit_bias_values=logit_bias_values,
        eos_window_masked=eos_window_masked,
    )
    # Engine steps 6a/6b/6c: same penalties as the random path.
    penalized = _repetition_penalty_scaling(penalized, state, repetition_penalty)
    penalized = _apply_stream_penalties(penalized, state, frequency_penalty, presence_penalty)
    sampled = torch.argmax(penalized, dim=-1).to(torch.int32)

    active = ~state.finished
    next_step = torch.where(active, state.step + 1, state.step)
    if ignore_eos:
        # Same rule as advance_codec_device_state: ``ignore_eos`` disables the
        # engine's EOS stop, so it disables this one too.
        is_eos = torch.zeros_like(state.finished)
    else:
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
    _advance_full_bins(state, sampled, emit)
    return TalkerCodecSampleResult(
        sampled_token=sampled,
        state=TalkerCodecDeviceState(
            history, history_len, next_step, state.max_tokens, finished, state.full_bins
        ),
        emit=emit,
    )
