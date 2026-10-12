# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""GPU-side Lychee speech/control sampling primitives."""

from __future__ import annotations

from enum import IntEnum

import torch


class LycheeControlMode(IntEnum):
    LISTENING = 0
    SPEAKING = 1
    BACKCHANNEL = 2


def sample_control_tokens(
    logits: torch.Tensor,
    *,
    modes: torch.Tensor,
    ticks: torch.Tensor,
    config,
    allowing_backchannel: bool = True,
    start_speak_token_factor: float = 1.2,
    backchannel_token_bias: float = 1.0,
    start_listen_token_factor: float = 1.0,
) -> torch.Tensor:
    """Apply the released 10-tick control grammar and select greedily."""

    if logits.ndim != 2:
        raise ValueError(f"Expected [batch, vocab] control logits, got {tuple(logits.shape)}")
    if modes.shape != ticks.shape or modes.shape[0] != logits.shape[0]:
        raise ValueError("Control modes/ticks must have one value per logits row")

    token_ids = torch.tensor(
        [
            config.sleep_token_id,
            config.detect_token_id,
            config.start_speaking_token_id,
            config.start_listening_token_id,
            config.keep_listening_token_id,
            config.keep_speaking_token_id,
            config.start_bc_token_id,
        ],
        dtype=torch.long,
        device=logits.device,
    )
    scores = logits.index_select(-1, token_ids).float()
    if not bool(torch.isfinite(scores).all()):
        raise FloatingPointError("Non-finite Lychee control logits; abort the native KV transaction")
    legal = torch.zeros_like(scores, dtype=torch.bool)
    phase = torch.remainder(ticks, config.control_token_chunk_size)
    ordinary = phase < config.control_token_chunk_size - 2
    detect = phase == config.control_token_chunk_size - 2
    decision = phase == config.control_token_chunk_size - 1
    legal[:, 0] = ordinary
    legal[:, 1] = detect

    listening = modes == int(LycheeControlMode.LISTENING)
    speaking = modes == int(LycheeControlMode.SPEAKING)
    backchannel = modes == int(LycheeControlMode.BACKCHANNEL)
    legal[:, 2] = decision & (listening | backchannel)
    legal[:, 3] = decision & (speaking | backchannel)
    legal[:, 4] = decision & listening
    legal[:, 5] = decision & speaking
    legal[:, 6] = decision & ((listening & allowing_backchannel) | backchannel)

    scores[:, 2] *= start_speak_token_factor
    scores[:, 3] *= torch.where(speaking, scores.new_tensor(start_listen_token_factor), scores.new_tensor(1.0))
    scores[:, 6] += torch.where(listening, scores.new_tensor(backchannel_token_bias), scores.new_tensor(0.0))
    scores.masked_fill_(~legal, float("-inf"))
    return token_ids[scores.argmax(dim=-1)]


def update_control_modes(
    modes: torch.Tensor,
    control_tokens: torch.Tensor,
    *,
    config,
) -> torch.Tensor:
    """Apply request-local control transitions without a device sync."""

    updated = modes.clone()
    updated = torch.where(
        control_tokens == config.start_speaking_token_id,
        updated.new_tensor(int(LycheeControlMode.SPEAKING)),
        updated,
    )
    updated = torch.where(
        control_tokens == config.start_listening_token_id,
        updated.new_tensor(int(LycheeControlMode.LISTENING)),
        updated,
    )
    updated = torch.where(
        control_tokens == config.start_bc_token_id,
        updated.new_tensor(int(LycheeControlMode.BACKCHANNEL)),
        updated,
    )
    return updated


def speech_audio_token_id_max(config) -> int:
    """Released service restricts audio IDs to the actual Token2Wav codebook."""
    return min(
        config.stoken_token_ids_max,
        config.stoken_audio_token_id_min + getattr(config, "stoken_codec_vocab_size", 6_561),
    )


def sample_speech_tokens(
    logits: torch.Tensor,
    *,
    modes: torch.Tensor,
    speaking_steps: torch.Tensor,
    config,
    generator: torch.Generator,
    speech_history: torch.Tensor | None = None,
    history_lengths: torch.Tensor | None = None,
    temperature: float | None = None,
) -> torch.Tensor:
    """Sample the speech channel with pad/delay/start prefix semantics.

    Each invocation samples one request with its own RNG. The model state
    dispatches independent active rows so batch order cannot share a stream.
    """

    if logits.shape[0] != 1:
        raise NotImplementedError("Independent Lychee speech RNG is currently implemented for one active request")
    audio_logits = logits[:, config.stoken_audio_token_id_min : speech_audio_token_id_max(config)]
    end_logits = logits[:, config.tts_end_token_id : config.tts_end_token_id + 1]
    candidate_logits = torch.cat((end_logits, audio_logits), dim=-1).float()
    if not bool(torch.isfinite(candidate_logits).all()):
        raise FloatingPointError("Non-finite Lychee speech logits; abort the native KV transaction")
    listening = modes == int(LycheeControlMode.LISTENING)
    delay = speaking_steps < config.stoken_delay_num
    start = speaking_steps == config.stoken_delay_num
    maximum = speaking_steps >= config.stoken_delay_num + config.stoken_max_tokens
    if bool((listening | delay | start | maximum).all()):
        # The forced prefix/pad path must not advance the request RNG.
        forced = torch.full_like(speaking_steps, config.stoken_pad_token_id)
        forced = torch.where(~listening & delay, forced.new_tensor(config.stoken_delay_token_id), forced)
        forced = torch.where(~listening & start, forced.new_tensor(config.tts_start_token_id), forced)
        return torch.where(~listening & maximum, forced.new_tensor(config.tts_end_token_id), forced)
    audio_ids = torch.arange(
        config.stoken_audio_token_id_min,
        speech_audio_token_id_max(config),
        dtype=torch.long,
        device=logits.device,
    )
    candidate_ids = torch.cat((audio_ids.new_tensor([config.tts_end_token_id]), audio_ids))
    if speech_history is not None and history_lengths is not None:
        candidate_logits = _apply_no_repeat_ngram(
            candidate_logits,
            candidate_ids=candidate_ids,
            history=speech_history,
            history_lengths=history_lengths,
            ngram_size=config.stoken_no_repeat_ngram_size,
        )

    effective_temperature = config.stoken_temperature if temperature is None else temperature
    do_sample = bool(config.stoken_do_sample) and effective_temperature > 0
    filtered_logits = candidate_logits
    if do_sample and effective_temperature != 1:
        filtered_logits = filtered_logits / effective_temperature
    if do_sample and config.stoken_top_k > 0:
        top_k = min(config.stoken_top_k, filtered_logits.shape[-1])
        threshold = torch.topk(filtered_logits, top_k, dim=-1).values[:, -1:]
        filtered_logits = filtered_logits.masked_fill(
            filtered_logits < threshold,
            float("-inf"),
        )
    if do_sample and 0 < config.stoken_top_p < 1:
        sorted_logits, sorted_indices = torch.sort(
            filtered_logits,
            dim=-1,
            descending=True,
        )
        cumulative = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_mask = cumulative > config.stoken_top_p
        sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
        sorted_mask[:, 0] = False
        remove_mask = torch.zeros_like(sorted_mask).scatter(
            1,
            sorted_indices,
            sorted_mask,
        )
        filtered_logits = filtered_logits.masked_fill(remove_mask, float("-inf"))

    if not bool(torch.isfinite(filtered_logits).any(dim=-1).all()):
        raise FloatingPointError("No legal Lychee speech candidate remains after filtering")
    if not do_sample:
        candidate_index = filtered_logits.argmax(dim=-1, keepdim=True)
    else:
        probabilities = torch.softmax(filtered_logits, dim=-1)
        if not bool(torch.isfinite(probabilities).all()) or not bool((probabilities.sum(dim=-1) > 0).all()):
            raise FloatingPointError("Invalid Lychee speech probability distribution; rebuild required")
        candidate_index = torch.multinomial(
            probabilities,
            num_samples=1,
            generator=generator,
        )
    sampled = candidate_ids[candidate_index[:, 0]]

    listening = modes == int(LycheeControlMode.LISTENING)
    delay = speaking_steps < config.stoken_delay_num
    start = speaking_steps == config.stoken_delay_num
    maximum = speaking_steps >= config.stoken_delay_num + config.stoken_max_tokens
    sampled = torch.where(listening, sampled.new_tensor(config.stoken_pad_token_id), sampled)
    sampled = torch.where(~listening & delay, sampled.new_tensor(config.stoken_delay_token_id), sampled)
    sampled = torch.where(~listening & start, sampled.new_tensor(config.tts_start_token_id), sampled)
    sampled = torch.where(~listening & maximum, sampled.new_tensor(config.tts_end_token_id), sampled)
    return sampled


def _apply_no_repeat_ngram(
    logits: torch.Tensor,
    *,
    candidate_ids: torch.Tensor,
    history: torch.Tensor,
    history_lengths: torch.Tensor,
    ngram_size: int,
) -> torch.Tensor:
    """Mask candidate tokens that would repeat an existing request-local n-gram."""

    if ngram_size <= 0 or history.shape[1] < ngram_size:
        return logits
    if history.shape[0] != logits.shape[0] or history_lengths.shape[0] != logits.shape[0]:
        raise ValueError("Speech history must have one row and length per logits row")

    prefix_size = ngram_size - 1
    safe_suffix_positions = (
        history_lengths[:, None] - prefix_size + torch.arange(prefix_size, device=history.device)[None, :]
    ).clamp_(0, history.shape[1] - 1)
    suffix = history.gather(1, safe_suffix_positions)
    windows = history.unfold(1, ngram_size, 1)
    starts = torch.arange(windows.shape[1], device=history.device)[None, :]
    valid_starts = starts <= (history_lengths - ngram_size)[:, None]
    prefix_matches = (windows[..., :prefix_size] == suffix[:, None, :]).all(dim=-1)
    matches = valid_starts & prefix_matches & (history_lengths >= prefix_size)[:, None]

    next_ids = windows[..., -1].contiguous()
    sorted_ids, candidate_order = torch.sort(candidate_ids)
    positions = torch.searchsorted(sorted_ids, next_ids)
    safe_positions = positions.clamp_max(sorted_ids.numel() - 1)
    legal_repeats = matches & (positions < sorted_ids.numel()) & (sorted_ids[safe_positions] == next_ids)
    candidate_positions = candidate_order[safe_positions]
    # Scatter only matched token IDs. A session-length by codec-vocabulary
    # broadcast would grow needlessly with every completed response.
    banned_counts = torch.zeros_like(logits, dtype=torch.int32)
    banned_counts.scatter_add_(1, candidate_positions, legal_repeats.to(torch.int32))
    return logits.masked_fill(banned_counts > 0, float("-inf"))


def update_speaking_steps(
    speaking_steps: torch.Tensor,
    *,
    old_modes: torch.Tensor,
    new_modes: torch.Tensor,
) -> torch.Tensor:
    """Advance the delay-prefix cursor after an active model tick."""

    listening = int(LycheeControlMode.LISTENING)
    was_voice = old_modes != listening
    is_voice = new_modes != listening
    continued = was_voice & is_voice & (old_modes == new_modes)
    entered = is_voice & ((~was_voice) | (old_modes != new_modes))
    updated = torch.where(continued, speaking_steps + 1, speaking_steps)
    updated = torch.where(entered, updated.new_zeros(()), updated)
    updated = torch.where(~is_voice, updated.new_full((), -1), updated)
    return updated


__all__ = [
    "LycheeControlMode",
    "sample_control_tokens",
    "sample_speech_tokens",
    "speech_audio_token_id_max",
    "update_control_modes",
    "update_speaking_steps",
]
