# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request-local Qwen Talker codec replay on runner-owned state."""

from typing import Any

import torch

from vllm_omni.core.prefix_cache.adapter import PrefixCacheRequestOwner


def bind_talker_history(state: Any, owner: PrefixCacheRequestOwner | None) -> None:
    """Start/replacement admission resets request-local generated decisions."""
    state.talker_codec_inputs = []
    state.talker_codec_owner = owner
    state.talker_next_input_position = None
    state.talker_next_input_embedding = None


def _codec_history(state: Any) -> list[torch.Tensor]:
    history = getattr(state, "talker_codec_inputs", None)
    if isinstance(history, list):
        return history
    raise ValueError("Talker replay requires retained runner-owned codec history")


def validate_local_talker_resume(state: Any, owner: PrefixCacheRequestOwner | None, *, num_output_tokens: int) -> None:
    """Same-worker resume must retain local codec decisions, never wire history."""
    if not isinstance(owner, PrefixCacheRequestOwner) or getattr(state, "talker_codec_owner", None) != owner:
        raise ValueError("Talker resume requires matching local scheduler owner")
    if type(num_output_tokens) is not int or num_output_tokens < 0:
        raise ValueError("Talker resume requires a valid accepted output frontier")
    accepted = _accepted_output_tokens(state)
    if num_output_tokens != len(accepted):
        raise ValueError("Talker resume detected missing or rolled-back accepted primaries")
    history_len = len(_codec_history(state))
    if history_len > num_output_tokens or history_len < max(0, num_output_tokens - 1):
        raise ValueError("Talker resume detected missing or rolled-back local codec history")


def _accepted_output_tokens(state: Any) -> list[int]:
    tokens = getattr(state, "output_token_ids", None)
    if not isinstance(tokens, list):
        raise ValueError("Talker request-local replay requires runner output token history")
    return tokens


def _hidden_last(payload: dict[str, Any], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    hidden = payload.get("hidden_states", {}).get("last") if isinstance(payload.get("hidden_states"), dict) else None
    if not isinstance(hidden, torch.Tensor) or hidden.ndim != 1:
        raise ValueError("Talker pending codec input requires retained hidden_states.last")
    return hidden.to(device=device, dtype=dtype).reshape(1, 1, -1)


def _validate_history_frontier(state: Any, *, spec_groups: int) -> list[torch.Tensor]:
    history = _codec_history(state)
    tokens = _accepted_output_tokens(state)
    if len(history) > len(tokens):
        raise ValueError("Talker local codec history is ahead of accepted primaries")
    if len(history) < max(0, len(tokens) - 1):
        raise ValueError("Talker local codec history is missing more than one pending primary")
    if history:
        row = history[-1]
        if not isinstance(row, torch.Tensor) or row.ndim != 1 or row.dtype != torch.long or row.numel() != spec_groups:
            raise ValueError("Talker local codec history row shape does not match the model")
    return history


def build_request_end_talker_codes(
    model: Any,
    state: Any,
    *,
    owner: PrefixCacheRequestOwner | None,
) -> torch.Tensor:
    """Build the request-end full-payload codec snapshot from local history.

    Partial recompute can materialize the latest accepted primary before its
    recompute sample is discarded, so the valid final/cleanup frontier is
    N-1 or N local rows for N accepted primaries. The payload consumed by
    Qwen code2wav remains aligned as prompt-zero rows plus only N-1 generated
    codec decisions.
    """
    local_owner = getattr(state, "talker_codec_owner", None)
    if not isinstance(local_owner, PrefixCacheRequestOwner) or owner != local_owner:
        raise ValueError("Talker request-end snapshot requires matching local scheduler owner")
    spec_groups = int(model.talker.num_code_groups)
    history = _validate_history_frontier(state, spec_groups=spec_groups)
    accepted = _accepted_output_tokens(state)
    prompt_len = len(state.prompt_token_ids)
    generated_rows = max(0, len(accepted) - 1)
    rows = history[:generated_rows]
    device = rows[0].device if rows else torch.device("cpu")
    prompt = torch.zeros((prompt_len, spec_groups), dtype=torch.long, device=device)
    if not rows:
        return prompt
    return torch.cat((prompt, torch.stack(rows, dim=0)), dim=0)


def collect_pending_talker_primary(
    model: Any,
    state: Any,
    *,
    row_start: int,
    input_ids: torch.Tensor,
    payload: dict[str, Any],
    input_device: torch.device,
    input_dtype: torch.dtype,
) -> tuple[int, torch.Tensor, torch.Tensor] | None:
    """Validate and collect one pending accepted primary before recompute.

    Returns ``(pending_position, primary, hidden)`` or ``None`` when the
    runner-local codec history already covers all accepted primaries. This
    function intentionally does not run MTP; the runner batches all returned
    rows for the step.
    """
    prompt_length = len(state.prompt_token_ids)
    owner = getattr(state, "talker_codec_owner", None)
    if not isinstance(owner, PrefixCacheRequestOwner):
        raise ValueError("Talker request-local replay requires a scheduler owner")
    accepted = _accepted_output_tokens(state)
    spec_groups = int(model.talker.num_code_groups)
    history = _validate_history_frontier(state, spec_groups=spec_groups)
    if len(history) == len(accepted):
        return None
    pending_index = len(history)
    pending_position = prompt_length + pending_index
    token = accepted[pending_index]
    if type(token) is not int or not 0 <= token < model.talker_config.text_config.vocab_size:
        raise ValueError("Talker accepted primary is outside its codec vocabulary")
    sampling_params = state.sampling_params
    if sampling_params is None:
        raise ValueError("Talker pending codec input requires sampling params")
    # Sync update_from_output checks max_tokens unconditionally before the next
    # schedule; ignore_eos/min_tokens only affect EOS/stop checks, not this cap.
    if len(accepted) >= sampling_params.max_tokens:
        raise ValueError("Talker pending codec input exceeds max_tokens")
    current_offset = pending_position - row_start
    if 0 <= current_offset < input_ids.shape[0]:
        primary = input_ids[current_offset : current_offset + 1].reshape(1, 1)
    else:
        primary = torch.tensor([[token]], dtype=torch.long).to(input_device, non_blocking=True)
    hidden = _hidden_last(payload, device=input_device, dtype=input_dtype)
    return pending_position, primary, hidden


def install_pending_talker_primary(
    model: Any,
    state: Any,
    *,
    pending_position: int,
    codes: torch.Tensor,
    embedding: torch.Tensor,
) -> None:
    """Install one owned row from a step-local batched MTP invocation."""
    spec_groups = int(model.talker.num_code_groups)
    codes = codes.reshape(-1)
    embedding = embedding.reshape(1, -1)
    if codes.shape != (spec_groups,):
        raise ValueError("Talker pending codec producer returned an invalid code width")
    if codes.dtype != torch.long:
        raise ValueError("Talker pending codec producer returned an invalid code tensor")
    _codec_history(state).append(codes)
    state.talker_next_input_position = pending_position
    state.talker_next_input_embedding = embedding


def preprocess_talker_history(
    model: Any,
    state: Any,
    *,
    row_start: int,
    input_ids: torch.Tensor,
    input_embeds: torch.Tensor,
    payload: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Prepare prompt/replay rows from already materialized local history."""
    owner = getattr(state, "talker_codec_owner", None)
    if not isinstance(owner, PrefixCacheRequestOwner):
        raise ValueError("Talker replay requires a scheduler-owned runner admission")
    if input_embeds is None:
        raise ValueError("Talker replay requires model input embeddings")
    prompt_length = len(state.prompt_token_ids)
    history = _validate_history_frontier(
        state,
        spec_groups=int(model.talker.num_code_groups),
    )
    if len(history) < len(_accepted_output_tokens(state)):
        raise ValueError("Talker pending codec input was not materialized before replay")
    meta = payload.get("meta", {})
    contiguous = (
        row_start >= prompt_length and input_ids.shape[0] == 1 and state.talker_next_input_position == row_start
    )
    steady = contiguous and bool(meta.get("decode_flag"))
    first_decode_handoff = (
        contiguous
        and row_start == prompt_length
        and not meta.get("decode_flag")
        and meta.get("prefill_consumed_text_tokens") == 1
        and meta.get("num_processed_tokens") == prompt_length
    )
    ids, embeds, update = model.talker_replay_inputs(
        input_ids,
        input_embeds,
        row_start=row_start,
        prompt_length=prompt_length,
        owner=owner,
        codec_inputs=history,
        payload=payload,
        restore_text=not (steady or first_decode_handoff),
        codec_embeddings=getattr(state, "talker_next_input_embedding", None) if contiguous else None,
    )
    state.talker_next_input_embedding = None
    state.talker_next_input_position = row_start + input_ids.shape[0]
    return ids, embeds, update
