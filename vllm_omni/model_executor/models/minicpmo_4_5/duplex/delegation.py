# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Keep delegation payloads out of the speech condition across duplex units."""

from collections.abc import Callable, Mapping

import torch


def decode_with_delegate_tags(
    token_ids: list[int], decode: Callable[[list[int]], str], special_token_ids: Mapping[str, int]
) -> str:
    """Retain delegation delimiters while the ordinary decoder skips controls.

    These are tokenizer special tokens, so ``skip_special_tokens=True`` alone
    silently destroys the tool protocol before it reaches the session plugin.
    """
    tags = {
        special_token_ids.get("delegate_open_token_id", -1): "<delegate>",
        special_token_ids.get("delegate_close_token_id", -1): "</delegate>",
    }
    tags.pop(-1, None)
    pieces = []
    start = 0
    for index, token_id in enumerate(token_ids):
        if token_id in tags:
            if index > start:
                pieces.append(decode(token_ids[start:index]))
            pieces.append(tags[token_id])
            start = index + 1
    if start < len(token_ids):
        pieces.append(decode(token_ids[start:]))
    return "".join(pieces)


def speech_condition_indices(
    token_ids: list[int], *, open_id: int, close_id: int, turn_end_id: int | None, muted: bool
) -> tuple[list[int], bool]:
    """Return the speech rows selected by the released Venus duplex model.

    The model's text stream is unchanged. An unterminated delegation span ends
    at the assistant turn boundary so it cannot mute the following turn.
    """
    keep = []
    for index, token in enumerate(token_ids):
        if token == open_id:
            muted = True
        elif token == close_id:
            muted = False
        elif not muted:
            keep.append(index)
    # EOS remains in the text/control stream even when delegation masks its
    # speech row. Reset after the unit, matching the reference implementation.
    if turn_end_id in token_ids:
        muted = False
    return keep, muted


def filter_delegate_condition(
    token_ids: list[int],
    hidden: torch.Tensor,
    special_token_ids: dict[str, int],
    state: dict,
    *,
    condition_start: int = 0,
) -> tuple[list[int], torch.Tensor]:
    # A sampled opening/closing decision still changes delegation state even
    # though MiniCPM excludes that decision's row from speech conditioning.
    condition_ids = token_ids[condition_start:] if condition_start else token_ids
    open_id = special_token_ids.get("delegate_open_token_id", -1)
    close_id = special_token_ids.get("delegate_close_token_id", -1)
    if open_id < 0 or close_id < 0:
        return condition_ids, hidden
    keep, muted = speech_condition_indices(
        token_ids,
        open_id=open_id,
        close_id=close_id,
        turn_end_id=special_token_ids.get("turn_eos_token_id"),
        muted=bool(state.get("delegate_muted", False)),
    )
    state["delegate_muted"] = muted
    if condition_start:
        keep = [index - condition_start for index in keep if index >= condition_start]
    if len(keep) == len(condition_ids):
        return condition_ids, hidden
    indices = torch.tensor(keep, dtype=torch.long, device=hidden.device)
    return [condition_ids[i] for i in keep], hidden.index_select(0, indices)


def delegate_speech_boundary(
    token_ids: list[int], speech_ids: list[int], *, turn_end_id: int | None, state: dict
) -> tuple[bool, bool, bool]:
    """Return (flush, restart, skip) without closing the dialogue turn.

    A fully delegated unit drains the previous speech once. Later silent
    units and a bare turn terminator do not invoke synthesis; newly spoken
    content starts with fresh Talker and vocoder state.
    """
    preflushed = bool(state.get("delegate_preflushed", False))
    speech_active = bool(state.get("delegate_speech_active", False))
    fully_muted = bool(token_ids) and not speech_ids
    bare_end = bool(speech_ids) and all(token == turn_end_id for token in speech_ids)
    flush = fully_muted and speech_active
    skip = fully_muted or (preflushed and bare_end)
    # Initial delegated units have no speech to flush, but their first audible
    # unit still needs a fresh prompt and the initial-chunk EOS budget.
    restart = not speech_active and not skip and bool(speech_ids)
    if flush:
        state["delegate_preflushed"] = True
        state["delegate_speech_active"] = False
    elif restart:
        state["delegate_preflushed"] = False
    if speech_ids and not skip:
        state["delegate_speech_active"] = True
    return flush, restart, skip
