# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Keep delegation payloads out of the speech condition across duplex units."""

import torch


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
    token_ids: list[int], hidden: torch.Tensor, special_token_ids: dict[str, int], state: dict
) -> tuple[list[int], torch.Tensor]:
    open_id = special_token_ids.get("delegate_open_token_id", -1)
    close_id = special_token_ids.get("delegate_close_token_id", -1)
    if open_id < 0 or close_id < 0:
        return token_ids, hidden
    keep, muted = speech_condition_indices(
        token_ids,
        open_id=open_id,
        close_id=close_id,
        turn_end_id=special_token_ids.get("turn_eos_token_id"),
        muted=bool(state.get("delegate_muted", False)),
    )
    state["delegate_muted"] = muted
    if len(keep) == len(token_ids):
        return token_ids, hidden
    indices = torch.tensor(keep, dtype=torch.long, device=hidden.device)
    return [token_ids[i] for i in keep], hidden.index_select(0, indices)


def delegate_speech_boundary(
    token_ids: list[int], speech_ids: list[int], *, turn_end_id: int | None, state: dict
) -> tuple[bool, bool, bool]:
    """Return (flush, restart, skip) without closing the dialogue turn.

    A fully delegated unit drains the previous speech once. Later silent
    units and a bare turn terminator do not invoke synthesis; newly spoken
    content starts with fresh Talker and vocoder state.
    """
    preflushed = bool(state.get("delegate_preflushed", False))
    fully_muted = bool(token_ids) and not speech_ids
    bare_end = bool(speech_ids) and all(token == turn_end_id for token in speech_ids)
    flush = fully_muted and bool(state.get("delegate_speech_active", False))
    skip = fully_muted or (preflushed and bare_end)
    restart = preflushed and not skip and bool(speech_ids)
    if flush:
        state["delegate_preflushed"] = True
        state["delegate_speech_active"] = False
    elif restart:
        state["delegate_preflushed"] = False
    if speech_ids and not skip:
        state["delegate_speech_active"] = True
    return flush, restart, skip
