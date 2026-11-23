# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Keep delegation payloads out of the speech condition across duplex units."""

import torch


def speech_condition_indices(
    token_ids: list[int], *, open_id: int, close_id: int, turn_end_id: int | None, muted: bool
) -> tuple[list[int], bool]:
    """Return aligned token/hidden rows, retaining EOS to drain pending speech.

    The model's text stream is unchanged. An unterminated delegation span ends
    at the assistant turn boundary so it cannot mute the following turn.
    """
    keep = []
    for index, token in enumerate(token_ids):
        if token == open_id:
            muted = True
        elif token == close_id:
            muted = False
        elif token == turn_end_id:
            keep.append(index)
            muted = False
        elif not muted:
            keep.append(index)
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
