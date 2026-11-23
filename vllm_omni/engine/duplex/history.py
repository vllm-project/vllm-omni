# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Turn boundaries shared by duplex history consumers."""

from collections.abc import Mapping, Sequence


def unanswered_turn_start(history: Sequence[Mapping[str, object]], *, lower_bound: int = 0) -> int:
    """Return the first pending user item, or len(history) if none.

    System updates leave user input pending; an assistant reply ends that run.
    A caller can bound the search at a tool exchange stored outside history.
    """
    start = len(history)
    for index in range(len(history) - 1, max(0, lower_bound) - 1, -1):
        role = history[index].get("role")
        if role == "assistant":
            break
        if role == "user":
            start = index
    return start
