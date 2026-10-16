# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Full request snapshots with a fast path for immutable scalar leaves."""

import copy
from typing import Any

_ATOMIC_TYPES = frozenset((type(None), bool, int, float, complex, str, bytes))


def copy_request_snapshot(
    value: Any,
    memo: dict[int, Any] | None = None,
    *,
    shared_types: tuple[type, ...] = (),
) -> Any:
    """Deep-copy all fields, preserving aliases and isolating mutable containers.

    Plain numeric/text trees do not need generic object dispatch or an identity
    lookup for each immutable scalar. Subclasses and other objects retain their
    regular deepcopy contracts. No model fields are filtered here; instances of
    ``shared_types`` are returned as-is (containers holding them are still
    copied), for callers that read them once right away.
    """
    kind = type(value)
    if kind in _ATOMIC_TYPES:
        return value
    if shared_types and isinstance(value, shared_types):
        return value
    if memo is None:
        memo = {}
    key = id(value)
    if key in memo:
        return memo[key]
    if kind is list:
        atomic = all(type(item) in _ATOMIC_TYPES for item in value)
        result = value.copy() if atomic else []
        memo[key] = result
        if not atomic:
            result.extend(copy_request_snapshot(item, memo, shared_types=shared_types) for item in value)
    elif kind is dict:
        result = {}
        memo[key] = result
        for k, item in value.items():
            result[copy_request_snapshot(k, memo)] = copy_request_snapshot(item, memo, shared_types=shared_types)
    else:
        return copy.deepcopy(value, memo)
    # Keep sources alive for the duration of a copy, including custom deepcopy
    # implementations that mutate a source container while being visited.
    memo.setdefault(id(memo), []).append(value)
    return result
