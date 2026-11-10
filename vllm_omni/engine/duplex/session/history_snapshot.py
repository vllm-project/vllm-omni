# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Versioned, detached views of authoritative duplex conversation history."""

from dataclasses import dataclass


def _copy_containers(value: object) -> object:
    # History contains protocol dictionaries/lists and immutable text/media
    # references. Copy their containers without cloning opaque media payloads
    # or invoking arbitrary objects' deepcopy hooks on the session loop.
    if isinstance(value, dict):
        return {key: _copy_containers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_containers(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_containers(item) for item in value)
    return value


def copy_history_message(message: dict[str, object]) -> dict[str, object]:
    return {key: _copy_containers(value) for key, value in message.items()}


@dataclass(frozen=True)
class DuplexHistorySnapshot:
    """Aligned columns captured synchronously on the session owner's loop.

    The record is frozen. Message containers are detached, mutable copies for
    existing plugin consumers; editing them cannot change session authority.
    Source IDs are meaningful only within the live session that issued them.
    """

    messages: tuple[dict[str, object], ...]
    item_ids: tuple[str | None, ...]
    source_ids: tuple[int, ...]
    source_revisions: tuple[int, ...]
    history_revision: int
