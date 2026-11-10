# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Opt-in per-stage history views derived from authoritative duplex history."""

from collections.abc import Mapping
from dataclasses import dataclass

from vllm_omni.engine.duplex.config import DuplexContextViewPolicy
from vllm_omni.engine.duplex.session.context_ledger import DuplexContextLedger
from vllm_omni.engine.duplex.session.history_snapshot import DuplexHistorySnapshot, copy_history_message

MEDIA_PART_TYPES: frozenset[str] = frozenset({"audio_url", "image_url"})


@dataclass(frozen=True)
class DuplexHistoryView:
    """Rendered messages aligned 1:1 with a history snapshot.

    ``node_ids[i]`` is the L1 node rendered at position ``i``, or ``None``
    where the snapshot message is passed through verbatim.

    This candidate context view can drop unanswered input's media when the
    policy keeps zero user runs. Consumers must preserve current request input
    from authoritative ``conversation`` and bind media payloads against it.
    """

    messages: tuple[dict[str, object], ...]
    source_ids: tuple[int, ...]
    node_ids: tuple[int | None, ...]


def render_history_view(
    snapshot: DuplexHistorySnapshot,
    ledger: DuplexContextLedger,
    policy: DuplexContextViewPolicy,
) -> DuplexHistoryView:
    ledger.refresh(snapshot)
    messages = snapshot.messages
    rendered: list[dict[str, object]] = list(messages)
    node_ids: list[int | None] = [None] * len(messages)
    pending: dict[int, dict[str, object]] = {}
    tail_start = _verbatim_tail_start(messages, policy.keep_media_user_turns)
    for run in _user_runs(messages[:tail_start]):
        derived: dict[int, Mapping[str, object]] = {}
        for index in run:
            node = ledger.l1_node_for(snapshot.source_ids[index])
            message = node.message if node is not None else _drop_media(messages[index])
            if message is not None:
                derived[index] = message
        # Dropping media must not erase a whole user turn from the view.
        if not derived or not any(_has_content(derived.get(index, messages[index])) for index in run):
            continue
        for index, message in derived.items():
            source_id = snapshot.source_ids[index]
            node = ledger.l1_node_for(source_id)
            if node is None:
                pending[index] = copy_history_message(dict(message))
                continue
            rendered[index] = copy_history_message(dict(node.message))
            node_ids[index] = node.node_id
    new_nodes = ledger.add_l1_nodes(
        snapshot, messages={snapshot.source_ids[index]: message for index, message in pending.items()}
    )
    for index, node in zip(pending, new_nodes, strict=True):
        rendered[index] = copy_history_message(dict(node.message))
        node_ids[index] = node.node_id
    return DuplexHistoryView(messages=tuple(rendered), source_ids=snapshot.source_ids, node_ids=tuple(node_ids))


def _drop_media(message: Mapping[str, object]) -> dict[str, object] | None:
    """Return the L1 rendering of a user message, or None if it is not eligible."""
    content = message.get("content")
    if message.get("role") != "user" or not isinstance(content, list):
        return None
    media = [part for part in content if _is_media(part)]
    if not media:
        return None
    kept = [part for part in content if not _is_media(part)]
    transcript = next(
        (part["transcript"] for part in media if isinstance(part.get("transcript"), str) and part["transcript"]),
        message.get("transcript"),
    )
    if isinstance(transcript, str) and transcript and not any(_text_of(part) == transcript for part in kept):
        kept.append({"type": "text", "text": transcript})
    return {key: kept if key == "content" else value for key, value in message.items()}


def _is_media(part: object) -> bool:
    return isinstance(part, Mapping) and part.get("type") in MEDIA_PART_TYPES


def _text_of(part: object) -> object:
    return part.get("text") if isinstance(part, Mapping) and part.get("type") == "text" else None


def _has_content(message: Mapping[str, object]) -> bool:
    content = message.get("content")
    return isinstance(content, str | list) and len(content) > 0


def _verbatim_tail_start(messages: tuple[dict[str, object], ...], keep_user_runs: int) -> int:
    if keep_user_runs == 0:
        return len(messages)
    runs = _user_runs(messages)
    return runs[-keep_user_runs][0] if len(runs) >= keep_user_runs else 0


def _user_runs(messages: tuple[dict[str, object], ...]) -> list[list[int]]:
    runs: list[list[int]] = []
    previous_is_user = False
    for index, message in enumerate(messages):
        is_user = message.get("role") == "user"
        if is_user and previous_is_user:
            runs[-1].append(index)
        elif is_user:
            runs.append([index])
        previous_is_user = is_user
    return runs
