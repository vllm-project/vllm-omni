# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""JoyAI's ``delegate`` action as one shared tool call, and its result context.

JoyAI writes a delegation as an action token, not as JSON::

    </response> Checking the pot. </delegation> Is the pot overheated?

The runtime never executes that question. It publishes it as a function call
(``call_id`` + ``name`` + ``arguments``) and the session ledger admits the
client's ``function_call_output`` exactly once. This module owns the model side
of that boundary only: the tool declaration, the action -> call conversion and
the result -> prompt context conversion. Call identity, deadlines, duplicate and
late results stay in ``vllm_omni.engine.duplex.session.delegation``'s owner,
``...session.tool_calls``.

A result is *not* a user turn. It never enters the conversation history; it
becomes a ``[Delegated Results]`` block addressed to the question it answers.
JoyAI's own rule is that a spoken reply answers the goal of the current chunk,
and the next reaction to that goal reuses the same evidence. The block follows
that rule: a result is carried by every observation until a reply that stores
text displays it, and it is reused once more if the same chunk speaks again.
A result that no open chunk can carry is marked expired and is not shown again.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass

from vllm_omni.experimental.fullduplex.joyvl.decision.output_parser import ParsedAction

#: The one function JoyAI declares.
DELEGATION_TOOL_NAME = "delegate"
#: Longer answers are cut: the foreground agent relays one short digest.
DELEGATION_RESULT_LIMIT = 2000
#: Accepted results kept per session; the oldest is evicted first.
MAX_DELEGATION_RESULTS = 8
#: Character budget of the rendered result block.
DELEGATION_RESULT_WINDOW_CHARS = 4000
#: An oversized delegation is refused rather than published as a huge call.
MAX_DELEGATION_ARGUMENT_BYTES = 64 * 1024
#: Results already marked expired are pruned once the list grows past this.
EXPIRED_RESULT_RETENTION = 32
#: How many stored replies may speak one result: the reply that answers it, plus
#: one reuse inside that same chunk (JoyAI keeps speaking for the chunk's goal).
MAX_RESULT_REUSES = 2

_DELEGATION_ACTION = (
    "**Delegate** — when a question is too hard or error-prone to answer reliably yourself, "
    "speak a brief note that you're delegating, then hand the question to the background solver:\n"
    "</response> Brief note that you're delegating. </delegation> <the question>\n"
    "The solver's answer arrives later as a delegated result for that question. "
    "Answer it as you would any other question."
)

_DELEGATION_PARAMETERS: dict[str, object] = {
    "type": "object",
    "properties": {
        "question": {"type": "string", "description": "The question the background helper should answer."},
        "note": {"type": "string", "description": "The short note already spoken to the user."},
    },
    "required": ["question"],
    "additionalProperties": False,
}


def delegation_tool_definition() -> dict[str, object]:
    """The Realtime ``tools`` entry that enables JoyAI delegation."""
    return {
        "type": "function",
        "name": DELEGATION_TOOL_NAME,
        "description": (
            "Ask a background helper to answer one question the current view cannot answer reliably. "
            "The answer comes back as context for a later observation."
        ),
        "parameters": json.loads(json.dumps(_DELEGATION_PARAMETERS)),
    }


def delegation_system_section() -> str:
    """The action description appended to the system prompt when enabled."""
    return _DELEGATION_ACTION


def delegation_arguments(action: ParsedAction) -> str | None:
    """Render one delegated action as function-call arguments.

    ``None`` means the action is not delegatable: a delegation without a
    question has nothing to hand over.
    """
    question = " ".join((action.delegated_question or "").split())
    if not question:
        return None
    note = " ".join((action.text or "").split())
    arguments: dict[str, str] = {"question": question}
    if note:
        arguments["note"] = note
    encoded = json.dumps(arguments, ensure_ascii=False, sort_keys=True)
    if len(encoded.encode("utf-8")) > MAX_DELEGATION_ARGUMENT_BYTES:
        return None
    return encoded


def _decode_arguments(arguments: object) -> tuple[str, str] | None:
    if not isinstance(arguments, str) or not arguments:
        return None
    try:
        decoded = json.loads(arguments)
    except ValueError:
        return None
    if not isinstance(decoded, dict):
        return None
    question = decoded.get("question")
    note = decoded.get("note")
    if not isinstance(question, str) or not question.strip():
        return None
    if note is not None and not isinstance(note, str):
        return None
    return question.strip(), (note or "").strip()


@dataclass(frozen=True, slots=True)
class DelegationResult:
    """One accepted tool result, as context for a later JoyAI prompt."""

    call_id: str
    question: str
    output: str
    note: str = ""

    @classmethod
    def from_call(cls, call_id: str, arguments: object, output: object) -> DelegationResult:
        """Build a result from the session's immutable call origin."""
        if not isinstance(output, str):
            raise ValueError("a delegated result requires a string output")
        decoded = _decode_arguments(arguments)
        if decoded is None:
            raise ValueError("the delegated call does not carry a question")
        question, note = decoded
        return cls(call_id=call_id, question=question, output=output[:DELEGATION_RESULT_LIMIT], note=note)


def result_records(runtime_config: Mapping[str, object]) -> list[dict[str, object]]:
    """Accepted results, oldest first, as stored in the session runtime."""
    raw = runtime_config.get("joyai_delegation_results")
    if not isinstance(raw, list):
        return []
    return [dict(item) for item in raw if isinstance(item, Mapping)]


def _result_from_record(raw: Mapping[str, object]) -> DelegationResult | None:
    call_id, question = raw.get("call_id"), raw.get("question")
    output = raw.get("output")
    if not isinstance(call_id, str) or not call_id:
        return None
    if not isinstance(question, str) or not question or not isinstance(output, str):
        return None
    note = raw.get("note")
    return DelegationResult(
        call_id=call_id,
        question=question,
        output=output,
        note=note if isinstance(note, str) else "",
    )


def unanswered_results(runtime_config: Mapping[str, object], *, chunk: object = None) -> list[DelegationResult]:
    """Results the prompt prepared now should carry.

    A result is shown while it is unanswered, and once more inside the chunk
    that answered it (JoyAI keeps speaking for that chunk's goal). A result
    answered in a *different* chunk has left that goal behind and is not shown
    again; neither is one that was answered twice or expired with its turn.
    """
    results: list[DelegationResult] = []
    for record in result_records(runtime_config):
        if record.get("expired"):
            continue
        answered = record.get("answered")
        if isinstance(answered, int) and answered > 0:
            # The reuse belongs to the chunk that answered it, not to whatever
            # chunk is current now.
            if answered >= MAX_RESULT_REUSES:
                continue
            answered_in = record.get("answered_in_chunk")
            if not (isinstance(answered_in, str) and isinstance(chunk, str) and answered_in == chunk):
                continue
        displayed = record.get("displayed_in_chunk")
        if isinstance(displayed, str) and isinstance(chunk, str) and displayed == chunk:
            continue
        result = _result_from_record(record)
        if result is not None:
            results.append(result)
    return results


def accept_result(
    runtime_config: Mapping[str, object],
    item: Mapping[str, object],
    *,
    accepted_at: int,
) -> dict[str, object]:
    """Return the runtime config with one more accepted result.

    Raises ``ValueError`` when the session-owned origin does not describe a
    delegation this session can answer.
    """
    call_id = item.get("call_id")
    if not isinstance(call_id, str) or not call_id:
        raise ValueError("function_call_output requires call_id")
    result = DelegationResult.from_call(call_id, item.get("arguments"), item.get("output"))
    records = [record for record in result_records(runtime_config) if record.get("call_id") != call_id]
    records.append(
        {
            "call_id": result.call_id,
            "question": result.question,
            "note": result.note,
            "output": result.output,
            "accepted_at": accepted_at,
        }
    )
    if len(records) > MAX_DELEGATION_RESULTS:
        expired = [record for record in records if record.get("expired")]
        live = [record for record in records if not record.get("expired")]
        expired_keep = max(0, MAX_DELEGATION_RESULTS - len(live))
        records = [*(expired[-expired_keep:] if expired_keep else []), *live[-MAX_DELEGATION_RESULTS:]]
        records = records[-MAX_DELEGATION_RESULTS:]
    updated = dict(runtime_config)
    updated["joyai_delegation_results"] = records
    return updated


def mark_displayed(
    runtime_config: Mapping[str, object],
    *,
    call_ids: list[str],
    chunk: object,
) -> dict[str, object] | None:
    """Return the runtime patch that shows these results in ``chunk``.

    A result keeps one reuse inside the chunk that answered it. Once that reuse
    is spent (``answered >= 2``), the result stays listed for that chunk but is
    no longer shown by ``unanswered_results``.
    """
    if not call_ids or not isinstance(chunk, str) or not chunk:
        return None
    records = result_records(runtime_config)
    changed = False
    for record in records:
        if record.get("call_id") in call_ids and record.get("displayed_in_chunk") != chunk:
            record["displayed_in_chunk"] = chunk
            changed = True
    if not changed:
        return None
    return _with_results(runtime_config, records)


def prune_expired(runtime_config: Mapping[str, object]) -> dict[str, object] | None:
    """Forget expired tombstones once the list has grown past a small bound."""
    records = result_records(runtime_config)
    expired = [record for record in records if record.get("expired")]
    if len(expired) <= EXPIRED_RESULT_RETENTION:
        return None
    live = [record for record in records if not record.get("expired")]
    return _with_results(runtime_config, [*expired[-EXPIRED_RESULT_RETENTION:], *live])


def _with_results(
    runtime_config: Mapping[str, object],
    records: list[dict[str, object]],
) -> dict[str, object]:
    """Return the complete runtime with the result list replaced.

    Every hook's return value is installed as the whole runtime, not merged into
    it. A hook that returned only ``{"joyai_delegation_results": ...}`` would
    drop ``tts_speaker`` and the rest of the session configuration, and the next
    prompt would fail on the first missing key.
    """
    updated = dict(runtime_config)
    if records:
        updated["joyai_delegation_results"] = records
    else:
        updated.pop("joyai_delegation_results", None)
    return updated


def settle_response(
    runtime_config: Mapping[str, object],
    *,
    committed: bool,
    chunk: object = None,
    expired: list[str] | None = None,
) -> dict[str, object] | None:
    """Apply one finished response to the results its prompt displayed.

    A stored reply answers every result displayed in it. A reply that stored
    nothing (a repeat, a silent decision, a failure) leaves the results
    unanswered, and the next frame of the open chunk shows them again.
    ``expired`` names results no open chunk can carry anymore.
    """
    records = result_records(runtime_config)
    if not records:
        return None
    expired_ids = set(expired or ())
    remaining: list[dict[str, object]] = []
    changed = False
    for record in records:
        updated = dict(record)
        displayed = "displayed_in_chunk" in record
        if committed and displayed:
            answered = record.get("answered", 0)
            updated["answered"] = answered + 1 if isinstance(answered, int) else 1
            # Remember which chunk answered it: the one reuse this chunk may
            # still make is scoped to that chunk, not to a later one.
            if isinstance(chunk, str) and chunk:
                updated["answered_in_chunk"] = chunk
            updated.pop("displayed_in_chunk", None)
            changed = True
        elif not committed and displayed:
            # Not answered: the next frame of this chunk may show it again.
            updated.pop("displayed_in_chunk", None)
            changed = True
        if updated.get("call_id") in expired_ids:
            updated["expired"] = True
            updated.pop("displayed_in_chunk", None)
            updated.pop("answered_in_chunk", None)
            changed = True
        remaining.append(updated)
    if not changed:
        return None
    return _with_results(runtime_config, remaining)


def expired_call_ids(runtime_config: Mapping[str, object]) -> set[str]:
    """Results the session can no longer deliver."""
    return {record["call_id"] for record in result_records(runtime_config) if record.get("expired")}


def expire_unanswered(runtime_config: Mapping[str, object]) -> dict[str, object] | None:
    """Mark every result the cancelled turn never answered as expired.

    A barge-in cancels the turn under a pending result. Nothing will carry that
    result into a prompt any more, so it must not surface later as if the
    question had been asked in the new context.
    """
    records = result_records(runtime_config)
    changed = False
    remaining: list[dict[str, object]] = []
    for record in records:
        updated = dict(record)
        if not updated.get("answered") and not updated.get("expired"):
            updated["expired"] = True
            updated.pop("displayed_in_chunk", None)
            updated.pop("answered_in_chunk", None)
            changed = True
        remaining.append(updated)
    if not changed:
        return None
    return _with_results(runtime_config, remaining)


def reconcile_delegation_runtime(
    current: Mapping[str, object],
    *,
    displayed: list[str],
    chunk: object,
) -> list[dict[str, object]] | None:
    """Merge a planned prompt's bookkeeping into the live runtime.

    The prompt was planned on a snapshot. While it was planned the session may
    have accepted another result or settled a response. Reconcile by call id:
    keep the live list (so nothing accepted meanwhile is lost) and apply only
    the display mark this plan owns. ``answered`` / ``expired`` belong to the
    live record; copying a plan-time snapshot back would undo a settle that ran
    while the prompt was being planned.
    """
    live = result_records(current)
    if not live:
        return None
    merged: list[dict[str, object]] = []
    for record in live:
        result = dict(record)
        if record.get("call_id") in displayed and isinstance(chunk, str) and chunk:
            result["displayed_in_chunk"] = chunk
        merged.append(result)
    return merged


def expired_results(runtime_config: Mapping[str, object], *, chunk: object) -> list[str]:
    """Results a response no longer displays inside any open chunk.

    A result is carried while its observation can still answer it. Once a
    response of a chunk both displayed and skipped it, the next chunk cannot
    answer it either, because the conversation it belongs to has moved on.
    """
    expired: list[str] = []
    for record in result_records(runtime_config):
        if record.get("expired"):
            continue
        answered = record.get("answered")
        displayed = record.get("displayed_in_chunk")
        if isinstance(answered, int) and answered > 0 and not isinstance(displayed, str):
            expired.append(record["call_id"])
    return expired


def render_result_blocks(
    results: list[DelegationResult],
    *,
    window_chars: int = DELEGATION_RESULT_WINDOW_CHARS,
) -> tuple[str, list[str]]:
    """Render results as one prompt block, newest kept whole.

    Returns the block text and the call ids it actually contains. Results that
    do not fit the window are not in the text, so a caller must not book them as
    displayed: they stay available for a later prompt.
    """
    blocks: list[str] = []
    rendered: list[str] = []
    remaining = max(0, window_chars)
    for result in reversed(results):
        block = f"Question: {result.question}\nAnswer: {result.output}"
        if len(block) > remaining:
            if remaining <= 0:
                break
            block = block[:remaining].rstrip() + " ..."
        blocks.append(block)
        rendered.append(result.call_id)
        remaining -= len(block)
        if remaining <= 0:
            break
    if not blocks:
        return "", []
    return "[Delegated Results]\n" + "\n\n".join(reversed(blocks)), list(reversed(rendered))
