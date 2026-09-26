# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import enum
from dataclasses import dataclass

SILENCE_TOKEN = "</silence>"
RESPONSE_TOKEN = "</response>"
DELEGATION_TAG = "</delegation>"
_DELEGATION_MARKERS = ("</delegation>", "<delegation>")


class Action(enum.Enum):
    SILENCE = "silence"
    RESPONSE = "response"
    DELEGATE = "delegate"


@dataclass
class ParsedAction:
    action: Action
    text: str = ""
    delegated_question: str | None = None
    raw: str = ""

    @property
    def spoke(self) -> bool:
        return self.action is not Action.SILENCE


def parse_action(raw: str) -> ParsedAction:
    text = (raw or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return ParsedAction(Action.SILENCE, raw=raw or "")

    marker = _first_marker(text)
    if marker != -1 and text[marker:].startswith(SILENCE_TOKEN):
        return ParsedAction(Action.SILENCE, raw=raw)

    if marker == -1:
        payload = _collapse(text.splitlines()[0])
    else:
        after = text[marker + len(RESPONSE_TOKEN) :].strip()
        payload = _collapse(after.splitlines()[0]) if after else ""

    for tag in _DELEGATION_MARKERS:
        idx = payload.find(tag)
        if idx != -1:
            return ParsedAction(
                Action.DELEGATE,
                text=_collapse(payload[:idx]),
                delegated_question=_collapse(payload[idx + len(tag) :]) or None,
                raw=raw,
            )

    return ParsedAction(Action.RESPONSE, text=payload, raw=raw)


def to_token_form(action: ParsedAction) -> str:
    if action.action is Action.SILENCE:
        return SILENCE_TOKEN
    if action.action is Action.DELEGATE:
        return f"{RESPONSE_TOKEN} {action.text} {DELEGATION_TAG} {action.delegated_question or ''}".strip()
    return f"{RESPONSE_TOKEN} {action.text}".strip()


def _first_marker(text: str) -> int:
    present = [i for i in (text.find(SILENCE_TOKEN), text.find(RESPONSE_TOKEN)) if i != -1]
    return min(present) if present else -1


def _collapse(text: str) -> str:
    return " ".join(text.split())
