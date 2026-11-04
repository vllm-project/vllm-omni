# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qwen duplex function calls.

Recognize one completed Hermes-style ``<tool_call>`` in Thinker text, and hold
that markup out of the spoken transcript until the call is complete. The server
assigns ``call_id``. Parsing goes through vLLM's Hermes tool parser; this file
does not define a second grammar.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast
from uuid import uuid4

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.tool_parsers.hermes_tool_parser import Hermes2ProToolParser
from vllm.tool_parsers.utils import partial_tag_overlap

TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"

_parser = Hermes2ProToolParser(cast(object, object()))
_PARSE_REQUEST = cast(ChatCompletionRequest, object())


def chat_template_tools(tools: object) -> list[dict[str, object]]:
    """Map Realtime ``tools`` objects onto the chat-template shape.

    Realtime tools are flat ``{type, name, description, parameters}``. The
    processor expects ``{type, function: {name, description, parameters}}``.
    Objects that already have a ``function`` mapping are passed through.
    """
    if not isinstance(tools, list):
        return []
    converted: list[dict[str, object]] = []
    for tool in tools:
        if not isinstance(tool, Mapping):
            continue
        function = tool.get("function")
        if isinstance(function, Mapping) and isinstance(function.get("name"), str) and function.get("name"):
            converted.append(dict(tool))
            continue
        name = tool.get("name")
        if not isinstance(name, str) or not name:
            continue
        payload: dict[str, object] = {"name": name}
        if isinstance(tool.get("description"), str):
            payload["description"] = tool["description"]
        if isinstance(tool.get("parameters"), Mapping):
            payload["parameters"] = dict(tool["parameters"])
        converted.append({"type": "function", "function": payload})
    return converted


def _reject_nonfinite(constant: str) -> None:
    raise ValueError(constant)


def _json_argument_text(arguments: object) -> str | None:
    """RFC 8259 argument text. Hermes leaves ``NaN`` / ``Infinity`` in the string."""
    if not isinstance(arguments, str):
        try:
            return json.dumps(
                {} if arguments is None else arguments,
                ensure_ascii=False,
                allow_nan=False,
            )
        except (TypeError, ValueError):
            return None
    try:
        json.loads(arguments, parse_constant=_reject_nonfinite)
    except ValueError:
        return None
    return arguments


def parse_completed_tool_call(inner: str) -> dict[str, str] | None:
    """Return one call from a finished ``<tool_call>`` body, or None."""
    wrapped = f"{TOOL_CALL_START}{inner}{TOOL_CALL_END}"
    extracted = _parser.extract_tool_calls(wrapped, _PARSE_REQUEST)
    if not extracted.tools_called or not extracted.tool_calls:
        return None
    function = extracted.tool_calls[0].function
    name = getattr(function, "name", None)
    if not isinstance(name, str) or not name:
        return None
    arguments = _json_argument_text(getattr(function, "arguments", None))
    if arguments is None:
        return None
    return {"call_id": f"call_{uuid4().hex}", "name": name, "arguments": arguments}


@dataclass
class ThinkerToolCursor:
    """Per-request Thinker text, and how much of it may already be spoken."""

    text: str = ""
    spoken: int = 0
    tool_turn: bool = False

    def holding_open_call(self) -> bool:
        rest = self.text[self.spoken :]
        return TOOL_CALL_START in rest and TOOL_CALL_END not in rest.split(TOOL_CALL_START, 1)[1]

    def absorb(self, incoming: str) -> tuple[str, dict[str, str] | None]:
        """Merge one Thinker delta and return ``(speakable, call)``.

        Both sides non-empty are appended. Qwen duplex requests DELTA text, so
        a longer chunk that starts with the text already seen is another delta.
        An open tag contributes no speech and no call. ``spoken`` already
        passed a finished tag, so the same span is not parsed again. A later
        tag with the same name and arguments is a new call. Text after a
        closed tag stays speech.
        """
        self.text = _merge_text(self.text, incoming)
        speech: list[str] = []
        call: dict[str, str] | None = None
        while self.spoken < len(self.text):
            rest = self.text[self.spoken :]
            start = rest.find(TOOL_CALL_START)
            if start == -1:
                overlap = partial_tag_overlap(rest, TOOL_CALL_START)
                take = len(rest) - overlap
                if take:
                    speech.append(rest[:take])
                    self.spoken += take
                break
            if start:
                speech.append(rest[:start])
                self.spoken += start
                rest = self.text[self.spoken :]
            end = rest.find(TOOL_CALL_END, len(TOOL_CALL_START))
            if end == -1:
                break
            inner = rest[len(TOOL_CALL_START) : end]
            parsed = parse_completed_tool_call(inner)
            self.spoken += end + len(TOOL_CALL_END)
            self.tool_turn = True
            if parsed is None:
                # A closed tag that is not a call is not speech and not history.
                call = {"call_id": "", "name": "", "invalid": "1"}
            else:
                call = parsed
            break
        piece = "".join(speech)
        return piece, call

    def finish(self) -> str:
        """Flush held text that is not an open tool call. Drop an unclosed tag."""
        if self.holding_open_call():
            self.spoken = len(self.text)
            self.tool_turn = True
            return ""
        rest = self.text[self.spoken :]
        self.spoken = len(self.text)
        return rest


def _merge_text(previous: str, incoming: str) -> str:
    """Append one DELTA chunk.

    Qwen duplex requests ``RequestOutputKind.DELTA``. A longer chunk that
    starts with the text already seen is another delta, not a cumulative
    snapshot. Replacing it dropped ``[' very', ' very good']`` to ``' very good'``.
    """
    if not incoming:
        return previous
    if not previous:
        return incoming
    return previous + incoming
