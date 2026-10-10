# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qwen duplex function calls.

The session ledger owns call identity. This module only recognizes one completed
Hermes-style ``<tool_call>`` in Thinker text, and holds that markup out of the
spoken transcript until the call is complete. Parsing goes through vLLM's
Hermes tool parser; this file does not define a second grammar.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
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


def parse_completed_tool_call(inner: str) -> dict[str, str] | None:
    """Return one call from a finished ``<tool_call>`` body, or None."""
    wrapped = f"{TOOL_CALL_START}{inner}{TOOL_CALL_END}"
    extracted = _parser.extract_tool_calls(wrapped, _PARSE_REQUEST)
    if not extracted.tools_called or not extracted.tool_calls:
        return None
    function = extracted.tool_calls[0].function
    name = getattr(function, "name", None)
    arguments = getattr(function, "arguments", None)
    if not isinstance(name, str) or not name:
        return None
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments if arguments is not None else {}, ensure_ascii=False)
    return {"call_id": f"call_{uuid4().hex}", "name": name, "arguments": arguments}


@dataclass
class ThinkerToolCursor:
    """Per-request Thinker text, and how much of it may already be spoken."""

    text: str = ""
    spoken: int = 0
    tool_turn: bool = False
    emitted_call_ids: set[str] = field(default_factory=set)

    def holding_open_call(self) -> bool:
        rest = self.text[self.spoken :]
        return TOOL_CALL_START in rest and TOOL_CALL_END not in rest.split(TOOL_CALL_START, 1)[1]

    def absorb(self, incoming: str) -> tuple[str, dict[str, str] | None]:
        """Merge one Thinker delta and return ``(speakable, call)``.

        A repeated cumulative snapshot does not open a second call. An open tag
        with no closing tag contributes no speech and no call.
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
            if parsed is None or parsed["call_id"] in self.emitted_call_ids:
                speech.append(rest[: end + len(TOOL_CALL_END)])
                continue
            self.emitted_call_ids.add(parsed["call_id"])
            self.tool_turn = True
            call = parsed
            break
        return "".join(speech), call

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
    if not incoming:
        return previous
    if not previous or incoming.startswith(previous):
        return incoming
    return previous + incoming
