# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""AURA_v2 duplex function calls.

The AURA_v2 checkpoint's ``chat_template.jinja`` wraps calls in
``<tool_call>`` / ``</tool_call>`` like Qwen3.5's Hermes format, but the body
inside is its own nested-tag shape --
``<function=name>\\n<parameter=p>\\nvalue\\n</parameter>\\n</function>`` --
not the ``<tool_call>{"name":...,"arguments":{...}}</tool_call>`` JSON body
vLLM's ``Hermes2ProToolParser`` parses; that parser raises on AURA_v2's real
output. So this file has its own small regex parser for the body, not
a wrapper around that parser. It is also not an import of
``qwen3_omni/duplex/tools_utils.py``: AURA Stage1 requests cumulative text,
not DELTA, so the scanning shape differs too.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

import regex as re

TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"

_FUNCTION_RE = re.compile(r"<function=([^>]+)>(.*?)</function>", re.DOTALL)
_PARAMETER_RE = re.compile(r"<parameter=([^>]+)>\n?(.*?)\n?</parameter>", re.DOTALL)


def chat_template_tools(tools: object) -> list[dict[str, object]]:
    """Map Realtime ``tools`` objects onto the chat-template shape.

    Realtime tools are flat ``{type, name, description, parameters}``. The
    AURA chat template expects ``{type, function: {name, description,
    parameters}}``. Objects that already have a ``function`` mapping pass
    through.
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


def build_tools_block(tools: list[dict[str, object]]) -> str:
    """Render the ``# Tools`` system block, verbatim per AURA_v2's own chat template.

    Empty ``tools`` renders nothing: callers skip prepending this to the
    system prompt in that case.
    """
    if not tools:
        return ""
    lines = "".join(f"\n{json.dumps(tool, ensure_ascii=False)}" for tool in tools)
    return (
        "# Tools\n\nYou have access to the following functions:\n\n<tools>"
        + lines
        + "\n</tools>"
        + (
            "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
            "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\n"
            "value_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second "
            "parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n"
            "<IMPORTANT>\nReminder:\n"
            "- Function calls MUST follow the specified format: an inner <function=...></function> "
            "block must be nested within <tool_call></tool_call> XML tags\n"
            "- Required parameters MUST be specified\n"
            "- You may provide optional reasoning for your function call in natural language BEFORE "
            "the function call, but NOT after\n"
            "- If there is no function call available, answer the question like normal with your "
            "current knowledge and do not tell the user about function calls\n</IMPORTANT>"
        )
    )


def _reject_nonfinite(constant: str) -> None:
    raise ValueError(constant)


def _parameter_value(raw: str) -> object:
    """A ``<parameter>`` body as JSON if it parses as one, else the raw string.

    The template puts no quotes around string values (``17 * 23``, ``台北``),
    so most bodies fail ``json.loads`` and fall through to the raw string;
    a body that is itself valid JSON (a number, ``true``/``false``, or an
    object) is still accepted, matching how the tool's own argument model
    would coerce it. ``parse_constant`` rejects the ``NaN``/``Infinity``
    extension so a literal word like that is not silently turned into a
    float.
    """
    stripped = raw.strip()
    try:
        return json.loads(stripped, parse_constant=_reject_nonfinite)
    except ValueError:
        return stripped


def _stable_call_id(anchor: str, name: str, arguments: str) -> str:
    """Same span parses to the same id; a later ``<|im_end|>`` must not remint it."""
    digest = hashlib.sha256(f"{anchor}\n{name}\n{arguments}".encode()).hexdigest()
    return f"call_{digest[:32]}"


def parse_completed_tool_call(inner: str, *, anchor: str | None = None) -> dict[str, str] | None:
    """Return one call from a finished ``<tool_call>`` body, or None.

    ``anchor`` is the span's offset in the cumulative text. Re-parsing the same
    span keeps ``call_id``; two identical calls at different offsets stay distinct.
    """
    match = _FUNCTION_RE.search(inner)
    if match is None:
        return None
    name = match.group(1).strip()
    if not name:
        return None
    arguments: dict[str, object] = {
        param_match.group(1).strip(): _parameter_value(param_match.group(2))
        for param_match in _PARAMETER_RE.finditer(match.group(2))
    }
    try:
        arguments_text = json.dumps(arguments, ensure_ascii=False)
    except (TypeError, ValueError):
        return None
    return {
        "call_id": _stable_call_id(inner if anchor is None else anchor, name, arguments_text),
        "name": name,
        "arguments": arguments_text,
    }


def extract_finished_tool_calls(cumulative_text: str) -> list[dict[str, str]]:
    """Return every completed ``<tool_call>`` in finished cumulative Stage1 text, in order.

    AURA Stage1 requests cumulative (not DELTA) text, so by the time a segment
    finishes the whole turn is in one string. The model may emit several
    calls back to back in one turn; a block that does not parse is skipped.
    """
    calls: list[dict[str, str]] = []
    start = cumulative_text.find(TOOL_CALL_START)
    while start != -1:
        end = cumulative_text.find(TOOL_CALL_END, start + len(TOOL_CALL_START))
        if end == -1:
            break
        call = parse_completed_tool_call(
            cumulative_text[start + len(TOOL_CALL_START) : end],
            anchor=str(start),
        )
        if call is not None:
            calls.append(call)
        start = cumulative_text.find(TOOL_CALL_START, end + len(TOOL_CALL_END))
    return calls


def strip_tool_call_markup(text: str) -> str:
    """Drop a ``<tool_call>`` block and everything after it from speech/display text.

    The template instructs the model to emit no suffix after ``</tool_call>``,
    so truncating at the opening tag is enough once the text is final.
    """
    index = text.find(TOOL_CALL_START)
    return text if index == -1 else text[:index]


def safe_streaming_prefix(text: str) -> str:
    """Prefix of ``text`` safe to stream now: before a tool-call tag, complete or forming.

    AURA Stage1 text is cumulative, so each chunk is checked from scratch (no
    per-request cursor). A complete tag truncates there; an in-progress tag
    (a tail that is itself a prefix of ``<tool_call>``) holds that tail back
    so a client never sees ``<tool_c`` one character at a time.
    """
    index = text.find(TOOL_CALL_START)
    if index != -1:
        return text[:index]
    max_overlap = min(len(TOOL_CALL_START) - 1, len(text))
    for overlap in range(max_overlap, 0, -1):
        if text.endswith(TOOL_CALL_START[:overlap]):
            return text[: len(text) - overlap]
    return text


__all__ = [
    "TOOL_CALL_END",
    "TOOL_CALL_START",
    "build_tools_block",
    "chat_template_tools",
    "extract_finished_tool_calls",
    "parse_completed_tool_call",
    "safe_streaming_prefix",
    "strip_tool_call_markup",
]
