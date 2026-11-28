# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tool calling for Sophon-Realtime (pure Python, no vLLM imports so it can be unit-tested on its own).

The thinker writes tool calls in the Qwen/Hermes format: <tool_call>{"name": ..., "arguments": {...}}</tool_call>.
- cut_tool(): what may be spoken / shown -- everything before the first <tool_call> (a partial "<too" at the end is
  held back).
- parse_tool_calls(): the calls themselves, parsed from the full thinker text when the reply finishes.
- run_mock(): server-side mock executor (session extra_body.tool_mocks), so a full call -> result -> spoken answer
  loop can be tested
  without a client that executes tools.
- flatten(): fallback for chat templates that do not know `tools=` (tools go into the system text, calls/results
  become plain text)."""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping

import regex as re

TC_OPEN, TC_CLOSE = "<tool_call>", "</tool_call>"
_RE_CALL = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.S)
MAX_STEPS = 3  # at most this many call -> result -> reply rounds per user turn

SYS_TOOL = {
    "zh": (
        "你可以调用上面列出的工具。需要用工具时：先用一句很短的口语告诉用户你要做什么（比如“我帮你查一下”），"
        "然后立刻按格式输出工具调用；工具名、参数和格式都不会被念出来，也不要在口语里复述它们。拿到工具结果后，"
        "用一两句自然的口语把结果告诉用户。闲聊或你自己就能回答的问题，不要调用工具。"
    ),
    "en": (
        "You can call the tools listed above. When you need one: first say one very short spoken phrase about what "
        'you\'re doing (e.g. "Let me check that"), then immediately output the tool call in the required format; the '
        "tool name, arguments and format are never spoken, so don't repeat them out loud. When the tool result comes "
        "back, tell the user the result in one or two natural spoken sentences. For chit-chat or anything you can "
        "answer yourself, don't call a tool."
    ),
}


WAIT = "<wait>"  # the model's "nothing to say right now" (e.g. a backend update with nothing new): reply stays silent


def cut_tool(text: str, final: bool = False) -> str:
    """The part of the thinker text that may be spoken: up to the first <tool_call>. While streaming, a trailing prefix
    of "<tool_call>" (e.g. "<to") is held back until the next characters show whether it is a call.
    A reply that starts with <wait> is silent (nothing is spoken or shown); while streaming, a prefix of it is held
    back."""
    head = text.lstrip()
    if head.startswith(WAIT) or (not final and head and WAIT.startswith(head)):
        return ""
    text = text.replace(WAIT, "")
    i = text.find(TC_OPEN)
    if i >= 0:
        return text[:i]
    if not final:
        for k in range(len(TC_OPEN) - 1, 0, -1):
            if text.endswith(TC_OPEN[:k]):
                return text[:-k]
    return text


def has_tool(text: str) -> bool:
    return TC_OPEN in text


def _loads(s: str):
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:  # noqa: BLE001
        pass
    i, j = s.find("{"), s.rfind("}")  # trailing junk / missing close tag: try the outermost {...}
    if i >= 0 and j > i:
        try:
            return json.loads(s[i : j + 1])
        except Exception:  # noqa: BLE001
            return None
    return None


def parse_tool_calls(text: str) -> list[dict]:
    """[{"id", "name", "arguments": dict}] for every well-formed call in the thinker text."""
    out = []
    for m in _RE_CALL.finditer(text or ""):
        j = _loads(m.group(1))
        if not isinstance(j, Mapping) or not isinstance(j.get("name"), str):
            continue
        args = j.get("arguments", j.get("parameters", {}))
        if isinstance(args, str):
            args = _loads(args) or {}
        out.append(
            {
                "id": "call_" + uuid.uuid4().hex[:12],
                "name": j["name"],
                "arguments": args if isinstance(args, Mapping) else {},
            }
        )
    return out


def normalize_tools(tools) -> list[dict]:
    """Accept OpenAI Realtime style ({"type": "function", "name", "description", "parameters"}) or Chat Completions
    style
    ({"type": "function", "function": {...}}); return Chat Completions style (what HF chat templates expect)."""
    out = []
    for t in tools or []:
        if not isinstance(t, Mapping):
            continue
        f = t.get("function") if isinstance(t.get("function"), Mapping) else t
        if not isinstance(f.get("name"), str):
            continue
        fn = {"name": f["name"]}
        if f.get("description"):
            fn["description"] = f["description"]
        fn["parameters"] = f.get("parameters") or {"type": "object", "properties": {}}
        out.append({"type": "function", "function": fn})
    return out


def run_mock(name: str, args: Mapping, mocks: Mapping | None) -> str:
    """Mock result for a tool call. mocks = {tool name or "*": result}; a string result may use {arg} placeholders,
    any other value is returned as JSON. Unknown tools get a generic {"ok": true} result."""
    m = None
    if isinstance(mocks, Mapping):
        m = mocks.get(name, mocks.get("*"))
    if m is None:
        return json.dumps(
            {"ok": True, "tool": name, "arguments": dict(args), "note": "mock result"}, ensure_ascii=False
        )
    if isinstance(m, str):
        try:
            return m.format(**{k: v for k, v in args.items()})
        except Exception:  # noqa: BLE001
            return m
    return json.dumps(m, ensure_ascii=False)


def tool_block(tools: list[dict]) -> str:
    lines = [
        "# Tools",
        "",
        "You may call one or more functions to assist with the user query.",
        "",
        "You are provided with function signatures within <tools></tools> XML tags:",
        "<tools>",
    ]
    lines += [json.dumps(t, ensure_ascii=False) for t in tools]
    lines += [
        "</tools>",
        "",
        "For each function call, return a json object with function name and arguments within "
        "<tool_call></tool_call> XML tags:",
        "<tool_call>",
        '{"name": <function-name>, "arguments": <args-json-object>}',
        "</tool_call>",
    ]
    return "\n".join(lines)


def _call_json(fn: Mapping) -> str:
    return json.dumps({"name": fn["name"], "arguments": fn["arguments"]}, ensure_ascii=False)


def flatten(messages: list[dict], tools: list[dict]) -> list[dict]:
    """For templates without tool support: same text the Qwen template would produce, built by hand."""
    out = []
    for m in messages:
        m = dict(m)
        if m["role"] == "system":
            m["content"] = f"{m['content']}\n\n{tool_block(tools)}"
        elif m["role"] == "assistant" and m.get("tool_calls"):
            calls = "\n".join(f"{TC_OPEN}\n{_call_json(c['function'])}\n{TC_CLOSE}" for c in m.pop("tool_calls"))
            m["content"] = (m.get("content") or "") + ("\n" if m.get("content") else "") + calls
        elif m["role"] == "tool":
            m = {"role": "user", "content": f"<tool_response>\n{m['content']}\n</tool_response>"}
        out.append(m)
    return out
