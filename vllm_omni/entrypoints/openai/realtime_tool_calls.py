# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Self-contained <tool_call> XML-tag parser for vLLM-Omni's /v1/realtime endpoint.

Qwen's own chat template (see the model's chat_template.json) instructs the
model to emit tool calls as:

    <tool_call>
    {"name": <function-name>, "arguments": <args-json-object>}
    </tool_call>

This is the same wire format vLLM's `Hermes2ProToolParser`
(vllm/tool_parsers/hermes_tool_parser.py) already parses for
/v1/chat/completions, and the algorithm below (regex name extraction,
length-based suffix diff for streaming arguments, partial-tag-overlap
withholding) intentionally mirrors that parser's approach.

We do not import that class directly: `ToolParser.extract_tool_calls*`
methods are typed against `ChatCompletionRequest | ResponsesRequest` and read
`request.tools`/`request.tool_choice` from it, and for Qwen3 specifically
tool-call parsing has moved to vLLM's structured-output/guided-decoding
engine rather than a plain text parser (see `vllm/tool_parsers/
qwen3_engine_tool_parser.py`) - neither shape fits this self-hosted,
websocket-driven streaming loop without either duck-typing a fake request
object or pulling in the structured-output engine. A small, self-contained
parser is easier to review and keeps this feature independent of vLLM
tool-parser internals that may change across versions.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

_TOOL_CALL_START = "<tool_call>"
_TOOL_CALL_END = "</tool_call>"
_NAME_RE = re.compile(r'"name"\s*:\s*"([^"]+)"')
_ARGS_KEY_RE = re.compile(r'"arguments"\s*:\s*')
_JSON_DECODER = json.JSONDecoder()


@dataclass
class ToolCallDelta:
    """One incremental piece of a tool call. `name` is set at most once per
    index (the first time it becomes available); `arguments_delta` is the new
    tail of the arguments JSON text since the last delta for this index."""

    index: int
    name: str | None = None
    arguments_delta: str = ""


@dataclass
class ToolCallStreamState:
    """Tracks incremental parsing progress across repeated calls with the
    growing `current_text` of a single generation."""

    sent_content_idx: int = 0
    tool_call_starts: list[int] = field(default_factory=list)
    names_sent: list[bool] = field(default_factory=list)
    streamed_args: list[str] = field(default_factory=list)

    def has_tool_calls(self) -> bool:
        return bool(self.tool_call_starts)


def _partial_tag_overlap(text: str, tag: str) -> int:
    """Length of the longest suffix of `text` that is a proper prefix of
    `tag` - e.g. text ending in "...</tool_c" against tag "</tool_call>"
    returns 5, so callers can withhold that suffix until it's known whether
    it's about to become part of the tag."""
    max_overlap = min(len(text), len(tag) - 1)
    for size in range(max_overlap, 0, -1):
        if text.endswith(tag[:size]):
            return size
    return 0


def _json_value_end(text: str) -> int | None:
    """If `text` starts (after optional leading whitespace) with a complete
    JSON object or array, return the index just past its closing bracket.
    Returns None if `text` doesn't start with '{'/'[' or the value isn't
    complete yet. Used to find exactly where an "arguments" value ends
    without assuming anything about what (if anything) follows it - argument
    values can themselves contain nested objects/arrays.

    `raw_decode` is exactly this primitive: it parses one JSON value from the
    start of the string and reports where it stopped, ignoring trailing text,
    and raises on an incomplete value (the normal case mid-stream)."""
    stripped = text.lstrip()
    # Restrict to object/array: `raw_decode` would also accept a bare scalar,
    # but "arguments" is always a JSON object per the tool-call format.
    if not stripped or stripped[0] not in "{[":
        return None
    try:
        _, end = _JSON_DECODER.raw_decode(stripped)
    except ValueError:
        return None
    return (len(text) - len(stripped)) + end


def extract_deltas(current_text: str, state: ToolCallStreamState) -> tuple[str, list[ToolCallDelta]]:
    """Given the full accumulated generation text so far, return
    (new plain-content substring, new tool-call deltas) since the last call
    with this `state`. Mutates `state` in place."""
    tool_deltas: list[ToolCallDelta] = []

    search_from = state.tool_call_starts[-1] + len(_TOOL_CALL_START) if state.tool_call_starts else 0
    while (idx := current_text.find(_TOOL_CALL_START, search_from)) != -1:
        state.tool_call_starts.append(idx)
        state.names_sent.append(False)
        state.streamed_args.append("")
        search_from = idx + len(_TOOL_CALL_START)

    if state.tool_call_starts:
        content_end = state.tool_call_starts[0]
    else:
        content_end = len(current_text) - _partial_tag_overlap(current_text, _TOOL_CALL_START)
    content_delta = ""
    if content_end > state.sent_content_idx:
        content_delta = current_text[state.sent_content_idx : content_end]
        state.sent_content_idx = content_end

    for i, start in enumerate(state.tool_call_starts):
        block_start = start + len(_TOOL_CALL_START)
        end_idx = current_text.find(_TOOL_CALL_END, block_start)
        if end_idx == -1:
            raw = current_text[block_start:]
            overlap = _partial_tag_overlap(raw, _TOOL_CALL_END)
            if overlap:
                raw = raw[: len(raw) - overlap]
        else:
            raw = current_text[block_start:end_idx]

        if not state.names_sent[i]:
            name_match = _NAME_RE.search(raw)
            if name_match is None:
                continue  # don't parse ahead to arguments until a name is available
            tool_deltas.append(ToolCallDelta(index=i, name=name_match.group(1)))
            state.names_sent[i] = True

        args_match = _ARGS_KEY_RE.search(raw)
        if args_match is None:
            continue
        tail = raw[args_match.end() :]
        # Find the exact end of the arguments JSON value via bracket-depth
        # matching, rather than assuming "the last '}' in the block is the
        # outer object's, not the value's" - that assumption breaks whenever
        # the trailing "}\n" (closing the outer {"name":..., "arguments":...}
        # object, before </tool_call>) arrives in the same chunk as the
        # arguments value's own closing brace: naively stripping one trailing
        # '}' either over- or under-trims depending on exact token boundaries,
        # and (worse) can desync the running length-based diff against
        # state.streamed_args so the final delta silently never gets sent,
        # leaving invalid trailing "}\n" baked into the accumulated arguments.
        value_end = _json_value_end(tail)
        args_str = tail[:value_end] if value_end is not None else tail
        new_args = args_str[len(state.streamed_args[i]) :]
        if new_args:
            tool_deltas.append(ToolCallDelta(index=i, arguments_delta=new_args))
            state.streamed_args[i] = args_str

    return content_delta, tool_deltas
