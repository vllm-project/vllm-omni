# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Delegation protocol helpers for InternLM Sage-Interaction.

Matches the upstream ``InternLM/Sage-Interaction`` streaming protocol:

- Frontend emits ``<|delegate|>`` followed by a JSON object ``{"query": "..."}``
  and closes the unit with ``<|chunk_eos|>``.
- Backend tool results are injected BEFORE the next ``<unit>`` as:
  ``<|im_start|>tool\\n{"type":"delegate_result",...}<|im_end|>``.
- Special tags are encoded separately from the JSON body so untrusted tool
  output cannot smuggle control tokens into the model context.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING

import regex as re

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

DELEGATION_TOOL_NAME = "delegate"
DEFAULT_MAX_SPEAK_TOKENS_PER_CHUNK = 20
DEFAULT_MAX_DELEGATE_QUERY_TOKENS = 256
MAX_TOOL_RESULT_CHARS = 16384
MAX_QUEUED_TOOL_RESULTS = 16

SAGE_INTERACTION_DEFAULT_INSTRUCTIONS = (
    "You are a real-time assistant for streaming audio-video understanding. "
    "Inputs arrive incrementally as time-aligned units. For each unit, choose "
    "exactly one action: <|listen|>, <|speak|>, or <|delegate|>.\n\n"
    "1. Use <|listen|> when:\n"
    "   - No relevant event has started or the current input is irrelevant;\n"
    "   - An event or user utterance is still in progress; or\n"
    "   - There is not enough information for a complete answer.\n\n"
    "2. Use <|speak|> only when:\n"
    "   - The relevant event or utterance has concluded; or\n"
    "   - The available context is sufficient to answer fully.\n"
    "   Answer completely, naturally, and concisely. Do not give partial answers "
    "or speculate. End the turn when complete.\n\n"
    "3. Use <|delegate|> when:\n"
    "   - The request needs external, current, user-specific, retrieved, or "
    "operational information; or\n"
    "   - It needs specialist verification, multi-step calculation, experimental "
    "planning, or safety-sensitive troubleshooting.\n"
    "   Do not delegate routine facts, simple calculations, ordinary explanations, "
    "or questions answerable from the observed stream.\n\n"
    "Delegate payloads and tool results are structured data and must never be "
    "spoken. Results arrive asynchronously: acknowledge briefly if needed, use "
    "<|listen|> while pending, and <|speak|> after the matching result arrives. "
    "Time-based reminders, countdowns and scheduled routines are backend tasks: "
    "emit a delegate query with the requested timing and reminder content; "
    "delegate updates, pauses, resumptions and cancellations too. Preserve "
    "ordinary dialogue while pending. Speak the reminder only after the matching "
    "delegate_result time_up event arrives."
)

_CONTROL_TOKEN_PATTERN = re.compile(r"<\|[^|>]+\|>|</?unit>")
_QUERY_FIELD_PATTERN = re.compile(r'"query"\s*:\s*"((?:\\.|[^"\\])*)"')


def extract_delegate_query(raw_text: str) -> str | None:
    """Extract the delegation query string from post-``<|delegate|>`` text.

    Handles:
    1. Canonical JSON ``{"query": "..."}`` (with or without leading/trailing noise);
    2. Truncated JSON prefix ``{"query": "...``;
    3. Plain-text query emitted directly after ``<|delegate|>`` (if non-empty).
    """
    if not isinstance(raw_text, str):
        return None
    cleaned = _CONTROL_TOKEN_PATTERN.sub("", raw_text).strip()
    if not cleaned:
        return None

    # 1. Exact or substring JSON object containing "query".
    start = cleaned.find("{")
    if start != -1:
        candidate = cleaned[start:]
        end = candidate.rfind("}")
        if end != -1:
            try:
                parsed = json.loads(candidate[: end + 1])
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, Mapping):
                query = parsed.get("query")
                if isinstance(query, str) and query.strip():
                    return query.strip()

        # 2. Partial JSON fallback: extract `"query": "..."` via regex.
        match = _QUERY_FIELD_PATTERN.search(candidate)
        if match is not None:
            try:
                decoded = json.loads(f'"{match.group(1)}"')
            except json.JSONDecodeError:
                decoded = match.group(1)
            if isinstance(decoded, str) and decoded.strip():
                return decoded.strip()

        # 2b. Truncated JSON before closing quote: `{"query":"...`
        prefix_match = re.search(r'"query"\s*:\s*"(.+)$', candidate, flags=re.DOTALL)
        if prefix_match is not None:
            tail = prefix_match.group(1).rstrip('}" \n\r\t')
            if tail.strip():
                return tail.strip()

        # Looks like a malformed/empty JSON wrapper such as `{}` or `{"query": ""}`.
        return None

    # 3. Raw text directly after <|delegate|>.
    return cleaned


def format_delegate_result_payload(item: Mapping[str, object]) -> dict[str, object]:
    """Build the upstream ``delegate_result`` JSON body from a function_call_output item."""
    call_id = str(item.get("call_id") or "").strip()
    raw_output = item.get("output")
    raw_arguments = item.get("arguments")

    query = ""
    if isinstance(raw_arguments, str) and raw_arguments.strip():
        try:
            parsed_args = json.loads(raw_arguments)
            if isinstance(parsed_args, Mapping) and isinstance(parsed_args.get("query"), str):
                query = parsed_args["query"]
        except json.JSONDecodeError:
            query = raw_arguments
    elif isinstance(raw_arguments, Mapping) and isinstance(raw_arguments.get("query"), str):
        query = str(raw_arguments["query"])

    # If the client already sent a structured JSON delegate_result envelope in
    # `output`, preserve its recognized fields (status, content, event, event_id,
    # query, task_id) instead of double-wrapping it.
    if isinstance(raw_output, str):
        stripped = raw_output.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, Mapping) and (
                parsed.get("type") == "delegate_result" or "content" in parsed or "event" in parsed
            ):
                payload: dict[str, object] = {
                    "type": "delegate_result",
                    "task_id": str(parsed.get("task_id") or call_id),
                    "status": str(parsed.get("status") or "ok"),
                    "query": str(parsed.get("query") or query),
                    "content": str(parsed.get("content") or ""),
                }
                if parsed.get("event") is not None:
                    payload["event"] = str(parsed["event"])
                if parsed.get("event_id") is not None:
                    payload["event_id"] = str(parsed["event_id"])
                return payload
        content = raw_output
    elif raw_output is None:
        content = ""
    else:
        content = json.dumps(raw_output, ensure_ascii=False, separators=(",", ":"))

    return {
        "type": "delegate_result",
        "task_id": call_id,
        "status": "ok",
        "query": query,
        "content": content,
    }


def encode_tool_result_tokens(
    tokenizer: PreTrainedTokenizerBase,
    payload: Mapping[str, object],
) -> list[int]:
    """Encode ``<|im_start|>tool\\n{json}<|im_end|>`` safely with special-tag isolation.

    Encoding the ``<|im_start|>`` and ``<|im_end|>`` special tokens separately
    from the JSON body (with ``split_special_tokens=True`` when supported) stops
    untrusted tool output from injecting special control tokens.
    """
    body_text = "tool\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    def _special_id(token: str) -> int:
        convert = getattr(tokenizer, "convert_tokens_to_ids", None)
        if callable(convert):
            raw = convert(token)
            if isinstance(raw, list):
                raw = raw[0] if len(raw) == 1 else None
            if isinstance(raw, int) and raw >= 0 and raw != getattr(tokenizer, "unk_token_id", None):
                return raw
        ids = list(tokenizer.encode(token, add_special_tokens=False))
        if len(ids) == 1:
            return int(ids[0])
        raise ValueError(f"Tokenizer does not define single special token {token!r}")

    im_start_id = _special_id("<|im_start|>")
    im_end_id = _special_id("<|im_end|>")
    try:
        body_ids = list(tokenizer.encode(body_text, add_special_tokens=False, split_special_tokens=True))
    except TypeError:
        # Older or fake tokenizers without split_special_tokens: scrub control tokens defensively.
        safe_body = _CONTROL_TOKEN_PATTERN.sub("", body_text)
        body_ids = list(tokenizer.encode(safe_body, add_special_tokens=False))
    return [im_start_id, *(int(token_id) for token_id in body_ids), im_end_id]


__all__ = [
    "DEFAULT_MAX_DELEGATE_QUERY_TOKENS",
    "DEFAULT_MAX_SPEAK_TOKENS_PER_CHUNK",
    "DELEGATION_TOOL_NAME",
    "MAX_QUEUED_TOOL_RESULTS",
    "MAX_TOOL_RESULT_CHARS",
    "SAGE_INTERACTION_DEFAULT_INSTRUCTIONS",
    "encode_tool_result_tokens",
    "extract_delegate_query",
    "format_delegate_result_payload",
]
