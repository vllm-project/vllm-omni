# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""MiniCPM-o 4.5 ids for the opt-in per-unit token trace (``extra_body.trace_tokens``).

``unit_token_trace`` reports what the Thinker (Stage 0) and the Talker (Stage 1)
consumed and produced for one unit; the plugin's hook of the same name calls it.

The Talker's per-unit input is built by the Thinker->Talker stage bridge
(``llm2tts``, which runs in the orchestrator process): the unit's Thinker text
ids, embedded and summed with their projected hidden states, then
``<audio_bos>``. The scheduler only sees placeholder ids for that prompt, so
the bridge records the real ids here for traced sessions, and the plugin's
``unit_token_trace`` takes one entry per Talker segment end.

Entries are keyed by the stage request id, which carries the session and the
epoch: a cancelled epoch's leftover hand-offs are dropped when the next epoch's
request records its first one, and everything of a session goes when its trace
stops. With no traced session the bridge does nothing beyond one set check.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping

from vllm_omni.engine.duplex.contracts import duplex_session_id_from_request_id
from vllm_omni.engine.duplex.plugin import token_id_list
from vllm_omni.model_executor.common.request_outputs import first_completion, multimodal_output

#: ``tts_config.audio_bos_token_id`` of the released checkpoint; the Talker
#: appends it after the hand-off condition (``_build_condition_embeddings``).
TTS_AUDIO_BOS_ID = 151687

_TRACED_SESSIONS: set[str] = set()
#: request id -> (session id, hand-offs not yet taken by a Talker segment end)
_HANDOFFS: dict[str, tuple[str, deque[list[int]]]] = {}


def start(session_id: str) -> None:
    _TRACED_SESSIONS.add(session_id)


def stop(session_id: str) -> None:
    _TRACED_SESSIONS.discard(session_id)
    for request_id in [rid for rid, (sid, _) in _HANDOFFS.items() if sid == session_id]:
        del _HANDOFFS[request_id]


def record_handoff(request_id: str, token_ids: list[int]) -> None:
    """Called by the stage bridge for every Talker hand-off; a no-op unless the session is traced."""
    if not _TRACED_SESSIONS:
        return
    session_id = duplex_session_id_from_request_id(request_id)
    if session_id is None or session_id not in _TRACED_SESSIONS:
        return
    entry = _HANDOFFS.get(request_id)
    if entry is None:
        # A new request of this session (a new epoch): an older one's hand-offs never reach a Talker.
        for stale in [rid for rid, (sid, _) in _HANDOFFS.items() if sid == session_id]:
            del _HANDOFFS[stale]
        entry = _HANDOFFS[request_id] = (session_id, deque())
    entry[1].append([int(token_id) for token_id in token_ids])


def pop_handoff(request_id: str) -> list[int] | None:
    entry = _HANDOFFS.get(request_id)
    return entry[1].popleft() if entry is not None and entry[1] else None


def _cursor_int(cursor: Mapping[str, object], key: str, limit: int) -> int:
    value = cursor.get(key, 0)
    return value if isinstance(value, int) and 0 <= value <= limit else 0


def unit_token_trace(
    *, stage_id: int, output: object, cursor: dict[str, object], runtime_config: Mapping[str, object]
) -> dict[str, object] | None:
    """Per-unit token ids of the Thinker (Stage 0) and the Talker (Stage 1).

    Thinker: the scheduler's prompt ids are placeholders. The worker ships
    the ids it actually appended (``<unit>``, the audio placeholders, the
    re-fed previous terminator and ``</unit>``; on the first unit the
    session context, reported as the prompt) in the output's multimodal
    metadata as ``duplex_prompt_token_ids``: one prompt-length row per
    segment, accumulated across segments, so this unit's ids are its tail.
    Without that row (or with one shorter than the unit's prompt delta) the
    placeholders are reported labelled ``placeholder:<id>``, never as pieces.
    Output: the delta of the cumulative generated ids.

    Talker: input is the bridge's hand-off (``record_handoff``) followed by
    the ``<audio_bos>`` the Talker appends; output the codec ids generated
    in the segment, labelled ``s3:<id>``.
    """
    completion = first_completion(output)
    generated = token_id_list(getattr(completion, "token_ids", None)) if completion is not None else []
    prompt = token_id_list(getattr(output, "prompt_token_ids", None))
    request_id = getattr(output, "request_id", None)
    first_segment = "seen" not in cursor
    if cursor.get("request_id") != request_id:
        # A new stage request (a new epoch) restarts the cumulative ids.
        cursor["request_id"], cursor["output_len"], cursor["prompt_len"] = request_id, 0, 0
    cursor["seen"] = True
    output_start = _cursor_int(cursor, "output_len", len(generated))
    prompt_start = _cursor_int(cursor, "prompt_len", len(prompt))
    cursor["output_len"], cursor["prompt_len"] = len(generated), len(prompt)
    new_output = generated[output_start:]
    new_prompt_len = len(prompt) - prompt_start
    if stage_id == 0:
        real = token_id_list(multimodal_output(output, completion).get("duplex_prompt_token_ids"))
        unit_input: list[object]
        if 0 < new_prompt_len <= len(real):
            unit_input = list(real[-new_prompt_len:])
        else:
            # The worker's ids are missing or shorter than this unit's prompt delta: what the
            # scheduler saw are placeholders, labelled so, as for the Talker below, rather
            # than rendered as if they were the tokens the model consumed.
            unit_input = [[token_id, f"placeholder:{token_id}"] for token_id in prompt[prompt_start:]]
        trace: dict[str, object] = {"stage": "thinker", "input": unit_input, "output": new_output}
        context_len = runtime_config.get("duplex_first_append_context_tokens")
        # Only the session's first segment carries the context; a later
        # Stage-0 window rebuild (sliding window) is reported whole.
        if first_segment and isinstance(context_len, int) and 0 < context_len <= len(unit_input):
            trace["prompt_input"] = unit_input[:context_len]
            trace["input"] = unit_input[context_len:]
        return trace
    if stage_id == 1:
        handoff = pop_handoff(request_id) if isinstance(request_id, str) else None
        talker_input: list[object]
        if handoff is not None:
            # The Talker's own boundary token; the Thinker tokenizer names
            # the same id differently, so it is labelled explicitly.
            talker_input = [*handoff, [TTS_AUDIO_BOS_ID, "<audio_bos>"]]
        else:
            talker_input = [[token_id, f"placeholder:{token_id}"] for token_id in prompt[prompt_start:]]
        return {
            "stage": "talker",
            "input": talker_input,
            "output": [[token_id, f"s3:{token_id}"] for token_id in new_output],
        }
    return None


__all__ = ["TTS_AUDIO_BOS_ID", "pop_handoff", "record_handoff", "start", "stop", "unit_token_trace"]
