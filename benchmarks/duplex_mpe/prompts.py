# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Portions derived from https://github.com/step-out/Duplex-MPE
# (Copyright (c) 2026 MPeval authors, MIT License; see benchmarks/duplex_mpe/LICENSE).
"""Canonical Duplex-MPE semantic judging prompt templates and SHA-256 hashing."""

from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import JsonValue

SemanticTaskKind = Literal["answer", "silence"]

SILENCE_SYSTEM_PROMPT = """You label one short piece of speech produced by a voice assistant
("Aria") that is present in a room where several humans are talking.

At this moment the humans were talking to EACH OTHER. Aria was not addressed and
was not supposed to take the floor.

Label the assistant's speech:

BACKCHANNEL - a brief phatic acknowledgement or reaction that does not take the
  floor: "yes", "yeah", "mm-hmm", "okay", "right", "wow", "got it", a short
  laugh. It adds no information and answers nothing. This is ALLOWED.

SUBSTANTIVE_INTRUSION - it does any of these: answers a question, responds to a
  request that was not directed at it, gives an explanation or advice,
  introduces new information, asks its own question, tries to steer the
  conversation, produces multiple clauses or prolonged speech, or repeats a
  nominal backchannel until it is effectively holding the floor.

Judge by CONTENT AND FUNCTION, not by length alone. A very short utterance that
answers something ("Nine dollars.") is SUBSTANTIVE_INTRUSION. A long run of
"yeah... yeah... right" that holds the floor is also SUBSTANTIVE_INTRUSION.

Reply with STRICT JSON only, "reason" under 15 words:
{"label": "BACKCHANNEL" | "SUBSTANTIVE_INTRUSION", "reason": "brief"}"""

SILENCE_USER_TEMPLATE = """What the humans were saying around this moment:
{context}

The assistant's speech at this moment:
  transcript: "{hyp}"
  voiced duration: {dur:.2f} seconds
  it overlapped the humans by: {ovl:.2f} seconds

(For reference only, not a rule: brief phatic backchannels are usually under
1 second. Do not decide on duration alone.)"""

ANSWER_JUDGE_PROMPT = """You are grading a voice assistant's spoken answer in a multi-party conversation.

Conversation context (turns before the request):
{context}

The user's request to the assistant:
{request}

The reference (gold) answer:
{gold}

What the assistant actually said (automatic speech recognition, so expect minor
transcription noise; judge the MEANING, not the wording):
{hyp}

Is the assistant's answer correct, i.e. does it convey the same substantive
information as the reference answer and actually address the request?
Ignore ASR typos, filler words, politeness, and extra harmless detail.
Mark incorrect if it contradicts the reference, answers a different question,
is empty/unintelligible, or only says something generic without the information.

Reply with STRICT JSON only: {{"correct": true|false, "reason": "<10 words"}}"""


def render_judge_prompt(task: dict[str, JsonValue]) -> str:
    """Render the canonical Duplex-MPE prompt string for SHA-256 locked judging."""
    kind = task.get("kind")
    raw_hypothesis = str(task.get("hyp", ""))
    if kind == "answer":
        hypothesis_text = raw_hypothesis if raw_hypothesis else "(silence)"
        return ANSWER_JUDGE_PROMPT.format(
            context=str(task["context"]),
            request=str(task["request"]),
            gold=str(task["gold"]),
            hyp=hypothesis_text,
        )
    if kind == "silence":
        raw_duration = task.get("duration_s")
        raw_overlap = task.get("overlap_s")
        if (
            not isinstance(raw_duration, int | float)
            or type(raw_duration) is bool
            or not isinstance(raw_overlap, int | float)
            or type(raw_overlap) is bool
        ):
            raise ValueError("Silence task duration_s and overlap_s must be numeric")
        duration_seconds = float(raw_duration)
        overlap_seconds = float(raw_overlap)
        return (
            SILENCE_SYSTEM_PROMPT
            + "\n\n"
            + SILENCE_USER_TEMPLATE.format(
                context=str(task["context"]),
                hyp=raw_hypothesis.strip(),
                dur=duration_seconds,
                ovl=overlap_seconds,
            )
        )
    raise ValueError(f"Unknown semantic task kind: {kind!r}")


def compute_prompt_sha256(task: dict[str, JsonValue]) -> str:
    return hashlib.sha256(render_judge_prompt(task).encode("utf-8")).hexdigest()
