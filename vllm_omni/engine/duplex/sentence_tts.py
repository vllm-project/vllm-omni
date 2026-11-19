# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared sentence-TTS payload and growing-text prefix math.

The orchestrator forwards ``SentenceTtsOutput`` to Talker. JoyAI and AURA both
need the unseen suffix of a cumulative string; each model still owns its own
splitter, pending buffer, and emission policy.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GrowingTextDelta:
    """Unseen suffix of a cumulative string, and whether the prefix restarted."""

    seen: str
    tail: str
    restarted: bool


def consume_growing_text(prefix: str, text: str) -> GrowingTextDelta:
    """Take the unseen suffix of ``text`` relative to ``prefix``.

    If ``text`` still starts with ``prefix``, ``tail`` is the extension and
    ``restarted`` is false. Otherwise the whole string is new: ``tail`` is
    ``text`` and ``restarted`` is true so the caller can drop pending units.
    """
    if text.startswith(prefix):
        return GrowingTextDelta(seen=text, tail=text[len(prefix) :], restarted=False)
    return GrowingTextDelta(seen=text, tail=text, restarted=True)


class SentenceTtsOutput:
    """Stage view that exposes only the sentence Talker should speak."""

    def __init__(self, request_id: str, text: str) -> None:
        self.request_id = request_id
        self.finished = False
        self.text = text
        self.cumulative_text = text
        self.outputs = [self]
        self.token_ids: list[int] = []


__all__ = ["GrowingTextDelta", "SentenceTtsOutput", "consume_growing_text"]
