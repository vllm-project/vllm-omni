# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared sentence-TTS payload the orchestrator forwards to Talker."""

from __future__ import annotations


class SentenceTtsOutput:
    """Stage view that exposes only the sentence Talker should speak."""

    def __init__(self, request_id: str, text: str) -> None:
        self.request_id = request_id
        self.finished = False
        self.text = text
        self.cumulative_text = text
        self.outputs = [self]
        self.token_ids: list[int] = []


__all__ = ["SentenceTtsOutput"]
