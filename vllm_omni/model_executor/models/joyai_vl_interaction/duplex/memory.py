# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bounded visual evidence and summaries, plus committed conversation memory."""

from dataclasses import dataclass

from vllm_omni.engine.duplex.plugin import duplex_history_digest

RECENT_MESSAGES = 8
SUMMARY_MAX_TOKENS = 512
SUMMARY_CHUNK_MESSAGES = 8


@dataclass(frozen=True)
class TextSummaryProgress:
    """Private fold of an oversized exchange; offsets address its transcript."""

    history_length: int
    history_digest: str
    offset: int
    text: str

    def matches(self, history: tuple[tuple[str, str], ...]) -> bool:
        return (
            len(history) >= self.history_length
            and duplex_history_digest(history[: self.history_length]) == self.history_digest
        )


@dataclass(frozen=True)
class TextSummaryChunk:
    previous: TextSummaryProgress | None
    end_offset: int
    complete: bool


@dataclass(frozen=True)
class TextSummaryLimit:
    """A rejected prefix's smaller raw-transcript span, retained across folds.

    Whole speaker labels plus one evidence character form the minimum
    excerpt; the model token budget remains a hard limit.
    """

    history_length: int
    history_digest: str
    max_chars: int

    def matches(self, history: tuple[tuple[str, str], ...]) -> bool:
        return (
            len(history) >= self.history_length
            and duplex_history_digest(history[: self.history_length]) == self.history_digest
        )


@dataclass
class JoyAISummaryMemory:
    text: str = ""
    covered_messages: int = 0
    covered_digest: str = ""
    partial: TextSummaryProgress | None = None
    excerpt_limit: TextSummaryLimit | None = None

    def matches(self, history: tuple[tuple[str, str], ...]) -> bool:
        return not self.covered_messages or (
            len(history) >= self.covered_messages
            and duplex_history_digest(history[: self.covered_messages]) == self.covered_digest
        )

    def validate(self, history: tuple[tuple[str, str], ...]) -> None:
        """A playback correction or conversation edit revokes derived memory."""
        if not self.matches(history):
            self.text = ""
            self.covered_messages = 0
            self.covered_digest = ""
            self.partial = None
            self.excerpt_limit = None
        if self.partial is not None and not self.partial.matches(history):
            self.partial = None
        if self.excerpt_limit is not None and not self.excerpt_limit.matches(history):
            self.excerpt_limit = None
