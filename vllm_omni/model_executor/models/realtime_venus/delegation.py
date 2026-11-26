# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Incremental parsing of Venus's natural-language delegation channel."""

from dataclasses import dataclass


@dataclass
class DelegateParser:
    """Recognize one complete request per model turn, across text deltas.

    This parser observes the text stream. Speech masking remains token-based
    in the shared MiniCPM Talker handoff, including for malformed spans.
    """

    pending: str = ""
    opened: bool = False
    completed_query: str | None = None
    completed_spans: int = 0
    failed: bool = False

    OPEN = "<delegate>"
    CLOSE = "</delegate>"
    MAX_QUERY_CHARS = 4000

    def feed(self, text: str, *, end_of_turn: bool = False) -> str | None:
        if self.failed:
            return None
        self.pending += text
        if not self.opened:
            start = self.pending.find(self.OPEN)
            if start < 0:
                # Retain only a possible partial opening tag, not the transcript.
                self.pending = next(
                    (
                        self.pending[-n:]
                        for n in range(len(self.OPEN) - 1, 0, -1)
                        if self.pending.endswith(self.OPEN[:n])
                    ),
                    "",
                )
                return None
            self.pending = self.pending[start + len(self.OPEN) :]
            self.opened = True
        end = self.pending.find(self.CLOSE)
        query = self.pending[:end] if end >= 0 else self.pending
        if self.OPEN in query:
            self._fail("nested delegate requests are invalid")
        # A partial closing tag may temporarily exceed the query limit.
        if len(query) > self.MAX_QUERY_CHARS + (len(self.CLOSE) - 1 if end < 0 else 0):
            self._fail("delegate request exceeds 4000 characters")
        if end < 0:
            if end_of_turn:
                self._fail("model turn ended before </delegate>")
            return None
        query = query.strip()
        if not query:
            self._fail("delegate request must not be empty")
        repeated = self.completed_query is not None
        if repeated and query != self.completed_query:
            self._fail("only one delegate request is allowed per model turn")
        self.pending = self.pending[end + len(self.CLOSE) :]
        self.opened = False
        self.completed_query = query
        self.completed_spans += 1
        # Validate any second span in the same delta before executing the first.
        self.feed("", end_of_turn=end_of_turn)
        # Native generation may repeat the same span while awaiting a backend
        # result. It is still this turn's one call, not another execution or a
        # protocol error that should interrupt its outstanding result.
        return None if repeated else query

    def _fail(self, message: str) -> None:
        self.failed = True
        self.pending = ""
        raise ValueError(message)
