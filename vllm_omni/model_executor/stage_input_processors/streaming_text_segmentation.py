# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Model-independent capacity-adaptive segmentation policy for streaming TTS text.

Sizes each text segment from the realized expansion ratio rho = A_k / n_k
(acoustic decode steps per text token) of the segment that just finished,
instead of a static chunk ramp (RFC #6496):

* ``observe_segment`` feeds rho of the finished segment into an EMA estimate.
* ``open_segment`` freezes the next segment's text-token budget as
  ``remaining_capacity / rho_hat``.
* ``select_cut`` picks the strongest punctuation cut point inside the fill
  window ``[min_fill_fraction * budget, budget]`` and hard-cuts at the
  ceiling otherwise. Every non-final segment fills at least
  ``floor(min_fill_fraction * budget)`` tokens, bounding the segment count.

The engine is model- and tokenizer-agnostic: it consumes a token sequence
whose punctuation is tokenized as its own token, and a scalar remaining
acoustic capacity supplied by the caller.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

# Punctuation strength ranks, strongest first.  Sentence-final punctuation is
# the safest cut: nothing legal continues the sentence after it.  Clause
# punctuation admits a continuation of the same sentence, and weak pauses
# only mark reading rhythm.  The policy accepts weaker ranks as the budget
# drains, and hard-cuts only at the ceiling.
_SENTENCE_FINAL = frozenset("。！？!?…")
_CLAUSE = frozenset("，；：,;:")
_WEAK_PAUSE = frozenset("、—")
_ALL_PUNCT = _SENTENCE_FINAL | _CLAUSE | _WEAK_PAUSE

CutReason = Literal["sentence_final", "clause", "weak_pause", "hard_cut", "end_of_text"]

# Minimum fill fraction of the frozen budget (alpha in RFC #6496). Cuts below
# it are rejected so short segments do not over-fragment the text.
_DEFAULT_MIN_FILL_FRACTION = 0.7


@dataclass(frozen=True)
class SegmentCut:
    """Decision for one text segment.

    Attributes:
        cut_index: Number of leading tokens consumed by this segment
            (exclusive end index into the token sequence).
        reason: Why this cut point was chosen.
        budget: The frozen text-token budget this segment opened with.
        fill_fraction: ``cut_index / budget``; >= ``min_fill_fraction`` for
            punctuation cuts and 1.0 for hard cuts.
        text_tokens: Number of text tokens in this segment (== cut_index).
    """

    cut_index: int
    reason: CutReason
    budget: int
    fill_fraction: float
    text_tokens: int


class CapacityAdaptiveSegmenter:
    """Sizes streaming TTS text segments from measured acoustic expansion.

    State is per-request: create one instance per streaming TTS request.
    Thread-safety is not provided; the caller owns the instance.
    """

    def __init__(
        self,
        *,
        ema_alpha: float = 0.3,
        min_fill_fraction: float = _DEFAULT_MIN_FILL_FRACTION,
        warmup_rho: float | None = None,
    ) -> None:
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError(f"ema_alpha must be in (0, 1], got {ema_alpha}")
        if not 0.0 < min_fill_fraction <= 1.0:
            raise ValueError(f"min_fill_fraction must be in (0, 1], got {min_fill_fraction}")
        if warmup_rho is not None and warmup_rho <= 0:
            raise ValueError(f"warmup_rho must be positive, got {warmup_rho}")
        self._ema_alpha = ema_alpha
        self._min_fill_fraction = min_fill_fraction
        self._warmup_rho = warmup_rho
        self._rho_hat: float | None = warmup_rho
        self._has_observation = False
        self._active_budget: int | None = None

    @property
    def rho_hat(self) -> float | None:
        """Current EMA estimate of acoustic decode steps per text token."""
        return self._rho_hat

    @property
    def active_budget(self) -> int | None:
        """Text-token budget frozen by the last ``open_segment`` call."""
        return self._active_budget

    def observe_segment(self, *, acoustic_steps: int, text_tokens: int) -> None:
        """Feed the realized rho = acoustic_steps / text_tokens of a finished segment.

        Updates the EMA estimate used by the *next* ``open_segment`` call;
        it never changes a budget already frozen by ``open_segment``.
        Degenerate observations (no text or no acoustic steps) are ignored.
        """
        if text_tokens <= 0 or acoustic_steps <= 0:
            return
        rho_k = acoustic_steps / text_tokens
        if not self._has_observation:
            # First real measurement seeds the estimate; the warmup value is
            # only a pre-data fallback and is not blended into the EMA.
            self._rho_hat = rho_k
            self._has_observation = True
        else:
            self._rho_hat = self._ema_alpha * rho_k + (1.0 - self._ema_alpha) * self._rho_hat

    def open_segment(
        self,
        *,
        remaining_capacity: int,
        max_text_tokens: int | None = None,
    ) -> int:
        """Freeze the text-token budget for the next segment.

        Budget is ``max(1, int(remaining_capacity / rho_hat))``; until the
        first observation it falls back to ``max_text_tokens`` (or 1) as a
        warmup default, mirroring the static ramp.  ``max_text_tokens``
        caps the budget (e.g. a stage-0 ``max_tokens`` ceiling).  The frozen
        budget is returned and remains fixed until the next call.
        """
        if remaining_capacity <= 0:
            raise ValueError(f"remaining_capacity must be positive, got {remaining_capacity}")
        if max_text_tokens is not None and max_text_tokens <= 0:
            raise ValueError(f"max_text_tokens must be positive, got {max_text_tokens}")
        if self._rho_hat is None:
            budget = max_text_tokens if max_text_tokens is not None else 1
        else:
            budget = max(1, int(remaining_capacity / self._rho_hat))
        if max_text_tokens is not None:
            budget = min(budget, max_text_tokens)
        self._active_budget = budget
        return budget

    def select_cut(self, tokens: Sequence[str]) -> SegmentCut:
        """Choose the cut point for the next segment from ``tokens``.

        The frozen budget delimits the acceptable window
        ``[alpha * budget, budget]``; inside it the rightmost sentence-final
        punctuation wins, then clause, then weak pause.  With no acceptable
        punctuation the segment hard-cuts at the budget ceiling, and when
        the text itself ends inside the window the cut lands at end of text.

        Punctuation must be tokenized as its own token: a token is classified
        only when *every* character of it is punctuation, so a committed
        ``e.g.`` token can never be mistaken for a sentence-final cut.
        """
        if self._active_budget is None:
            raise RuntimeError("open_segment() must be called before select_cut()")
        budget = self._active_budget
        window_start = max(0, int(self._min_fill_fraction * budget))
        window_end = min(budget, len(tokens))

        last_sentence_final = -1
        last_clause = -1
        last_weak = -1
        for index in range(window_start, window_end):
            token = tokens[index]
            if not token or any(ch not in _ALL_PUNCT for ch in token):
                continue
            ch = token[0]
            if ch in _SENTENCE_FINAL:
                last_sentence_final = index
            elif ch in _CLAUSE:
                last_clause = index
            elif ch in _WEAK_PAUSE:
                last_weak = index

        if last_sentence_final >= 0:
            cut, reason = last_sentence_final + 1, "sentence_final"
        elif last_clause >= 0:
            cut, reason = last_clause + 1, "clause"
        elif last_weak >= 0:
            cut, reason = last_weak + 1, "weak_pause"
        elif window_end == len(tokens) and window_end > 0:
            cut, reason = window_end, "end_of_text"
        else:
            cut, reason = window_end, "hard_cut"

        return SegmentCut(
            cut_index=cut,
            reason=reason,
            budget=budget,
            fill_fraction=cut / budget,
            text_tokens=cut,
        )
