# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Online capacity-adaptive segmentation for streaming TTS text (CAPS).

RFC #6496 mechanism 2.  Speech spends acoustic capacity slowly while text
arrives densely, so each released text segment is sized from the *realized*
expansion ratio ``rho = acoustic_steps / text_tokens`` of the segment that just
finished, instead of a static chunk ramp.

The mechanism is **online**: text arrives token by token from an upstream
stage, so the cut decision is made as each token arrives and never depends on
the complete text::

    segmenter = CapacityAdaptiveSegmenter(warmup_expansion_ratio=warmup)
    segmenter.start_segment(remaining_capacity=budget)
    while token := next_token():
        cut = segmenter.append_token(token)   # None while the segment stays open
    final = segmenter.finish()                # forced split of the tail

Thresholds are frozen when a segment opens and are **per punctuation level**,
progressively tightening: a sentence-final boundary may cut once
``level1_capacity_ratio`` of the capacity is filled, a clause boundary waits
for ``level2_capacity_ratio`` and a weak pause for
``level3_capacity_ratio``.  Nothing cuts before its own level threshold.  The
only other exits are ``force_split_at`` (the hard capacity ceiling) and the
forced split performed by :meth:`CapacityAdaptiveSegmenter.finish`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Punctuation tiers (exclusive; the strongest matching terminal wins).
# Mirrors the CAPS reference tiers: L1 sentence-final, L2 clause, L3 weak pause.
# ---------------------------------------------------------------------------
_LEVEL1_PUNCTUATION: tuple[str, ...] = ("。", "！", "？", ".", "!", "?")
_LEVEL2_PUNCTUATION: tuple[str, ...] = ("，", "、", "；", "：", ",", ";", ":")
_LEVEL3_PUNCTUATION: tuple[str, ...] = ("\n", "\r", "\t", "\f", "\v", "…", "……", "—", "——")
_LEVEL3_BREAK_WHITESPACE: frozenset[str] = frozenset(p for p in _LEVEL3_PUNCTUATION if p.isspace())
# Closing quotes/brackets may follow a boundary punctuation; they are skipped
# when locating a token's right boundary ("done." followed by '"').
_TRAILING_CLOSERS: frozenset[str] = frozenset("\"')]}）】》」』”’")

# Default frozen-threshold ratios (progressively tightening).
DEFAULT_LEVEL1_CAPACITY_RATIO = 0.70
DEFAULT_LEVEL2_CAPACITY_RATIO = 0.80
DEFAULT_LEVEL3_CAPACITY_RATIO = 0.90

# Acoustic steps reserved before converting remaining capacity into a
# text-token capacity, so a segment never plans to consume the whole budget.
DEFAULT_SAFETY_MARGIN = 8

# EMA weight applied to a newly observed expansion ratio.
DEFAULT_EMA_ALPHA = 0.3

# Punctuation levels.  A segment may open at level 1, 2 or 3; 0 means "no
# boundary punctuation", used for forced cuts and the final split.
_NO_PUNCTUATION = 0
_LEVEL1 = 1
_LEVEL2 = 2
_LEVEL3 = 3
_PUNCTUATION_LEVELS = (_LEVEL1, _LEVEL2, _LEVEL3)


def _exclusive_terminals(tier: tuple[str, ...], *lower_tiers: tuple[str, ...]) -> tuple[str, ...]:
    """Return the terminals unique to ``tier``, longest first.

    Levels are exclusive so a token classifies once: ``"……"`` belongs to L3
    only, even though the raw L3 list also contains ``"…"``.
    """
    lower = set().union(*lower_tiers) if lower_tiers else set()
    unique = {punct for punct in tier if punct not in lower}
    return tuple(sorted(unique, key=len, reverse=True))


def _build_terminals_by_level() -> tuple[tuple[int, tuple[str, ...]], ...]:
    level3 = _exclusive_terminals(_LEVEL3_PUNCTUATION, _LEVEL2_PUNCTUATION)
    level2 = _exclusive_terminals(_LEVEL2_PUNCTUATION, _LEVEL1_PUNCTUATION)
    level1 = _exclusive_terminals(_LEVEL1_PUNCTUATION)
    # Weakest first, so a multi-character boundary is not shadowed by a prefix.
    return ((_LEVEL3, level3), (_LEVEL2, level2), (_LEVEL1, level1))


_TERMINALS_BY_LEVEL = _build_terminals_by_level()


def classify_punctuation_level(token: str) -> int:
    """Return the punctuation level at ``token``'s right boundary.

    Text arrives from the upstream tokenizer, so a boundary punctuation is
    normally attached to the last word (``"years."``, ``"好好，"``).  Only the
    token's trailing boundary is classified: trailing whitespace and closing
    quotes/brackets are skipped, then the remaining suffix is matched against
    the tiers.  Returns 0 when the token carries no boundary punctuation.
    """
    if not token:
        return _NO_PUNCTUATION

    index = len(token) - 1
    saw_level3_break = False
    while index >= 0:
        char = token[index]
        if char.isspace():
            if char in _LEVEL3_BREAK_WHITESPACE:
                saw_level3_break = True
            index -= 1
            continue
        if char in _TRAILING_CLOSERS:
            index -= 1
            continue
        break

    if index < 0:
        return _LEVEL3 if saw_level3_break else _NO_PUNCTUATION

    boundary = token[: index + 1]
    for level, terminals in _TERMINALS_BY_LEVEL:
        if boundary.endswith(terminals):
            return level
    return _LEVEL3 if saw_level3_break else _NO_PUNCTUATION


def _validate_capacity_ratios(ratios: tuple[float, float, float]) -> None:
    if not all(math.isfinite(ratio) for ratio in ratios):
        raise ValueError("level capacity ratios must be finite")
    if not 0.0 < ratios[0] <= ratios[1] <= ratios[2] <= 1.0:
        raise ValueError("level capacity ratios must satisfy 0 < level1 <= level2 <= level3 <= 1")


@dataclass(frozen=True)
class SplitThresholds:
    """Frozen per-segment thresholds, all measured in text tokens.

    ``min_tokens_level1`` <= ``min_tokens_level2`` <= ``min_tokens_level3`` <=
    ``force_split_at``.  A boundary cuts only once the open segment has reached
    its own level's threshold; ``force_split_at`` cuts regardless of boundary.
    """

    min_tokens_level1: int
    min_tokens_level2: int
    min_tokens_level3: int
    force_split_at: int

    def min_tokens_for_level(self, level: int) -> int:
        """Threshold for a punctuation ``level`` (1, 2 or 3)."""
        if level == _LEVEL1:
            return self.min_tokens_level1
        if level == _LEVEL2:
            return self.min_tokens_level2
        if level == _LEVEL3:
            return self.min_tokens_level3
        raise ValueError(f"punctuation level must be one of {_PUNCTUATION_LEVELS}, got {level}")


@dataclass(frozen=True)
class SegmentCut:
    """One committed segment."""

    text_tokens: int
    punct_level: int
    is_forced: bool


def derive_text_token_capacity(
    *,
    remaining_capacity: int,
    expansion_ratio: float,
    safety_margin: int = DEFAULT_SAFETY_MARGIN,
    max_text_tokens: int | None = None,
) -> int:
    """Convert remaining acoustic capacity into a text-token capacity.

    ``capacity = floor((remaining_capacity - safety_margin) / expansion_ratio)``,
    capped by ``max_text_tokens`` when given.  ``max_text_tokens`` only lowers
    the capacity, never bypasses it, so a caller-supplied ceiling cannot plan a
    segment whose predicted acoustic cost exceeds the remaining budget.
    """
    if remaining_capacity <= 0:
        raise ValueError(f"remaining_capacity must be positive, got {remaining_capacity}")
    if safety_margin < 0:
        raise ValueError(f"safety_margin must be nonnegative, got {safety_margin}")
    if not math.isfinite(expansion_ratio) or expansion_ratio < 1.0:
        raise ValueError(f"expansion_ratio must be finite and >= 1.0, got {expansion_ratio}")
    if max_text_tokens is not None and max_text_tokens <= 0:
        raise ValueError(f"max_text_tokens must be positive, got {max_text_tokens}")

    capacity = math.floor((remaining_capacity - safety_margin) / expansion_ratio)
    if capacity < 1:
        raise ValueError(
            "remaining capacity cannot fit one predicted text token after the safety margin: "
            f"remaining_capacity={remaining_capacity}, safety_margin={safety_margin}, "
            f"expansion_ratio={expansion_ratio}"
        )
    if max_text_tokens is None:
        return capacity
    return min(capacity, max_text_tokens)


def compute_thresholds(
    *,
    text_token_capacity: int,
    level1_capacity_ratio: float = DEFAULT_LEVEL1_CAPACITY_RATIO,
    level2_capacity_ratio: float = DEFAULT_LEVEL2_CAPACITY_RATIO,
    level3_capacity_ratio: float = DEFAULT_LEVEL3_CAPACITY_RATIO,
) -> SplitThresholds:
    """Freeze one segment's thresholds from its text-token capacity.

    Each level's threshold is ``ceil(capacity * ratio)``, so stronger boundaries
    cut earlier and weaker ones require more accumulated text.  The ratios must
    satisfy ``0 < level1 <= level2 <= level3 <= 1`` and the ceiling is exactly
    the capacity, so tier spacing can never enlarge the budget.
    """
    if text_token_capacity < 1:
        raise ValueError(f"text_token_capacity must be positive, got {text_token_capacity}")
    ratios = (level1_capacity_ratio, level2_capacity_ratio, level3_capacity_ratio)
    _validate_capacity_ratios(ratios)

    return SplitThresholds(
        min_tokens_level1=math.ceil(text_token_capacity * level1_capacity_ratio),
        min_tokens_level2=math.ceil(text_token_capacity * level2_capacity_ratio),
        min_tokens_level3=math.ceil(text_token_capacity * level3_capacity_ratio),
        force_split_at=text_token_capacity,
    )


class CapacityAdaptiveSegmenter:
    """Online CAPS state machine for a single streaming TTS request.

    One instance per request; the owning stage drives it from one task, so no
    locking is provided.
    """

    def __init__(
        self,
        *,
        ema_alpha: float = DEFAULT_EMA_ALPHA,
        safety_margin: int = DEFAULT_SAFETY_MARGIN,
        level1_capacity_ratio: float = DEFAULT_LEVEL1_CAPACITY_RATIO,
        level2_capacity_ratio: float = DEFAULT_LEVEL2_CAPACITY_RATIO,
        level3_capacity_ratio: float = DEFAULT_LEVEL3_CAPACITY_RATIO,
        warmup_expansion_ratio: float | None = None,
    ) -> None:
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError(f"ema_alpha must be in (0, 1], got {ema_alpha}")
        if safety_margin < 0:
            raise ValueError(f"safety_margin must be nonnegative, got {safety_margin}")
        if warmup_expansion_ratio is not None and (
            not math.isfinite(warmup_expansion_ratio) or warmup_expansion_ratio < 1.0
        ):
            raise ValueError(f"warmup_expansion_ratio must be finite and >= 1.0, got {warmup_expansion_ratio}")

        ratios = (level1_capacity_ratio, level2_capacity_ratio, level3_capacity_ratio)
        _validate_capacity_ratios(ratios)

        self._ema_alpha = ema_alpha
        self._safety_margin = safety_margin
        self._capacity_ratios = ratios
        self._expansion_ratio = warmup_expansion_ratio
        self._has_observation = False
        self._thresholds: SplitThresholds | None = None
        self._token_count = 0

    @property
    def expansion_ratio(self) -> float | None:
        """Estimated acoustic steps per text token, or None before warmup."""
        return self._expansion_ratio

    @property
    def thresholds(self) -> SplitThresholds | None:
        """Thresholds frozen by the most recent :meth:`start_segment`."""
        return self._thresholds

    @property
    def token_count(self) -> int:
        """Text tokens accumulated in the segment currently open."""
        return self._token_count

    def observe_segment(self, *, acoustic_steps: int, text_tokens: int) -> None:
        """Feed the realized ``rho`` of a finished segment into the estimate.

        The estimate only affects segments opened afterwards; a frozen segment
        keeps its thresholds.  Degenerate observations are ignored so a
        truncated or empty segment cannot poison the estimate.
        """
        if text_tokens <= 0 or acoustic_steps <= 0:
            return
        observed_ratio = acoustic_steps / text_tokens
        if not self._has_observation:
            # The first real measurement seeds the estimate; a warmup value is
            # only a pre-data fallback and is not blended into the EMA.
            self._expansion_ratio = observed_ratio
            self._has_observation = True
            return
        assert self._expansion_ratio is not None
        self._expansion_ratio = self._ema_alpha * observed_ratio + (1.0 - self._ema_alpha) * self._expansion_ratio

    def start_segment(self, *, remaining_capacity: int, max_text_tokens: int | None = None) -> SplitThresholds:
        """Open a segment and freeze its thresholds.

        Needs an expansion-ratio estimate: either the constructor's warmup value
        or at least one :meth:`observe_segment` call.  Raises ``ValueError`` when
        the remaining capacity cannot fit a single predicted text token.
        """
        if self._expansion_ratio is None:
            raise RuntimeError(
                "no expansion-ratio estimate available; pass warmup_expansion_ratio "
                "or call observe_segment() before start_segment()"
            )
        capacity = derive_text_token_capacity(
            remaining_capacity=remaining_capacity,
            expansion_ratio=self._expansion_ratio,
            safety_margin=self._safety_margin,
            max_text_tokens=max_text_tokens,
        )
        level1_ratio, level2_ratio, level3_ratio = self._capacity_ratios
        self._thresholds = compute_thresholds(
            text_token_capacity=capacity,
            level1_capacity_ratio=level1_ratio,
            level2_capacity_ratio=level2_ratio,
            level3_capacity_ratio=level3_ratio,
        )
        self._token_count = 0
        return self._thresholds

    def append_token(self, token: str) -> SegmentCut | None:
        """Append one streaming token, returning a cut when it commits.

        The decision uses the token count *including* the token just appended,
        which is what the frozen thresholds are expressed in.  A boundary cuts
        only once its own level threshold is reached; the ceiling cuts
        regardless of boundary.  Returns ``None`` while the segment stays open.

        A committed cut opens the next segment with the same frozen thresholds;
        call :meth:`start_segment` again to refresh them from updated capacity
        and expansion-ratio estimates.
        """
        thresholds = self._thresholds
        if thresholds is None:
            raise RuntimeError("start_segment() must be called before append_token()")

        self._token_count += 1
        punct_level = classify_punctuation_level(token)
        if self._token_count >= thresholds.force_split_at:
            return self._commit(punct_level=punct_level, is_forced=True)
        if punct_level != _NO_PUNCTUATION and self._token_count >= thresholds.min_tokens_for_level(punct_level):
            return self._commit(punct_level=punct_level, is_forced=False)
        return None

    def append_tokens(self, tokens: Sequence[str]) -> list[SegmentCut]:
        """Append tokens in arrival order and collect the cuts they commit."""
        cuts: list[SegmentCut] = []
        for token in tokens:
            cut = self.append_token(token)
            if cut is not None:
                cuts.append(cut)
        return cuts

    def finish(self) -> SegmentCut | None:
        """Force the final split of the tail, or ``None`` when it is empty."""
        if self._token_count <= 0:
            return None
        return self._commit(punct_level=_NO_PUNCTUATION, is_forced=True)

    def _commit(self, *, punct_level: int, is_forced: bool) -> SegmentCut:
        cut = SegmentCut(text_tokens=self._token_count, punct_level=punct_level, is_forced=is_forced)
        self._token_count = 0
        return cut
