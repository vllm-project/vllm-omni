# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Turn-taking policy of the Sophon-Realtime duplex session: when to ask the turn question, how to read the answer,
and how replies are drafted, merged, resumed and trimmed.

Framework-independent (no vllm / vllm_omni / torch imports). The defaults are the values the released models were
tuned with. A few of them can be changed per session through ``extra_body`` (see ``SESSION_OVERRIDES``); the rest are
fixed for the deployment.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class TurnPolicy:
    # After the user pauses, the turn question is asked at these silences (s); a reply starts when P(finished)
    # reaches the threshold of that checkpoint. The last checkpoint doubles as the fallback.
    checks: tuple[float, ...] = (0.2, 0.35, 0.5, 0.8, 1.3, 2.0)
    thresholds: tuple[float, ...] = (0.95, 0.85, 0.7, 0.5, 0.35, 0.25)
    fallback_s: float = 2.0  # reply anyway after this much silence
    speech_prob: float = 0.5  # frame speech probability above which a frame counts as speech
    ignore_short_on_fallback: bool = False  # on fallback, ignore very short back-channels instead of replying
    # Speculative draft: start writing the reply this early in a pause, held silent until the turn is judged finished.
    spec_at_s: float = 0.05  # <= 0 disables
    spec_min_speech_s: float = 0.3  # only once the user has spoken this long
    # Minimum pause before speaking once the model says "finished", applied only after this much user speech
    # (long turns breathe mid-sentence; short ones are answered at once).
    min_release_s: float = 0.0
    min_release_after_s: float = 1.0
    # The user speaks again within this window after a reply started: abort it and answer both utterances together.
    merge_window_s: float = 2.0
    # Back-channel over a reply that was cut off: continue the cut reply instead of answering the back-channel.
    bc_resume: bool = True
    bc_threshold: float = 0.5  # P(back-channel) needed (and above P(finished))
    bc_window_s: float = 4.0  # the cut must be at most this old
    # Noisy input (street, phone): while a segment keeps going, ask every periodic_s;
    # reply at P(finished) >= periodic_threshold.
    periodic_s: float = 0.0  # 0 disables
    periodic_min_speech_s: float = 1.0
    periodic_threshold: float = 0.9
    periodic_dynamic_range: float = (
        4.0  # peak / median frame RMS the segment needs (about 12 dB): speech, not steady noise
    )
    vad_min_statistics: bool = False  # robust noise floor (minimum statistics) for noisy callers
    # Replies.
    lead_trim_s: float = 1.0  # drop up to this much leading silence from each spoken reply
    repetition_guard: bool = False  # cut a reply that starts repeating itself
    repetition_penalty: float = 1.0
    history_turns: int = 8
    history_seconds: float = 120.0  # user audio kept in the history
    # Camera frames (vision build).
    frame_ttl_s: float = 3.0
    image_max_side: int = 448
    # Engine bookkeeping.
    reservation_timeout_s: float = 15.0  # an append not confirmed within this time is rolled back (a reply is resent)
    retry_max: int = 3  # resends of a refused reply append

    def __post_init__(self) -> None:
        if len(self.checks) != len(self.thresholds) or not self.checks:
            raise ValueError("checks and thresholds must be non-empty and of equal length")
        if any(b <= a for a, b in zip(self.checks, self.checks[1:])):
            raise ValueError("checks must be increasing")
        for name, v in ((n, getattr(self, n)) for n in _NON_NEGATIVE):
            if not (isinstance(v, (int, float)) and math.isfinite(v) and v >= 0):
                raise ValueError(f"{name} must be a finite number >= 0")
        for name in ("speech_prob", "bc_threshold", "periodic_threshold"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not all(0.0 <= t <= 1.0 for t in self.thresholds):
            raise ValueError("thresholds must be in [0, 1]")

    def with_session(self, extra_body: Mapping | None) -> TurnPolicy:
        """This policy with the per-session overrides from ``extra_body`` applied (unknown keys are ignored here;
        ``validate_session`` reports them)."""
        changes = {}
        for key, (attr, kind) in SESSION_OVERRIDES.items():
            if isinstance(extra_body, Mapping) and extra_body.get(key) is not None:
                changes[attr] = kind(extra_body[key])
        return replace(self, **changes) if changes else self


_NON_NEGATIVE = (
    "fallback_s",
    "spec_min_speech_s",
    "min_release_s",
    "min_release_after_s",
    "merge_window_s",
    "bc_window_s",
    "periodic_s",
    "periodic_min_speech_s",
    "periodic_dynamic_range",
    "lead_trim_s",
    "repetition_penalty",
    "history_turns",
    "history_seconds",
    "frame_ttl_s",
    "image_max_side",
    "reservation_timeout_s",
    "retry_max",
)

# extra_body key -> (TurnPolicy field, type). Fixed when the session is created.
SESSION_OVERRIDES: dict[str, tuple[str, type]] = {
    "turn_fallback_s": ("fallback_s", float),
    "min_release_s": ("min_release_s", float),
    "min_release_after_s": ("min_release_after_s", float),
}
SESSION_LIMITS: dict[str, tuple[float, float]] = {
    "turn_fallback_s": (0.3, 10.0),
    "min_release_s": (0.0, 5.0),
    "min_release_after_s": (0.0, 30.0),
}


def validate_session(extra_body: Mapping | None) -> str | None:
    """None if the per-session overrides are usable, else a message for the client."""
    if not isinstance(extra_body, Mapping):
        return None
    for key, (lo, hi) in SESSION_LIMITS.items():
        v = extra_body.get(key)
        if v is None:
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not lo <= v <= hi:
            return f"extra_body.{key} must be a number in [{lo}, {hi}]"
    return None


DEFAULT_POLICY = TurnPolicy()
__all__ = ["DEFAULT_POLICY", "SESSION_OVERRIDES", "TurnPolicy", "validate_session"]
