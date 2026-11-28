# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""StreamOV evidence scores for completed native duplex units."""

from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvidenceScorer:
    """Rank a unit against past units without inspecting future input."""

    previous_frame: bytes = b""
    visual_history: deque[float] = field(default_factory=lambda: deque(maxlen=64))
    audio_history: deque[float] = field(default_factory=lambda: deque(maxlen=64))

    @staticmethod
    def _rank(value: float, history: deque[float]) -> float:
        rank = sum(1.0 if item < value else 0.5 if item == value else 0.0 for item in history)
        result = rank / len(history) if history and value > 0 else 0.0
        history.append(value)
        return result

    def observe(self, frame: bytes, energy: float) -> tuple[float, float]:
        difference = 0.0
        if frame and len(frame) == len(self.previous_frame):
            difference = sum(abs(a - b) for a, b in zip(frame, self.previous_frame, strict=True)) / len(frame)
        self.previous_frame = frame
        return self._rank(difference, self.visual_history), self._rank(energy, self.audio_history)

    def score(self, frame: bytes, energy: float, onset: float) -> tuple[str, float]:
        visual, audio = self.observe(frame, energy)
        aligned = min(visual, audio, onset)
        return max(
            (("vision", visual - aligned), ("audio", audio - aligned), ("av", aligned)),
            key=lambda item: item[1],
        )


def select_evidence_units(
    units: list[dict[str, Any]], unit_budget: int, history_budget: int
) -> list[tuple[int, dict[str, Any]]]:
    """Keep a dense recent suffix and spend the remaining budget on evidence."""
    recent_budget = max(0, unit_budget - history_budget)
    split = len(units)
    while split and not units[split - 1].get("historical"):
        cost = units[split - 1]["length"]
        if cost > recent_budget:
            break
        recent_budget -= cost
        split -= 1

    available = min(history_budget, unit_budget)
    selected = []
    ranking = sorted(range(split), key=lambda index: (units[index].get("evidence_score", 0.0), index), reverse=True)
    for index in ranking:
        unit = units[index]
        if unit.get("evidence_score", 0.0) <= 0:
            break
        removed = unit.get("remove_media")
        length = unit["length"] - (removed[1] if removed else 0)
        if length <= available:
            retained = {**unit, "length": length, "historical": True}
            retained.pop("remove_media", None)
            selected.append((index, retained))
            available -= length
    return sorted(selected) + [(index, units[index]) for index in range(split, len(units))]
