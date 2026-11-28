# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Text conditioning of ABot-World 0.5 Iris: basic-action captions and skills.

Iris conditions on 512 text tokens split into a 384-token global row (the
scene prompt) and a 128-token local row. The local row holds an active
skill's prompt, else the caption of the held or tapped basic-action keys.
The captions, priorities and pulse lengths follow the upstream release
(amap-cvlab/ABot-World ``abot_world/iris/utils/action_conditions.py`` and
``abot_world/common/examples.py`` at d4a4f2d7).
"""

from __future__ import annotations

from collections.abc import Set
from dataclasses import dataclass, field

IRIS_SKILL_CHUNKS = 4

IRIS_ACTION_CAPTIONS: dict[str, str] = {
    "ctrl": "The character bends their knees, lowers their body, and maintains a stable crouched posture close to "
    "the ground",
    "shift": "The character leans forward, accelerates into a fast sprint, and moves quickly through the scene",
    "space": "The character bends their knees, pushes off the ground, rises into the air, and lands back on their feet",
    "mouseright": "The character raises the equipped weapon, carefully lines it up with the target, and maintains a "
    "steady aiming posture",
    "mouseleft": "The character readies the equipped weapon, directs it toward the target, and performs a forceful "
    "attack",
    "alt": "The character lowers their body, tucks into a roll, rolls across the ground, and returns to a stable "
    "standing posture",
}
# Combinations are matched before single keys, in this order.
_COMBINATION_CAPTIONS: dict[tuple[str, str], str] = {
    ("shift", "space"): "The character sprints forward, pushes off the ground into a running jump, travels through "
    "the air, and lands back on their feet",
    ("mouseright", "mouseleft"): "The character keeps the equipped weapon aimed at the target and performs an attack "
    "while maintaining the aiming posture",
    ("ctrl", "shift"): "The character stays low in a crouched posture and moves quickly forward with bent knees, "
    "keeping their body close to the ground",
}
_PULSE_CHUNKS = {"alt": 2, "mouseleft": 2, "space": 1}
_PRIORITY = ("alt", "space", "mouseleft", "mouseright", "ctrl", "shift")
IRIS_ACTION_KEYS = tuple(IRIS_ACTION_CAPTIONS)
_KEYS = frozenset(IRIS_ACTION_KEYS)


def _select(keys: Set[str]) -> tuple[str, ...]:
    for combination in _COMBINATION_CAPTIONS:
        if keys.issuperset(combination):
            return combination
    return next(((key,) for key in _PRIORITY if key in keys), ())


@dataclass
class IrisActionCaptions:
    """Basic-action caption state of one request, stepped once per chunk.

    Held keys caption every chunk they stay held. A tap of Space, Alt or
    MouseLeft starts a pulse that captions a fixed number of chunks, and any
    new tap interrupts the running pulse.
    """

    selected: tuple[str, ...] = ()
    pulse: str | None = None
    remaining: int = 0

    def step(self, held: Set[str], tapped: Set[str]) -> tuple[tuple[str, ...], str]:
        """Return the next chunk's action keys and caption; empty when none applies."""
        held = held & _KEYS
        tapped = tapped & _KEYS
        sustained = held - _PULSE_CHUNKS.keys()
        if tapped:
            self.pulse, self.remaining = None, 0
            self.selected = next(
                (
                    combination
                    for combination in _COMBINATION_CAPTIONS
                    if set(combination) <= sustained | tapped and tapped.intersection(combination)
                ),
                _select(tapped),
            )
        candidates = set(sustained) if not tapped and not self.remaining else sustained & set(self.selected)
        if tapped:
            candidates.update(self.selected)
        if self.remaining and self.pulse is not None:
            candidates.add(self.pulse)
        action = _select(candidates)
        pulse = next((key for key in action if key in _PULSE_CHUNKS), None)
        if pulse is not None and (pulse != self.pulse or pulse in tapped):
            self.remaining = _PULSE_CHUNKS[pulse]
        self.pulse = pulse
        self.remaining = max(0, self.remaining - 1) if pulse else 0
        if not action:
            return (), ""
        if len(action) == 1:
            return action, IRIS_ACTION_CAPTIONS[action[0]]
        return action, _COMBINATION_CAPTIONS[(action[0], action[1])]


@dataclass
class IrisText:
    """The text rows of one Iris request.

    ``global_prompt`` and ``local_prompt`` are the texts encoded in the
    current prompt embeddings. ``skill_in_chunk`` records whether the chunk
    being generated shows a skill, so finishing it consumes one skill chunk.
    """

    global_prompt: str
    local_prompt: str = ""
    captions: IrisActionCaptions = field(default_factory=IrisActionCaptions)
    skill_in_chunk: bool = False
