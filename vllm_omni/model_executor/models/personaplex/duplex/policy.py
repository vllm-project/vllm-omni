# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""PersonaPlex frame/token contract (the model *policy*, no engine state).

This mirrors the ``minicpmo45`` model-package layout from PR #3907: the token
contract lives in ``policy.py``, the model-owned runtime in ``runtime.py``.
Everything here is a constant or a pure function of the checkpoint's lockstep
conventions; nothing touches weights, devices, or streaming state.

Frame layout per 80 ms tick: 17 token rows -- row 0 inner-monologue text,
rows 1-8 agent audio codebooks, rows 9-16 user audio codebooks.
"""

from __future__ import annotations

ZERO_TEXT_TOKEN = 3
# Mimi token constants for the hybrid system prompt (agent silence / user sine).
SILENCE_TOKENS = (948, 243, 1178, 546, 1736, 1030, 1978, 2008)
SINE_TOKENS = (430, 1268, 381, 1611, 1095, 1495, 56, 472)
AUDIO_SILENCE_FRAME_CNT = 6  # 0.5 s at 12.5 Hz


def wrap_with_system_tags(text: str) -> str:
    t = text.strip()
    return t if t.startswith("<system>") else f"<system> {t} <system>"


__all__ = [
    "AUDIO_SILENCE_FRAME_CNT",
    "SILENCE_TOKENS",
    "SINE_TOKENS",
    "ZERO_TEXT_TOKEN",
    "wrap_with_system_tags",
]
