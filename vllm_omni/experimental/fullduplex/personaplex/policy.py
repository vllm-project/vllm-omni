# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PersonaPlex frame/token contract (the model *policy*, no engine state).

This mirrors the ``minicpmo45`` model-package layout from PR #3907: the token
contract lives in ``policy.py``, the model-owned runtime in ``runtime.py``.
Everything here is a constant or a pure function of the checkpoint's lockstep
conventions; nothing touches weights, devices, or streaming state.

Frame layout per 80 ms tick: 17 token rows -- row 0 inner-monologue text,
rows 1-8 agent audio codebooks, rows 9-16 user audio codebooks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Inner-monologue token ids that carry no displayable text (EPAD, PAD).
NON_TEXT_TOKEN_IDS = frozenset({0, 3})
ZERO_TEXT_TOKEN = 3
# Mimi token constants for the hybrid system prompt (agent silence / user sine).
SILENCE_TOKENS = (948, 243, 1178, 546, 1736, 1030, 1978, 2008)
SINE_TOKENS = (430, 1268, 381, 1611, 1095, 1495, 56, 472)
# LM initial ("beginning of stream") token ids: audio rows use the codec
# cardinality, the text row uses the text vocabulary size.
AUDIO_INITIAL_TOKEN = 2048
TEXT_INITIAL_TOKEN = 32000
TEXT_VOCAB_SIZE = 32000
AUDIO_SILENCE_FRAME_CNT = 6  # 0.5 s at 12.5 Hz


@dataclass(frozen=True)
class PrefillStep:
    """One tick of a recycled slot's system-prompt replay (see batched serving)."""

    kind: str  # "sacrifice" | "voice" | "voice_end" | "tokens"
    embedding: Any = None  # [1, 1, H] tensor for kind == "voice"
    moshi_tokens: Any = None  # [8] tensor for kind == "tokens"
    text_token: int | None = None
    user_sine: bool = False


def wrap_with_system_tags(text: str) -> str:
    t = text.strip()
    return t if t.startswith("<system>") else f"<system> {t} <system>"


__all__ = [
    "AUDIO_INITIAL_TOKEN",
    "AUDIO_SILENCE_FRAME_CNT",
    "NON_TEXT_TOKEN_IDS",
    "PrefillStep",
    "SILENCE_TOKENS",
    "SINE_TOKENS",
    "TEXT_INITIAL_TOKEN",
    "TEXT_VOCAB_SIZE",
    "ZERO_TEXT_TOKEN",
    "wrap_with_system_tags",
]
