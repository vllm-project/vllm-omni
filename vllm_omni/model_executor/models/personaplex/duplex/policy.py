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

from vllm_omni.model_executor.models.personaplex.prompting import (
    AUDIO_SILENCE_FRAME_CNT,
    SILENCE_TOKENS,
    SINE_TOKENS,
    ZERO_TEXT_TOKEN,
    wrap_with_system_tags,
)

__all__ = [
    "AUDIO_SILENCE_FRAME_CNT",
    "SILENCE_TOKENS",
    "SINE_TOKENS",
    "ZERO_TEXT_TOKEN",
    "wrap_with_system_tags",
]
