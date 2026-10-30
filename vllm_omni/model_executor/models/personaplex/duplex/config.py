# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Constants of the PersonaPlex full-duplex integration.

PersonaPlex (``nvidia/personaplex-7b-v1``) is a Moshi finetune: a pure-lockstep
full-duplex speech-to-speech model running at the Mimi codec frame rate
(12.5 Hz / 80 ms). The values are fixed by the pretrained checkpoint; they
live here so the rest of the package never hard-codes them.
"""

from __future__ import annotations

# Mimi codec constants (loaders.py: SAMPLE_RATE / FRAME_RATE).
SAMPLE_RATE = 24000
FRAME_RATE = 12.5
FRAME_SIZE = int(SAMPLE_RATE / FRAME_RATE)  # 1920 samples per 80 ms frame
CHUNK_PERIOD_MS = int(round(1000 / FRAME_RATE))  # 80

# Default assistant persona shipped with PersonaPlex (offline.py:335).
DEFAULT_PERSONA = "You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way."
# Bundled voice prompt used when a session names none.
DEFAULT_VOICE = "NATF2.pt"

__all__ = ["CHUNK_PERIOD_MS", "DEFAULT_PERSONA", "DEFAULT_VOICE", "FRAME_RATE", "FRAME_SIZE", "SAMPLE_RATE"]
