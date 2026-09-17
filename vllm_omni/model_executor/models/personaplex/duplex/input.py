# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""PCM input framing for PersonaPlex: 24 kHz ``pcm_f32le`` into 1920-sample (80 ms) units."""

from __future__ import annotations

from vllm_omni.model_executor.common.duplex.pcm_buffer import (
    FixedFramePcmAppendBuffer,
    FixedFramePcmAppendReservation,
)
from vllm_omni.model_executor.models.personaplex.duplex.config import (
    CHUNK_PERIOD_MS,
    FRAME_SIZE,
    SAMPLE_RATE,
)

PersonaPlexPcmAppendReservation = FixedFramePcmAppendReservation


class PersonaPlexPcmAppendBuffer(FixedFramePcmAppendBuffer):
    """Transactionally frame 24 kHz float PCM into PersonaPlex 80 ms units."""

    def __init__(self) -> None:
        super().__init__(
            sample_rate_hz=SAMPLE_RATE,
            frame_samples=FRAME_SIZE,
            chunk_period_ms=CHUNK_PERIOD_MS,
            model="PersonaPlex",
        )


__all__ = ["PersonaPlexPcmAppendBuffer", "PersonaPlexPcmAppendReservation"]
