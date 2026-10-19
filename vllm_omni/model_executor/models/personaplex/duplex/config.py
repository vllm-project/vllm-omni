# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Constants of the PersonaPlex full-duplex integration (defined in ``personaplex.prompting``)."""

from __future__ import annotations

from vllm_omni.model_executor.models.personaplex.prompting import (
    CHUNK_PERIOD_MS,
    DEFAULT_PERSONA,
    DEFAULT_VOICE,
    FRAME_RATE,
    FRAME_SIZE,
    SAMPLE_RATE,
)

__all__ = ["CHUNK_PERIOD_MS", "DEFAULT_PERSONA", "DEFAULT_VOICE", "FRAME_RATE", "FRAME_SIZE", "SAMPLE_RATE"]
