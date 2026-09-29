# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""PersonaPlex output projection: cumulative Code2Wav audio and inner-monologue text into deltas."""

from __future__ import annotations

from vllm_omni.model_executor.common.duplex.data_plane import CumulativeAudioTextDataPlane
from vllm_omni.model_executor.models.personaplex.duplex.config import SAMPLE_RATE


class PersonaPlexDataPlaneSession(CumulativeAudioTextDataPlane):
    """Project cumulative staged PersonaPlex output into Realtime deltas (24 kHz by default)."""

    default_sample_rate_hz = SAMPLE_RATE


__all__ = ["PersonaPlexDataPlaneSession"]
