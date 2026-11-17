# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""PersonaPlex output projection: per-step Code2Wav audio and inner-monologue text into deltas."""

from __future__ import annotations

from vllm_omni.model_executor.common.duplex.data_plane import CumulativeAudioTextDataPlane


class PersonaPlexDataPlaneSession(CumulativeAudioTextDataPlane):
    """Project staged PersonaPlex output into Realtime deltas.

    Code2Wav advertises its rate (``sr``) on every audio payload — see
    ``personaplex_code2wav.py`` — so the projection needs no default rate.
    """

    # The plugin runs Code2Wav with DELTA output (see configure_sampling_params).
    audio_output_is_delta = True


__all__ = ["PersonaPlexDataPlaneSession"]
