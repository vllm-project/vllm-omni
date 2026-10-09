# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""PersonaPlex output projection: Code2Wav PCM deltas and cumulative text."""

from __future__ import annotations

import numpy as np

from vllm_omni.engine.duplex.plugin import DuplexDataPlaneContext
from vllm_omni.model_executor.common.duplex.data_plane import CumulativeAudioTextDataPlane, _RequestCursor
from vllm_omni.model_executor.common.request_outputs import (
    audio_sample_count,
    audio_value,
    multimodal_output,
    sample_rate_hz,
    text_delta,
    text_value,
    unwrap_request_output,
)
from vllm_omni.model_executor.models.personaplex.duplex.config import SAMPLE_RATE


class PersonaPlexDataPlaneSession(CumulativeAudioTextDataPlane):
    """Consume each Stage 1 PCM emission once, retaining only per-request cursors.

    Request lifetime, terminal state and context handling remain the framework
    implementation. Identical PCM emissions are distinct contributions, not
    cumulative snapshots or replay identifiers. Delivery/playback ledgers remain
    owned by the session runner, not by this projector.
    """

    default_sample_rate_hz = SAMPLE_RATE

    def _project_output(self, output: object, *, context: DuplexDataPlaneContext) -> dict[str, object] | None:
        output, completion = unwrap_request_output(output)
        request_id = getattr(output, "request_id", None)
        if not isinstance(request_id, str) or not request_id:
            request_id = None
        state = self._requests.setdefault(request_id, _RequestCursor()) if request_id is not None else _RequestCursor()
        multimodal = multimodal_output(output, completion)
        audio = audio_value(multimodal)
        if isinstance(audio, list):
            # Coalesce only this emission's deferred CPU chunks, never history.
            audio = (
                np.concatenate([np.asarray(chunk, dtype=np.float32).reshape(-1) for chunk in audio]) if audio else None
            )
        samples = audio_sample_count(audio) or 0
        text = text_value(multimodal, completion)
        delta_text = text_delta(text, state.text)
        rate = sample_rate_hz(multimodal, default=self.default_sample_rate_hz)
        # Empty PCM must not generate a header-only WAV event. Failed encoding
        # must consume neither the audio counter nor the transcript cursor.
        encoded = self._encode_audio(audio, rate, context.response_format, context.speed) if samples else None
        if samples and not encoded:
            raise RuntimeError("PersonaPlex could not encode a nonempty audio delta")
        state.audio_samples += samples
        if text:
            state.text = text
        if not encoded and not delta_text:
            return None
        return {
            "supported": True,
            "stage_role": self.stage_role,
            "is_listen": False,
            "data_plane_request_id": request_id,
            "text": delta_text,
            "audio_data": encoded or "",
            "audio_format": context.response_format,
            "sample_rate_hz": rate,
            "audio_duration_ms": round(samples * 1000 / max(1, rate)),
            "audio_sample_count": samples,
            "end_of_turn": False,
            "uses_model_runner_scheduler": self.uses_model_runner_scheduler,
            "runner_kv_backed": self.runner_kv_backed,
            "runtime_impl": self.runtime_impl,
            "owned_runtime": self.owned_runtime,
        }


__all__ = ["PersonaPlexDataPlaneSession"]
