# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stereo 48 kHz delta data plane projection for Magenta RealTime 2."""

from __future__ import annotations

import numpy as np
import torch

from vllm_omni.engine.duplex.plugin import DuplexDataPlaneContext
from vllm_omni.model_executor.common.duplex.data_plane import (
    CumulativeAudioTextDataPlane,
    _RequestCursor,
)
from vllm_omni.model_executor.common.request_outputs import (
    audio_value,
    multimodal_output,
    sample_rate_hz,
    text_delta,
    text_value,
    unwrap_request_output,
)
from vllm_omni.model_executor.models.magenta_rt2.duplex.capabilities import SAMPLE_RATE


def _normalize_stereo_chunk(chunk: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Normalize a single delta audio chunk to 2D ``[samples, 2]`` when stereo or 1D mono."""
    if isinstance(chunk, torch.Tensor):
        if chunk.ndim == 2 and chunk.shape[0] == 2 and chunk.shape[1] != 2:
            return chunk.transpose(0, 1).contiguous()
        if chunk.ndim > 2:
            return chunk.reshape(-1, 2) if chunk.shape[-1] == 2 else chunk.reshape(-1)
        return chunk
    arr = np.asarray(chunk, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[0] == 2 and arr.shape[1] != 2:
        return np.ascontiguousarray(arr.T)
    if arr.ndim > 2:
        return arr.reshape(-1, 2) if arr.shape[-1] == 2 else arr.reshape(-1)
    return arr


def _join_stereo_delta_chunks(audio: object | None) -> object | None:
    if audio is None:
        return None
    if isinstance(audio, (list, tuple)):
        if not audio:
            return None
        normed = [_normalize_stereo_chunk(c) for c in audio]
        if all(isinstance(c, torch.Tensor) for c in normed):
            return torch.cat(normed, dim=0)  # type: ignore[arg-type]
        return np.concatenate([np.asarray(c, dtype=np.float32) for c in normed], axis=0)
    if isinstance(audio, (torch.Tensor, np.ndarray)):
        return _normalize_stereo_chunk(audio)
    return _normalize_stereo_chunk(np.asarray(audio, dtype=np.float32))


def _stereo_sample_frames(audio: object | None) -> int:
    if audio is None:
        return 0
    if isinstance(audio, torch.Tensor):
        if audio.numel() == 0:
            return 0
        return int(audio.shape[0])
    arr = np.asarray(audio)
    if arr.size == 0:
        return 0
    return int(arr.shape[0])


class MagentaRT2DataPlaneSession(CumulativeAudioTextDataPlane):
    """Project DELTA 48 kHz stereo ``SpectroStreamCode2Wav`` outputs for Realtime delivery."""

    default_sample_rate_hz = SAMPLE_RATE
    audio_output_is_delta = True

    def _project_output(self, output: object, *, context: DuplexDataPlaneContext) -> dict[str, object] | None:
        output, completion = unwrap_request_output(output)
        request_id = getattr(output, "request_id", None)
        if not isinstance(request_id, str) or not request_id:
            request_id = None
        state = self._requests.setdefault(request_id, _RequestCursor()) if request_id is not None else _RequestCursor()
        mm = multimodal_output(output, completion)
        raw_audio = audio_value(mm)
        audio_delta = _join_stereo_delta_chunks(raw_audio)
        delta_samples = _stereo_sample_frames(audio_delta)
        if delta_samples <= 0:
            audio_delta = None

        text = text_value(mm, completion)
        delta_text = text_delta(text, state.text)
        if text:
            state.text = text

        rate = sample_rate_hz(mm, default=self.default_sample_rate_hz)
        encoded = self._encode_audio(audio_delta, rate, context.response_format, context.speed)
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
            "audio_duration_ms": round(delta_samples * 1000 / max(1, rate)),
            "end_of_turn": False,
            "uses_model_runner_scheduler": self.uses_model_runner_scheduler,
            "runner_kv_backed": self.runner_kv_backed,
            "runtime_impl": self.runtime_impl,
            "owned_runtime": self.owned_runtime,
        }


__all__ = ["MagentaRT2DataPlaneSession"]
