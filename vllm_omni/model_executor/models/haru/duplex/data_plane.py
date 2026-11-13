# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from https://github.com/higurashi-ai/Haru (Apache-2.0), haru_vllm/haru_vllm/duplex/data_plane.py @ d3612b0.
"""Output projection: each frame's audio and text token into realtime deltas."""

from __future__ import annotations

import torch

from vllm_omni.engine.duplex.plugin import DuplexDataPlaneContext
from vllm_omni.model_executor.common.duplex.data_plane import CumulativeAudioTextDataPlane
from vllm_omni.model_executor.common.request_outputs import (
    audio_sample_count,
    audio_value,
    multimodal_output,
    sample_rate_hz,
    unwrap_request_output,
)
from vllm_omni.model_executor.models.haru.sampling import StreamDetokenizer


class HaruDataPlane(CumulativeAudioTextDataPlane):
    """Each output carries one new frame: its audio is passed on whole, its text token detokenized here."""

    def __init__(self, encode_audio) -> None:
        super().__init__(encode_audio)
        self.tokenizer = None
        self.silent_ids: set[int] = set()
        self._detokenizers: dict[str, StreamDetokenizer] = {}

    def close_stream(self, request_id: str) -> None:
        super().close_stream(request_id)
        self._detokenizers.pop(request_id, None)

    def close_session(self, session_id: str, *, active_request_id: str | None = None) -> None:
        super().close_session(session_id, active_request_id=active_request_id)
        for request_id in [r for r in self._detokenizers if r not in self._requests]:
            del self._detokenizers[request_id]

    def _project_output(self, output: object, *, context: DuplexDataPlaneContext) -> dict[str, object] | None:
        request_output, completion = unwrap_request_output(output)
        request_id = getattr(request_output, "request_id", None)
        multimodal = multimodal_output(request_output, completion)
        audio = audio_value(multimodal)
        if isinstance(audio, (list, tuple)):  # frames of more than one engine step
            audio = torch.cat([torch.as_tensor(chunk).reshape(-1) for chunk in audio]) if audio else None
        samples = audio_sample_count(audio) or 0
        rate = sample_rate_hz(multimodal, default=self.default_sample_rate_hz)
        encoded = self._encode_audio(audio, rate, context.response_format, context.speed) if samples else None
        text = ""
        if self.tokenizer is not None and isinstance(request_id, str):
            detokenizer = self._detokenizers.get(request_id)
            if detokenizer is None:
                detokenizer = self._detokenizers[request_id] = StreamDetokenizer(self.tokenizer, self.silent_ids)
            for token in getattr(completion, "token_ids", None) or ():
                text += detokenizer.push(int(token))
        if not encoded and not text:
            return None
        return {
            "supported": True,
            "stage_role": self.stage_role,
            "is_listen": False,
            "data_plane_request_id": request_id,
            "text": text,
            "audio_data": encoded or "",
            "audio_format": context.response_format,
            "sample_rate_hz": rate,
            "audio_duration_ms": round(samples * 1000 / max(1, rate)),
            "end_of_turn": False,
            "uses_model_runner_scheduler": self.uses_model_runner_scheduler,
            "runner_kv_backed": self.runner_kv_backed,
            "runtime_impl": self.runtime_impl,
            "owned_runtime": self.owned_runtime,
        }
