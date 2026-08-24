# SPDX-License-Identifier: Apache-2.0
"""OpenAI speech adapter for the native Irodori v4-Small diffusion pipeline."""

from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING, Any

import numpy as np

from vllm_omni.diffusion.models.irodori_tts.pipeline_irodori_tts import IrodoriTTSPipeline
from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import DiffusionTTSAdapter, PreparedRequest

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest


_NUMERIC_OPTIONS: dict[str, tuple[float, float, bool]] = {
    "num_steps": (1, 100, True),
    "seed": (0, 2**63 - 1, True),
    "seconds": (0.5, 30.0, False),
    "duration_scale": (0.25, 4.0, False),
    "cfg_scale_text": (0.0, 10.0, False),
    "cfg_scale_caption": (0.0, 10.0, False),
    "cfg_scale_speaker": (0.0, 10.0, False),
    "cfg_refresh_interval": (1, 100, True),
}

_FORWARDED_EXTRAS = frozenset(
    {
        "seconds",
        "duration_scale",
        "cfg_scale_text",
        "cfg_scale_caption",
        "cfg_scale_speaker",
        "cfg_refresh_interval",
    }
)


@register_tts_adapter
class IrodoriTTSAdapter(DiffusionTTSAdapter):
    """Keep Irodori's intentionally small public request surface explicit."""

    name = "irodori_tts"
    model_archs = frozenset({"IrodoriTTSPipeline"})
    pipeline_cls = IrodoriTTSPipeline

    @staticmethod
    def _number_error(key: str, value: Any) -> str | None:
        minimum, maximum, integer = _NUMERIC_OPTIONS[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return f"extra_params.{key} must be a numeric value."
        if not math.isfinite(float(value)) or not minimum <= float(value) <= maximum:
            return f"extra_params.{key} must be within [{minimum}, {maximum}]."
        if integer and not isinstance(value, int):
            return f"extra_params.{key} must be an integer in [{int(minimum)}, {int(maximum)}]."
        return None

    def validate(self, request: OpenAICreateSpeechRequest) -> str | None:
        if not isinstance(request.input, str) or not request.input.strip():
            return "Input text cannot be empty."
        if request.is_streaming() or request.word_timestamps:
            return "Irodori-TTS supports final-only non-streaming audio; streaming and word timestamps are unavailable."
        if (request.response_format or "wav").lower() not in {"wav", "pcm"}:
            return "Irodori-TTS supports only non-streaming WAV or PCM responses."
        if request.speed is not None and request.speed != 1.0:
            return "Irodori-TTS does not support speed adjustment; use extra_params.duration_scale instead."
        forbidden = {
            "voice": request.voice,
            "task_type": request.task_type,
            "language": request.language,
            "ref_text": request.ref_text,
            "ref_audio_2": request.ref_audio_2,
            "speaker_embedding": request.speaker_embedding,
            "x_vector_only_mode": request.x_vector_only_mode,
            "max_new_tokens": request.max_new_tokens,
            "initial_codec_chunk_frames": request.initial_codec_chunk_frames,
            "non_streaming_mode": request.non_streaming_mode,
        }
        used = sorted(name for name, value in forbidden.items() if value not in (None, False))
        if used:
            return f"Irodori-TTS does not support: {', '.join(used)}."
        refs = request.ref_audio
        if isinstance(refs, list):
            if not refs:
                return "Irodori-TTS ref_audio list cannot be empty."
            if any(not isinstance(item, str) for item in refs):
                return "Every Irodori-TTS ref_audio list item must be a URI string."
            for item in refs:
                if error := self.ctx.server._validate_ref_audio_format(item):
                    return error
        elif refs is not None:
            if error := self.ctx.server._validate_ref_audio_format(refs):
                return error
        extras = request.extra_params or {}
        if not isinstance(extras, dict):
            return "extra_params must be a JSON object/dict."
        unknown = sorted(set(extras) - set(_NUMERIC_OPTIONS))
        if unknown:
            return f"Unsupported Irodori-TTS extra_params: {unknown}."
        for key, value in extras.items():
            if error := self._number_error(key, value):
                return error
        return None

    async def build(
        self,
        request: OpenAICreateSpeechRequest,
        sampling_params_list: list,
        has_inline_ref_audio: bool,
    ) -> PreparedRequest:
        del sampling_params_list, has_inline_ref_audio
        refs = request.ref_audio
        prompt: dict[str, Any] = {"input": request.input, "caption": request.instructions or ""}
        if refs is not None:
            ref_list = [refs] if isinstance(refs, str) else refs
            assert isinstance(ref_list, list)  # validated above
            resolved = await self.ctx.server._resolve_ref_audio_many(ref_list, min_duration=1.0, max_duration=120.0)
            total_duration = sum(len(wav) / sr for wav, sr in resolved)
            if total_duration > 120.0:
                raise ValueError("Combined Irodori-TTS reference audio must be at most 120 seconds.")
            prompt["ref_audio"] = [(np.asarray(wav, dtype=np.float32), sr) for wav, sr in resolved]
        return PreparedRequest(prompt=prompt, model_type=self.name)

    def apply_sampling_overrides(
        self,
        sampling_params_list: list,
        request: OpenAICreateSpeechRequest,
        prompt: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> list:
        if not sampling_params_list:
            raise ValueError("Irodori-TTS requires a diffusion sampling-parameter stage.")
        result = copy.deepcopy(sampling_params_list)
        params = result[0]
        extras = dict(request.extra_params or {})
        if "num_steps" in extras:
            params.num_inference_steps = int(extras.pop("num_steps"))
        if request.seed is not None:
            params.seed = int(request.seed)
        elif "seed" in extras:
            params.seed = int(extras.pop("seed"))
        params.extra_args = {key: value for key, value in extras.items() if key in _FORWARDED_EXTRAS}
        return result
