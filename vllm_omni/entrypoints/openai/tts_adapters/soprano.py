# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Soprano's fixed-voice English Speech API, including native audio deltas."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from vllm import SamplingParams
from vllm.exceptions import VLLMValidationError
from vllm.utils.async_utils import make_async

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import (
    ARTTSAdapter,
    PreparedRequest,
    TTSGenerationError,
    apply_max_new_tokens,
    resolve_stage_model_path,
)

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

_UNSUPPORTED_FIELDS = (
    "ref_audio",
    "ref_text",
    "ref_audio_2",
    "speaker_embedding",
    "x_vector_only_mode",
    "task_type",
    "non_streaming_mode",
    "instructions",
    "ambient_sound",
    "duration_seconds",
    "initial_codec_chunk_frames",
)
_SAMPLING_FIELDS = frozenset({"temperature", "top_p", "top_k", "repetition_penalty"})


@register_tts_adapter
class SopranoAdapter(ARTTSAdapter):
    name = "soprano"
    stage_keys = frozenset({"soprano_ar"})
    # The checkpoint's original Qwen3 architecture is shared by text models.
    # Only the explicit Soprano pipeline stage identifies this adapter.
    max_new_tokens_max = 512
    supported_output_sample_rates = frozenset({32000})

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self._tokenizer: Any = None
        self._build_prompt_async = make_async(self._build_prompt, executor=getattr(ctx.server, "_tts_executor", None))

    def _load_supported_speakers(self) -> list[str]:
        return ["default"]

    def _load_supported_languages(self) -> frozenset[str]:
        return frozenset({"English"})

    def _load_codec_frame_rate(self) -> None:
        # Soprano consumes continuous predictors, not discrete codec frames.
        return None

    def validate(self, request: OpenAICreateSpeechRequest) -> str | None:
        if not request.input.strip():
            return "Input text cannot be empty"
        if request.voice is not None and request.voice.lower() != "default":
            return "Soprano supports only voice='default' (or omitted): the checkpoint's fixed voice"
        if request.language is not None and request.language.lower() not in {"en", "english"}:
            return "Soprano supports English only; omit language or use 'en' / 'English'"
        for field in _UNSUPPORTED_FIELDS:
            if getattr(request, field) is not None:
                return f"Soprano does not support '{field}'"
        if request.is_streaming() and not getattr(
            getattr(self.ctx.engine_client, "model_config", None), "async_chunk", False
        ):
            return "This Soprano deployment supports complete responses; use soprano_streaming.yaml for streaming"
        if request.is_streaming() and request.response_format not in {"pcm", "wav"}:
            return "Soprano streaming supports pcm or wav only"
        if not request.is_streaming() and request.response_format != "wav":
            return "Soprano Speech API initially supports response_format='wav' only"
        if request.speed is not None and request.speed != 1.0:
            return "Soprano does not support speed adjustment; use speed=1.0 or omit it"
        if request.word_timestamps:
            return "Soprano does not support word_timestamps"
        if request.seed is not None and request.seed < 0:
            return "Soprano seed must be nonnegative"
        if error := self._validate_max_new_tokens(request):
            return error
        extra = request.extra_params or {}
        if unknown := extra.keys() - _SAMPLING_FIELDS:
            return f"Unsupported Soprano extra_params: {', '.join(sorted(unknown))}"
        for key, value in extra.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                return f"Soprano extra_params.{key} must be a finite number"
            if key == "top_k" and not isinstance(value, int):
                return "Soprano extra_params.top_k must be an integer"
        try:
            SamplingParams(**extra)
        except (TypeError, ValueError, VLLMValidationError) as exc:
            return f"Invalid Soprano sampling parameters: {exc}"
        return None

    def _build_prompt(self, text: str) -> dict:
        from transformers import AutoTokenizer

        from vllm_omni.model_executor.models.soprano.offline import prepare_inputs

        if self._tokenizer is None:
            model = resolve_stage_model_path(self.ctx.engine_client)
            if model is None:
                raise ValueError("Soprano tokenizer needs the deployed checkpoint path")
            self._tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=False)
        inputs = prepare_inputs(text, self._tokenizer)
        if len(inputs) != 1:
            raise ValueError(
                "Soprano Speech API currently accepts one synthesis unit; "
                "use offline.synthesize for multi-sentence text"
            )
        return inputs[0]

    async def build(self, request, sampling_params_list, has_inline_ref_audio) -> PreparedRequest:
        return PreparedRequest(
            prompt=await self._build_prompt_async(request.input),
            tts_params={},
            model_type=self.name,
        )

    def apply_sampling_overrides(self, sampling_params_list, request, prompt=None, request_id=None) -> list:
        params = apply_max_new_tokens(sampling_params_list, request)
        if (penalty := (request.extra_params or {}).get("repetition_penalty")) is not None:
            params = copy.deepcopy(params)
            params[0].repetition_penalty = penalty
        return params

    def collect_response_metadata(self, audio_output: Mapping[str, Any], collect: dict) -> None:
        for key in ("audio_truncated", "audio_synthesis_error"):
            value = audio_output.get(key)
            if value is not None:
                values = value if isinstance(value, list) else [value]
                collect[key] = any(bool(item) for item in values)

    def validate_stream_output(self, audio_output: Mapping[str, Any]) -> None:
        status: dict[str, bool] = {}
        self.collect_response_metadata(audio_output, status)
        if status.get("audio_synthesis_error"):
            raise TTSGenerationError("Soprano streaming synthesis failed; no successful audio tail was produced")

    def validate_stream_audio(self, *, has_audio: bool) -> None:
        if not has_audio:
            raise TTSGenerationError("Soprano produced no playable audio")
