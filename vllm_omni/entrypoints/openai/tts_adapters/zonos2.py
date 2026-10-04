# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""ZONOS2 text/language/rate/quality/reference-audio speech adapter."""

from __future__ import annotations

import math
from collections.abc import Mapping
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from vllm.utils.async_utils import make_async

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import (
    ARTTSAdapter,
    PreparedRequest,
    TTSGenerationError,
    apply_max_new_tokens,
)
from vllm_omni.model_executor.models.zonos2.zonos2_keys import STATE, TOKEN_BUDGET

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

_LANGUAGES = {
    "english": "en_us",
    "en": "en_us",
    "en_us": "en_us",
    "en_gb": "en_gb",
    "chinese": "cmn",
    "zh": "cmn",
    "zh_cn": "cmn",
    "cmn": "cmn",
    "french": "fr_fr",
    "fr": "fr_fr",
    "fr_fr": "fr_fr",
    "german": "de",
    "de": "de",
    "spanish": "es",
    "es": "es",
    "italian": "it",
    "it": "it",
    "portuguese": "pt_br",
    "pt": "pt_br",
    "pt_br": "pt_br",
    "japanese": "ja",
    "ja": "ja",
    "korean": "ko",
    "ko": "ko",
}
_FIELDS = frozenset(
    {
        "input",
        "model",
        "voice",
        "response_format",
        "speed",
        "stream",
        "stream_format",
        "max_new_tokens",
        "sample_rate",
        "seed",
        "extra_params",
        "language",
        "ref_audio",
        "speaker_embedding",
        "x_vector_only_mode",
    }
)
_SAMPLING = frozenset(
    {
        "temperature",
        "top_k",
        "top_p",
        "min_p",
        "repetition_window",
        "repetition_penalty",
        "repetition_codebooks",
    }
)
_CONDITIONING = frozenset(
    {
        "text_normalization",
        "speaking_rate_bucket",
        "speaking_rate",
        "quality_buckets",
        "quality_values",
        "clean_speaker_background",
        "accurate_mode",
        "emotion_cfg_scale",
        "cfg_scale",
    }
)


@register_tts_adapter
class Zonos2Adapter(ARTTSAdapter):
    name = STATE
    stage_keys = frozenset({STATE})
    model_archs = frozenset({"Zonos2ForConditionalGeneration", "Zonos2TalkerForConditionalGeneration"})
    supported_output_sample_rates = frozenset({44100})
    max_new_tokens_max = 1024
    native_speed_control = True
    validates_generation = True

    def __init__(self, ctx):
        super().__init__(ctx)
        self._processor = None
        self._speaker = None

    def _get_processor(self):
        from vllm_omni.model_executor.models.zonos2.zonos2_processor import Zonos2Processor

        if self._processor is None:
            engine_client = self.ctx.engine_client
            if engine_client is None:
                raise RuntimeError("ZONOS2 processor requires an initialized engine client")
            self._processor = Zonos2Processor(engine_client.model_config.hf_config)
        return self._processor

    def _load_supported_speakers(self) -> list[str]:
        return ["default"]

    def _load_supported_languages(self) -> frozenset[str]:
        return frozenset(_LANGUAGES)

    @staticmethod
    def _language(request):
        language = (request.language or "English").strip().lower().replace("-", "_")
        if language not in _LANGUAGES:
            raise ValueError(f"Unsupported ZONOS2 language: {request.language!r}; use an explicit supported language")
        return _LANGUAGES[language]

    def _conditioning(self, request):
        from vllm_omni.model_executor.models.zonos2.zonos2_conditioning import (
            _resolve_quality_buckets,
            _resolve_speaking_rate_bucket,
        )

        extra = request.extra_params or {}
        config = SimpleNamespace(model_config=self._get_processor().config)
        model_config = config.model_config
        for name in ("speaking_rate",):
            if extra.get(name) is not None and (not math.isfinite(float(extra[name])) or float(extra[name]) <= 0):
                raise ValueError(f"ZONOS2 {name} must be positive and finite")
        if extra.get("speaking_rate_bucket") is not None and type(extra["speaking_rate_bucket"]) is not int:
            raise ValueError("ZONOS2 speaking_rate_bucket must be an integer")
        features = list(model_config.quality_features or model_config.quality_buckets)
        for name in ("quality_buckets", "quality_values"):
            value = extra.get(name)
            if isinstance(value, dict):
                unknown = set(value) - set(features)
                if unknown:
                    raise ValueError(f"Unknown ZONOS2 quality features: {sorted(unknown)}")
                values = list(value.values())
            elif isinstance(value, (list, tuple)):
                if len(value) > len(features):
                    raise ValueError("Too many ZONOS2 quality feature values")
                values = list(value)
            else:
                values = []
            for item in values:
                if item is None:
                    continue
                if name == "quality_buckets" and type(item) is not int:
                    raise ValueError("ZONOS2 quality buckets must be integer indices")
                if name == "quality_values" and not math.isfinite(float(item)):
                    raise ValueError("ZONOS2 quality values must be finite")

        # An omitted OpenAI speed defaults to 1.0; keep the frozen no-rate
        # prompt unless the client supplied the field explicitly.
        speed = request.speed if "speed" in request.model_fields_set else None
        rate = _resolve_speaking_rate_bucket(
            config,
            speaking_rate_bucket=extra.get("speaking_rate_bucket"),
            speaking_rate=extra.get("speaking_rate"),
            speed=speed,
            speaking_rate_enabled=True,
        )
        quality = _resolve_quality_buckets(
            config,
            quality_buckets=extra.get("quality_buckets"),
            quality_values=extra.get("quality_values"),
            quality_enabled=True,
        )
        return {
            "language": self._language(request),
            "speaking_rate_bucket": rate,
            "quality_buckets": quality,
            "text_normalization": extra.get("text_normalization", True),
            "clean_speaker_background": extra.get("clean_speaker_background", False),
            "accurate_mode": extra.get("accurate_mode", True),
        }

    def validate(self, request: OpenAICreateSpeechRequest) -> str | None:
        if not request.input or not request.input.strip():
            return "Input text cannot be empty"
        if request.voice is not None and request.voice.lower() != "default":
            return "ZONOS2 supports voice='default'; use ref_audio for speaker conditioning"
        if request.sample_rate not in (None, 44100):
            return "ZONOS2 uses sample_rate=44100"
        if request.max_new_tokens is not None and request.max_new_tokens > 1024:
            return "max_new_tokens cannot exceed 1024"
        if request.is_streaming() and request.response_format not in ("wav", "pcm"):
            return "ZONOS2 streaming requires response_format='wav' or 'pcm'"
        for field in sorted(request.model_fields_set - _FIELDS):
            if getattr(request, field) is not None:
                return f"ZONOS2 does not support '{field}' (emotion/style CFG is not implemented)"
        if request.ref_audio is not None and request.speaker_embedding is not None:
            return "ref_audio and speaker_embedding are mutually exclusive"
        if request.x_vector_only_mode is False:
            return "ZONOS2 reference conditioning supports speaker embeddings only"
        if isinstance(request.ref_audio, list):
            return "ZONOS2 supports a single reference audio"
        if request.ref_audio is not None:
            error = self.ctx.server._validate_ref_audio_format(request.ref_audio)
            if error:
                return error
        extra = request.extra_params or {}
        unknown = set(extra) - _SAMPLING - _CONDITIONING
        if unknown:
            return f"Unsupported ZONOS2 parameters: {sorted(unknown)}"
        try:
            from vllm_omni.model_executor.models.zonos2.zonos2_sampler import Zonos2SamplingParams

            for key in ("emotion_cfg_scale", "cfg_scale"):
                if key in extra and float(extra[key]) != 1.0:
                    return "ZONOS2 emotion CFG is not implemented; only cfg_scale=emotion_cfg_scale=1 is supported"
            for key in ("text_normalization", "clean_speaker_background", "accurate_mode"):
                if key in extra and not isinstance(extra[key], bool):
                    return f"ZONOS2 {key} must be boolean"
            Zonos2SamplingParams.from_runtime({"extra_args": {k: v for k, v in extra.items() if k in _SAMPLING}})
            self._language(request)
            if (
                any(k in extra for k in ("speaking_rate", "speaking_rate_bucket", "quality_buckets", "quality_values"))
                or "speed" in request.model_fields_set
            ):
                self._conditioning(request)
            if request.speaker_embedding is not None:
                import torch

                speaker = torch.tensor(request.speaker_embedding, dtype=torch.float32)
                if speaker.shape not in ((2048,), (1, 2048)) or not torch.isfinite(speaker).all():
                    return "ZONOS2 speaker_embedding requires a finite 2048D vector"
        except (ValueError, TypeError) as exc:
            return str(exc)
        return None

    def apply_sampling_overrides(
        self,
        sampling_params_list: list,
        request: OpenAICreateSpeechRequest,
        prompt: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> list:
        params = apply_max_new_tokens(sampling_params_list, request)
        if params is sampling_params_list:
            params = [param.clone() for param in sampling_params_list]
        if params:
            params[0].extra_args = {k: v for k, v in (params[0].extra_args or {}).items() if k not in _CONDITIONING}
            params[0].extra_args.update({k: v for k, v in (request.extra_params or {}).items() if k in _SAMPLING})
            if request.seed is not None:
                params[0].seed = request.seed
        return params

    def validate_generation(
        self,
        tts_params: Mapping[str, object],
        *,
        stage0_finish_reason: str | None,
        output_tokens: int,
    ) -> None:
        budget = tts_params.get(TOKEN_BUDGET)
        if not isinstance(budget, int) or budget <= 0:
            return
        if stage0_finish_reason == "length" or output_tokens >= budget:
            raise TTSGenerationError(
                f"ZONOS2 reached its codec-frame budget ({output_tokens}/{budget}); "
                "the generated speech may be incomplete. Increase the budget or use a shorter input.",
                retryable=True,
            )

    async def build(
        self, request: OpenAICreateSpeechRequest, sampling_params_list: list, has_inline_ref_audio: bool
    ) -> PreparedRequest:
        error = self.validate(request)
        if error is not None:
            raise ValueError(error)
        executor = getattr(self.ctx.server, "_tts_executor", None)
        kwargs = self._conditioning(request)
        budget = request.max_new_tokens
        if budget is None:
            budget = sampling_params_list[0].max_tokens if sampling_params_list else 1024
        tts_params = {TOKEN_BUDGET: budget}
        if request.speaker_embedding is not None:
            import torch

            kwargs["speaker_embedding"] = torch.tensor(request.speaker_embedding, dtype=torch.float32)
        if request.ref_audio is not None:
            from vllm_omni.model_executor.models.zonos2.zonos2_speaker import Zonos2SpeakerEncoder

            try:
                wav, sr, key = await self.ctx.server._resolve_ref_audio(request.ref_audio)
            except (OSError, ValueError, RuntimeError) as exc:
                raise ValueError(
                    "Cannot decode ZONOS2 reference audio; check the audio input and installed audio backend/ffmpeg"
                ) from exc
            if self._speaker is None:
                self._speaker = Zonos2SpeakerEncoder()
            kwargs["speaker_embedding"] = await make_async(self._speaker.encode, executor=executor)(wav, sr)
            tts_params["ref_audio_cache_key"] = key
        prompt = await make_async(self._get_processor().build_prompt, executor=executor)(request.input, **kwargs)
        return PreparedRequest(prompt=prompt, tts_params=tts_params, model_type=self.name)
