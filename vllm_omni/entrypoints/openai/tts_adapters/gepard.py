# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Gepard-1.0 serving adapter for ``/v1/audio/speech``.

Zero-shot only: the trained ``null_prefix`` is the default voice. Prompt IDs
are assembled with the same layout helper the offline path uses. Frame budget
rides ``SamplingParams.max_tokens`` (one token = one 1024-sample frame).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vllm.utils.async_utils import make_async

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import (
    ARTTSAdapter,
    OutputPolicy,
    PreparedRequest,
    apply_max_new_tokens,
    conditioning_cache_salt,
)

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
    from vllm_omni.model_executor.models.gepard.configuration_gepard import GepardConfig

_DEFAULT_VOICE = "default"

# Declared schema fields Gepard does not consume. Rejecting them here is
# load-bearing: the pydantic model silently drops undeclared keys, so a
# declared-but-ignored field would otherwise succeed and mislead.
_UNSUPPORTED_OPTIONAL_FIELDS: tuple[tuple[str, str], ...] = (
    ("ref_audio", "ref_audio"),
    ("ref_text", "ref_text"),
    ("ref_audio_2", "ref_audio_2"),
    ("speaker_embedding", "speaker_embedding"),
    ("x_vector_only_mode", "x_vector_only_mode"),
    ("task_type", "task_type"),
    ("non_streaming_mode", "non_streaming_mode"),
    ("instructions", "instructions"),
    ("language", "language"),
    ("ambient_sound", "ambient_sound"),
    ("duration_seconds", "duration_seconds"),
    ("initial_codec_chunk_frames", "initial_codec_chunk_frames"),
)


@register_tts_adapter
class GepardAdapter(ARTTSAdapter):
    name = "gepard"
    stage_keys = frozenset({"gepard"})
    model_archs = frozenset({"GepardTalkerForConditionalGeneration"})

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self._tokenizer: Any = None
        self._gepard_config: GepardConfig | None = None
        self._build_prompt_async = make_async(self._build_prompt, executor=getattr(ctx.server, "_tts_executor", None))

    def _load_supported_speakers(self) -> set[str]:
        return {_DEFAULT_VOICE}

    def validate(self, request: OpenAICreateSpeechRequest) -> str | None:
        if not request.input or not request.input.strip():
            return "Input text cannot be empty"

        if request.voice is not None:
            voice = request.voice.lower()
            if voice != _DEFAULT_VOICE:
                return (
                    f"Invalid voice '{request.voice}'. Supported: {_DEFAULT_VOICE}. "
                    "Voice cloning from reference audio is not available yet."
                )

        if request.speed is not None and request.speed != 1.0:
            return "Gepard does not support 'speed' adjustments; audio is rendered at the native 22.05 kHz rate"

        if request.extra_params:
            keys = ", ".join(sorted(request.extra_params))
            return (
                f"Gepard does not accept extra_params (got: {keys}). "
                "Sampling is fixed in-model; temperature/top_p/top_k cannot be overridden."
            )

        if request.word_timestamps:
            return "Gepard does not support 'word_timestamps'; there is no forced-aligner stage in this pipeline"

        fmt = (request.response_format or "").lower()
        if fmt == "opus":
            return (
                "Gepard does not support 'response_format'='opus'; "
                "native 22.05 kHz is not an Opus sample rate (8000, 12000, 16000, 24000, 48000)"
            )

        if request.max_new_tokens is not None:
            if request.max_new_tokens < self.max_new_tokens_min or request.max_new_tokens > self.max_new_tokens_max:
                return f"max_new_tokens must be between {self.max_new_tokens_min} and {self.max_new_tokens_max} frames"

        for attr, field in _UNSUPPORTED_OPTIONAL_FIELDS:
            if getattr(request, attr, None) is not None:
                return f"Gepard does not support '{field}'"

        return None

    def apply_sampling_overrides(
        self,
        sampling_params_list: list,
        request: OpenAICreateSpeechRequest,
        prompt: dict | None = None,
        request_id: str | None = None,
    ) -> list:
        return apply_max_new_tokens(sampling_params_list, request)

    async def build(
        self,
        request: OpenAICreateSpeechRequest,
        sampling_params_list: list,
        has_inline_ref_audio: bool,
    ) -> PreparedRequest:
        tts_params: dict = {}
        prompt = await self._build_prompt_async(request.input)
        prompt["cache_salt"] = conditioning_cache_salt(request, tts_params)
        return PreparedRequest(
            prompt=prompt,
            tts_params=tts_params,
            model_type=self.name,
            output_policy=OutputPolicy(accumulate_nonstreaming=True),
        )

    def _build_prompt(self, text: str) -> dict[str, Any]:
        from vllm_omni.model_executor.models.gepard.prompt import build_gepard_prompt_ids

        prompt_token_ids = build_gepard_prompt_ids(self._tokenize(text), config=self._config())
        return {
            "prompt_token_ids": prompt_token_ids,
            "additional_information": {"text": [text]},
        }

    def _model_config(self) -> Any:
        engine = self.ctx.engine_client
        assert engine is not None
        return engine.model_config

    def _config(self) -> GepardConfig:
        if self._gepard_config is None:
            from vllm_omni.model_executor.models.gepard.configuration_gepard import GepardConfig

            model_config = self._model_config()
            self._gepard_config = GepardConfig.from_checkpoint(
                model_config.model, revision=getattr(model_config, "revision", None)
            )
        return self._gepard_config

    def _tokenize(self, text: str) -> list[int]:
        tokenizer = self._tokenizer
        if tokenizer is None:
            from transformers import AutoTokenizer

            model_config = self._model_config()
            tokenizer = AutoTokenizer.from_pretrained(
                model_config.model,
                revision=getattr(model_config, "revision", None),
                trust_remote_code=True,
            )
            self._tokenizer = tokenizer
        return tokenizer(text, add_special_tokens=False)["input_ids"]
