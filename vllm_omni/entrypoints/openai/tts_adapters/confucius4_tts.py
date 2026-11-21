# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Confucius4-TTS serving adapter."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from vllm.inputs import tokens_input

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import (
    ARTTSAdapter,
    PreparedRequest,
    apply_max_new_tokens,
)
from vllm_omni.model_executor.models.confucius4_tts.configuration_confucius4_tts import (
    DEFAULT_MAX_TEXT_TOKENS,
    MAX_GENERATED_SEMANTIC_TOKENS,
    OUTPUT_SAMPLE_RATE,
)
from vllm_omni.model_executor.models.confucius4_tts.prompt_utils import (
    LANGUAGE_TOKEN_MAP,
    tokenize_confucius4_prompt,
)

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import (
        OpenAICreateSpeechRequest,
    )


def confucius4_conditioning_cache_salt(
    *,
    text: str,
    lang: str,
    ref_audio_cache_key: str,
    voice_name: str | None = None,
    voice_created_at: int | None = None,
) -> str:
    """Hash the conditioning hidden behind the placeholder token prompt."""
    digest = hashlib.sha256()
    for value in (
        text,
        lang,
        ref_audio_cache_key,
        voice_name,
        voice_created_at,
    ):
        digest.update(b"\x00")
        if value is not None:
            digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()[:32]


@register_tts_adapter
class Confucius4TTSAdapter(ARTTSAdapter):
    name = "confucius4_tts"
    stage_keys = frozenset({"confucius4_tts_talker"})
    model_archs = frozenset({"Confucius4TTSTalkerForConditionalGeneration"})
    max_new_tokens_max = MAX_GENERATED_SEMANTIC_TOKENS
    supported_output_sample_rates = frozenset({OUTPUT_SAMPLE_RATE})

    @classmethod
    def matches(cls, model_stage: str | None, model_arch: str | None) -> bool:
        # The pipeline owns a dedicated stage key. Only fall back to the
        # checkpoint architecture when the caller has no stage metadata.
        if model_stage is not None:
            return model_stage in cls.stage_keys
        return model_arch in cls.model_archs

    @staticmethod
    def _language(request: OpenAICreateSpeechRequest) -> str:
        extras = request.extra_params
        if isinstance(extras, dict) and "lang" in extras:
            return str(extras["lang"])
        return request.language or "zh"

    @staticmethod
    def _single_ref_audio(
        request: OpenAICreateSpeechRequest,
    ) -> str | None:
        source = request.ref_audio
        if isinstance(source, list):
            return source[0] if len(source) == 1 else None
        return source

    def validate(
        self,
        request: OpenAICreateSpeechRequest,
    ) -> str | None:
        server = self.ctx.server
        if not request.input or not request.input.strip():
            return "Input text cannot be empty"
        if request.stream or request.stream_format is not None:
            return "Confucius4-TTS does not support streaming. Use stream=false"
        if request.extra_params is not None and not isinstance(
            request.extra_params,
            dict,
        ):
            return "extra_params must be a JSON object/dict."

        lang = self._language(request)
        if lang not in LANGUAGE_TOKEN_MAP:
            return f"language/extra_params.lang must be one of {sorted(LANGUAGE_TOKEN_MAP)}"

        if isinstance(request.ref_audio, list) and len(request.ref_audio) != 1:
            return "Confucius4-TTS accepts exactly one reference audio"
        voice_lower = request.voice.lower() if isinstance(request.voice, str) else None
        uploaded_audio = (
            server._get_uploaded_audio_data(voice_lower)
            if voice_lower and voice_lower in server.uploaded_speakers
            else None
        )
        source = self._single_ref_audio(request)
        if source is None and uploaded_audio is None:
            return "Confucius4-TTS requires 'ref_audio' or an uploaded audio voice for voice cloning"
        if source is not None:
            format_error = server._validate_ref_audio_format(source)
            if format_error:
                return format_error
        return self._validate_max_new_tokens(request)

    async def build(
        self,
        request: OpenAICreateSpeechRequest,
        sampling_params_list: list,
        has_inline_ref_audio: bool,
    ) -> PreparedRequest:
        del sampling_params_list, has_inline_ref_audio
        server = self.ctx.server
        model_path = str(server.engine_client.model_config.model)
        lang = self._language(request)
        token_ids = tokenize_confucius4_prompt(
            model_path,
            request.input,
            lang,
        )
        if len(token_ids) > DEFAULT_MAX_TEXT_TOKENS:
            raise ValueError(f"Confucius4-TTS formatted prompt exceeds {DEFAULT_MAX_TEXT_TOKENS} tokens")

        voice_lower = request.voice.lower() if isinstance(request.voice, str) else None
        source = self._single_ref_audio(request)
        using_uploaded_voice = False
        if source is None and voice_lower and voice_lower in server.uploaded_speakers:
            source = server._get_uploaded_audio_data(voice_lower)
            using_uploaded_voice = source is not None
        if source is None:
            raise ValueError("Confucius4-TTS reference audio disappeared after validation")

        wav_list, sample_rate, cache_key = await server._resolve_ref_audio(source)
        params: dict[str, Any] = {
            "text": [request.input],
            "lang": [lang],
            "text_token_ids": [token_ids],
            "voice": [[wav_list, sample_rate]],
            "ref_audio_cache_key": [cache_key],
        }
        voice_created_at = None
        if using_uploaded_voice and voice_lower:
            voice_created_at = server._voice_created_at(voice_lower)
            params["voice_name"] = [voice_lower]
            params["voice_created_at"] = [voice_created_at]

        prompt = tokens_input(prompt_token_ids=[0] * (len(token_ids) + 2))
        prompt["additional_information"] = params
        prompt["cache_salt"] = confucius4_conditioning_cache_salt(
            text=request.input,
            lang=lang,
            ref_audio_cache_key=str(cache_key),
            voice_name=voice_lower if using_uploaded_voice else None,
            voice_created_at=voice_created_at,
        )
        return PreparedRequest(
            prompt=prompt,
            tts_params=params,
            model_type=self.name,
        )

    def apply_sampling_overrides(
        self,
        sampling_params_list: list,
        request: OpenAICreateSpeechRequest,
        prompt: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> list:
        del prompt, request_id
        if request.max_new_tokens is None:
            return sampling_params_list
        return apply_max_new_tokens(sampling_params_list, request)

    def collect_response_metadata(self, audio_output: Mapping[str, Any], collect: dict) -> None:
        """Reject a request whose Stage 0 produced no semantic code.

        An immediate EOS is a legal sampling outcome; Stage 1 answers it with an
        empty waveform so the failure stays local to that request. Serving then
        returns a 500 instead of a successful zero-length audio file.
        """
        for key in ("audio", "model_outputs"):
            if key not in audio_output:
                continue
            value = audio_output[key]
            chunks = value if isinstance(value, (list, tuple)) else [value]
            sizes = [int(chunk.numel()) for chunk in chunks if hasattr(chunk, "numel")]
            if sizes and sum(sizes) == 0:
                collect["audio_synthesis_error"] = True
            return

    def _load_supported_speakers(self) -> list[str]:
        return []

    def _load_supported_languages(self) -> frozenset[str]:
        return frozenset(LANGUAGE_TOKEN_MAP)
