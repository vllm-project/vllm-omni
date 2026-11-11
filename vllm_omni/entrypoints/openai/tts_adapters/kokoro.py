# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Kokoro-82M serving adapter for ``/v1/audio/speech``.

Voices are the checkpoint's ``voices/*.pt`` style packs (``af_heart``,
``bm_george``, ...); a voice's first letter selects its G2P language unless the
request names one. ``speed`` is native (it scales predicted phoneme durations).
The engine prompt is a one-token placeholder; text, voice, speed and seed ride
``additional_information`` to the in-model chunk generator.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar

import torch

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import (
    ARTTSAdapter,
    OutputPolicy,
    PreparedRequest,
    conditioning_cache_salt,
)
from vllm_omni.model_executor.models.kokoro.text_frontend import (
    DEFAULT_VOICE,
    LANG_CODES,
    LANGUAGE_ALIASES,
    list_voices,
    missing_g2p_package,
    resolve_lang_code,
    resolve_voices_dir,
)

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

# Declared schema fields Kokoro does not consume. Rejecting them is
# load-bearing: the pydantic model silently drops undeclared keys, so a
# declared-but-ignored field would otherwise succeed and mislead.
_UNSUPPORTED_OPTIONAL_FIELDS: tuple[str, ...] = (
    "ref_audio",
    "ref_text",
    "ref_audio_2",
    "ref_audio_codes",
    "speaker_embedding",
    "x_vector_only_mode",
    "task_type",
    "instructions",
    "ambient_sound",
    "duration_seconds",
    "max_new_tokens",
    "initial_codec_chunk_frames",
    "non_streaming_mode",
)

_AUTO_LANGUAGE = "Auto"
# The server's generic placeholder voice name.
_PLACEHOLDER_VOICE = "default"


def _resolve_voice(voice: str | None) -> str:
    voice = (voice or DEFAULT_VOICE).lower()
    return DEFAULT_VOICE if voice == _PLACEHOLDER_VOICE else voice


@register_tts_adapter
class KokoroAdapter(ARTTSAdapter):
    name = "kokoro"
    stage_keys = frozenset({"kokoro"})
    model_archs = frozenset({"KokoroForConditionalGeneration"})
    # The network scales predicted durations by 1/speed; the server must not
    # time-stretch the result again.
    native_speed_control: ClassVar[bool] = True

    def _load_supported_speakers(self) -> list[str]:
        engine = self.ctx.engine_client
        assert engine is not None, "Kokoro is served through the AR engine client"
        model_config = engine.model_config
        voices = list_voices(resolve_voices_dir(model_config.model, getattr(model_config, "revision", None)))
        # load_capabilities() takes the first entry as the default speaker.
        if DEFAULT_VOICE in voices:
            voices.remove(DEFAULT_VOICE)
            voices.insert(0, DEFAULT_VOICE)
        return voices

    def _load_supported_languages(self) -> frozenset[str]:
        return frozenset({_AUTO_LANGUAGE, *LANGUAGE_ALIASES})

    def _load_codec_frame_rate(self) -> float | None:
        return None

    def validate(self, request: OpenAICreateSpeechRequest) -> str | None:
        if not request.input or not request.input.strip():
            return "Input text cannot be empty"

        voice = _resolve_voice(request.voice)
        supported = self.capabilities.supported_speakers
        if voice not in supported:
            return f"Invalid voice '{request.voice}'. Supported: {', '.join(sorted(supported))}"

        language = request.language if request.language != _AUTO_LANGUAGE else None
        try:
            lang_code = resolve_lang_code(voice, language)
        except ValueError as exc:
            return str(exc)
        package = missing_g2p_package(lang_code)
        if package is not None:
            return f"Kokoro {LANG_CODES[lang_code]} is not installed on this server; it needs: pip install '{package}'"

        if request.extra_params:
            keys = ", ".join(sorted(request.extra_params))
            return f"Kokoro does not accept extra_params (got: {keys})"

        if request.word_timestamps:
            return "Kokoro does not support 'word_timestamps'; there is no forced-aligner stage in this pipeline"

        for field in _UNSUPPORTED_OPTIONAL_FIELDS:
            if getattr(request, field, None) is not None:
                return f"Kokoro does not support '{field}'"

        return None

    def collect_response_metadata(self, audio_output: Mapping[str, Any], collect: dict) -> None:
        # forward() sets meta["error"] to 1 for a request whose synthesis
        # failed; its audio is empty or cut short, so serving answers 500.
        meta = audio_output.get("meta")
        flag = meta.get("error") if isinstance(meta, Mapping) else None
        if isinstance(flag, list):
            flag = torch.cat([f.reshape(-1) for f in flag]) if flag else None
        if isinstance(flag, torch.Tensor) and bool(flag.any()):
            collect["audio_synthesis_error"] = True

    async def build(
        self,
        request: OpenAICreateSpeechRequest,
        sampling_params_list: list,
        has_inline_ref_audio: bool,
    ) -> PreparedRequest:
        additional_information: dict[str, list] = {
            "text": [request.input],
            "voice": [_resolve_voice(request.voice)],
            "speed": [float(request.speed if request.speed is not None else 1.0)],
        }
        if request.language and request.language != _AUTO_LANGUAGE:
            additional_information["language"] = [request.language]
        if request.seed is not None:
            additional_information["seed"] = [int(request.seed)]
        tts_params: dict = {}
        prompt = {
            # Placeholder: the network reads additional_information, not ids.
            "prompt_token_ids": [0],
            "additional_information": additional_information,
            "cache_salt": conditioning_cache_salt(request, tts_params),
        }
        return PreparedRequest(
            prompt=prompt,
            tts_params=tts_params,
            model_type=self.name,
            output_policy=OutputPolicy(accumulate_nonstreaming=True),
        )
