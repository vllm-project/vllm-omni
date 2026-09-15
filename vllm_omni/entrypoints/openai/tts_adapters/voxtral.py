# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Voxtral TTS serving adapter."""

from typing import TYPE_CHECKING, Any

from vllm.utils.async_utils import make_async

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import ARTTSAdapter, PreparedRequest, apply_max_new_tokens
from vllm_omni.entrypoints.openai.tts_adapters.capabilities import load_supported_speakers

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest


@register_tts_adapter
class VoxtralTTSAdapter(ARTTSAdapter):
    stage_keys = frozenset({"audio_generation"})
    name = "voxtral_tts"

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self._build_prompt_async = make_async(self._build_prompt, executor=getattr(ctx.server, "_tts_executor", None))

    def _build_prompt(self, request: "OpenAICreateSpeechRequest") -> dict[str, Any]:
        """Build a Voxtral prompt for either a preset voice or inline reference audio."""
        from mistral_common.protocol.speech.request import SpeechRequest
        from vllm.inputs import tokens_input

        ref_audio = request.ref_audio
        if not request.voice and not ref_audio:
            raise ValueError("Voxtral requires either a voice name or ref_audio.")
        if isinstance(ref_audio, str) and ref_audio.startswith("data:"):
            _, _, ref_audio = ref_audio.partition(",")
        server = self.ctx.server
        if server._tts_tokenizer is None:
            from vllm.tokenizers import cached_tokenizer_from_config

            server._tts_tokenizer = cached_tokenizer_from_config(self.ctx.engine_client.model_config).instruct
        if request.voice is not None:
            tokenized = server._tts_tokenizer.encode_speech_request(
                SpeechRequest(input=request.input, voice=request.voice)
            )
            prompt = tokens_input(prompt_token_ids=tokenized.tokens)
            prompt["additional_information"] = {"voice": [request.voice]}
            return prompt
        tokenized = server._tts_tokenizer.encode_speech_request(SpeechRequest(input=request.input, ref_audio=ref_audio))
        audio = tokenized.audios[0]
        return {
            "prompt_token_ids": tokenized.tokens,
            "multi_modal_data": {"audio": [(audio.audio_array, audio.sampling_rate)]},
        }

    def validate(self, request: "OpenAICreateSpeechRequest") -> str | None:
        """Validate Voxtral TTS request parameters. Returns error message or None."""
        server = self.ctx.server
        if not request.input or not request.input.strip():
            return "Input text cannot be empty"

        # Voxtral TTS requires either a preset voice or ref_audio for voice cloning.
        if request.voice is None and request.ref_audio is None:
            return "Either 'voice' (preset speaker) or 'ref_audio' (voice cloning) must be provided"

        if request.ref_audio is not None:
            fmt_err = server._validate_ref_audio_format(request.ref_audio)
            if fmt_err:
                return fmt_err

        if request.voice is not None:
            request.voice = request.voice.lower()
            available_speakers = server._get_available_speakers()
            if available_speakers and request.voice not in available_speakers:
                return f"Invalid speaker '{request.voice}'. Supported: {', '.join(sorted(available_speakers))}"

        if request.max_new_tokens is not None:
            if request.max_new_tokens < self.max_new_tokens_min:
                return f"max_new_tokens must be at least {self.max_new_tokens_min}"
            if request.max_new_tokens > self.max_new_tokens_max:
                return f"max_new_tokens cannot exceed {self.max_new_tokens_max}"

        return None

    async def build(
        self, request: "OpenAICreateSpeechRequest", sampling_params_list: list, has_inline_ref_audio: bool
    ) -> PreparedRequest:
        prompt = await self._build_prompt_async(request)
        return PreparedRequest(prompt=prompt, tts_params={}, model_type="voxtral_tts")

    def _load_supported_speakers(self) -> set[str]:
        config = self.ctx.engine_client.model_config.hf_config.audio_config
        return load_supported_speakers(self.ctx.engine_client, config)

    def apply_sampling_overrides(
        self,
        sampling_params_list: list,
        request: "OpenAICreateSpeechRequest",
        prompt: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> list:
        return apply_max_new_tokens(sampling_params_list, request)
