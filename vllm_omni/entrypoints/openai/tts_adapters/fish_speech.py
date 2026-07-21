# SPDX-License-Identifier: Apache-2.0
"""Fish Speech serving adapter (retires the legacy ``_is_fish_speech`` flag)."""

from typing import TYPE_CHECKING

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import ARTTSAdapter, PreparedRequest, conditioning_cache_salt

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest


@register_tts_adapter
class FishSpeechAdapter(ARTTSAdapter):
    stage_keys = frozenset({"fish_speech_slow_ar"})
    name = "fish_tts"

    def validate(self, request: "OpenAICreateSpeechRequest") -> str | None:
        """Validate Fish Speech request parameters. Returns error message or None."""
        server = self.ctx.server
        err = server._apply_uploaded_speaker(request)
        if err:
            return err
        if not request.input or not request.input.strip():
            return "Input text cannot be empty"

        if request.ref_audio is not None:
            fmt_err = server._validate_ref_audio_format(request.ref_audio)
            if fmt_err:
                return fmt_err
            if not request.ref_text or not request.ref_text.strip():
                return "Voice cloning requires 'ref_text' (transcript of the reference audio)"

        if request.max_new_tokens is not None:
            if request.max_new_tokens < self.max_new_tokens_min:
                return f"max_new_tokens must be at least {self.max_new_tokens_min}"
            if request.max_new_tokens > self.max_new_tokens_max:
                return f"max_new_tokens cannot exceed {self.max_new_tokens_max}"

        return None

    async def build(
        self, request: "OpenAICreateSpeechRequest", sampling_params_list: list, has_inline_ref_audio: bool
    ) -> PreparedRequest:
        server = self.ctx.server
        ref_audio_data = None
        if request.ref_audio is not None:
            wav_list, sr = await server._resolve_ref_audio(request.ref_audio)
            ref_audio_data = (wav_list, sr)
        prompt = await server._build_fish_speech_prompt_async(
            request, ref_audio_data=ref_audio_data, has_inline_ref_audio=has_inline_ref_audio
        )
        tts_params = {}
        prompt["cache_salt"] = conditioning_cache_salt(request, tts_params)
        return PreparedRequest(prompt=prompt, tts_params=tts_params, model_type="fish_speech")
