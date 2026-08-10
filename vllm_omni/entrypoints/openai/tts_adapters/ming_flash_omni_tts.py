# SPDX-License-Identifier: Apache-2.0
"""MingFlashOmniTTS serving adapter."""

from typing import TYPE_CHECKING

from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import ARTTSAdapter, PreparedRequest

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

logger = init_logger(__name__)


@register_tts_adapter
class MingFlashOmniTTSAdapter(ARTTSAdapter):
    stage_keys = frozenset({"ming_tts"})
    name = "ming_flash_omni_tts"

    def validate(self, request: "OpenAICreateSpeechRequest") -> str | None:
        """Validate Ming-flash-omni standalone-talker request parameters."""
        server = self.ctx.server
        if not request.input or not request.input.strip():
            return "Input text cannot be empty"
        if request.instructions is not None:
            if not isinstance(request.instructions, str):
                return "instructions must be a string"
            if len(request.instructions) > server._max_instructions_length:
                return f"instructions exceeds max length {server._max_instructions_length}"

        if request.task_type is not None:
            return "'task_type' is not supported for Ming-flash-omni TTS"
        if request.language is not None:
            return "'language' is not supported for Ming-flash-omni TTS (language is inferred from input text)"
        if request.x_vector_only_mode is not None:
            return "'x_vector_only_mode' is not supported for Ming-flash-omni TTS"
        if request.initial_codec_chunk_frames is not None:
            return "'initial_codec_chunk_frames' is not supported for Ming-flash-omni TTS"

        # Per-request voice cloning from raw audio is not yet wired up: Ming
        # extracts spk_emb / prompt_wav_lat / prompt_wav_emb model-side via
        # register_prompt_wav() at engine init. For ad-hoc cloning, callers
        # should pre-compute speaker_embedding and pass it directly.
        if request.ref_audio is not None:
            return (
                "'ref_audio' is not yet supported for Ming-flash-omni TTS; "
                "use a preset 'voice' or 'speaker_embedding' instead"
            )
        if request.ref_text is not None:
            return "'ref_text' is not yet supported for Ming-flash-omni TTS"

        if request.max_new_tokens is not None and request.max_new_tokens <= 0:
            return "'max_new_tokens' must be a positive integer"
        return None

    async def build(
        self, request: "OpenAICreateSpeechRequest", sampling_params_list: list, has_inline_ref_audio: bool
    ) -> PreparedRequest:
        server = self.ctx.server
        prompt = server._build_ming_flash_omni_prompt(request)
        return PreparedRequest(prompt=prompt, tts_params={}, model_type="ming_flash_omni_tts")
