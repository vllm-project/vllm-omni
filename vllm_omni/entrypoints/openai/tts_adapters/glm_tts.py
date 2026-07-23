# SPDX-License-Identifier: Apache-2.0
"""GLM-TTS serving adapter."""

from typing import TYPE_CHECKING

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import ARTTSAdapter, PreparedRequest

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest


@register_tts_adapter
class GlmTTSAdapter(ARTTSAdapter):
    stage_keys = frozenset({"glm_tts"})
    name = "glm_tts"

    def validate(self, request: "OpenAICreateSpeechRequest") -> str | None:
        """Validate GLM-TTS request — requires ref_audio for voice cloning."""
        server = self.ctx.server
        err = server._apply_uploaded_speaker(request)
        if err:
            return err
        if not request.input or not request.input.strip():
            return "Input text cannot be empty"

        if request.ref_audio is None:
            return "GLM-TTS requires 'ref_audio' for zero-shot voice cloning"
        fmt_err = server._validate_ref_audio_format(request.ref_audio)
        if fmt_err:
            return fmt_err
        if not request.ref_text or not request.ref_text.strip():
            return "GLM-TTS voice cloning requires 'ref_text' (transcript of the reference audio)"

        if request.max_new_tokens is not None:
            if request.max_new_tokens < self.max_new_tokens_min:
                return f"max_new_tokens must be >= {self.max_new_tokens_min}"
            if request.max_new_tokens > self.max_new_tokens_max:
                return f"max_new_tokens cannot exceed {self.max_new_tokens_max}"
        return None

    async def build(
        self, request: "OpenAICreateSpeechRequest", sampling_params_list: list, has_inline_ref_audio: bool
    ) -> PreparedRequest:
        prompt = await self.ctx.server._build_glm_tts_prompt(request, has_inline_ref_audio=has_inline_ref_audio)
        # GLM-TTS dynamic-token sampling stays in the orchestrator tail
        # (keyed on _tts_model_type) during this incremental migration.
        return PreparedRequest(prompt=prompt, tts_params={}, model_type="glm_tts")
