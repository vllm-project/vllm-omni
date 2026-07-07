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
        err = self.ctx.server._apply_uploaded_speaker(request)
        if err:
            return err
        return self.ctx.server._validate_glm_tts_request(request)

    async def build(
        self, request: "OpenAICreateSpeechRequest", sampling_params_list: list, has_inline_ref_audio: bool
    ) -> PreparedRequest:
        prompt = await self.ctx.server._build_glm_tts_prompt(request, has_inline_ref_audio=has_inline_ref_audio)
        # GLM-TTS dynamic-token sampling stays in the orchestrator tail
        # (keyed on _tts_model_type) during this incremental migration.
        return PreparedRequest(prompt=prompt, tts_params={}, model_type="glm_tts")
