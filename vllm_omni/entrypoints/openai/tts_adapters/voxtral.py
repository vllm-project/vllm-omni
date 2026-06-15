# SPDX-License-Identifier: Apache-2.0
"""Voxtral TTS serving adapter."""

from typing import TYPE_CHECKING

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import ARTTSAdapter, PreparedRequest

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest


@register_tts_adapter
class VoxtralTTSAdapter(ARTTSAdapter):
    stage_keys = frozenset({"audio_generation"})
    name = "voxtral_tts"

    def validate(self, request: "OpenAICreateSpeechRequest") -> str | None:
        return self.ctx.server._validate_voxtral_tts_request(request)

    async def build(
        self, request: "OpenAICreateSpeechRequest", sampling_params_list: list, has_inline_ref_audio: bool
    ) -> PreparedRequest:
        prompt = await self.ctx.server._build_voxtral_prompt_async(request)
        return PreparedRequest(prompt=prompt, tts_params={}, model_type="voxtral_tts")
