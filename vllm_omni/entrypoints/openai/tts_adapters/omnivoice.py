# SPDX-License-Identifier: Apache-2.0
"""OmniVoice serving adapter (AR engine_client path).

OmniVoice can also be served through the pure-diffusion engine via
``for_diffusion`` (``_create_diffusion_speech``); unifying that path is a
follow-up. This adapter covers the AR-stage deployment.
"""

from typing import TYPE_CHECKING

import numpy as np

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import ARTTSAdapter, PreparedRequest

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest


@register_tts_adapter
class OmniVoiceAdapter(ARTTSAdapter):
    stage_keys = frozenset({"omnivoice_generator"})
    name = "omnivoice"

    def validate(self, request: "OpenAICreateSpeechRequest") -> str | None:
        if not request.input or not request.input.strip():
            return "Input text cannot be empty"
        return self.ctx.server._apply_uploaded_speaker(request)

    async def build(
        self, request: "OpenAICreateSpeechRequest", sampling_params_list: list, has_inline_ref_audio: bool
    ) -> PreparedRequest:
        server = self.ctx.server
        prompt: dict = {"input": request.input}
        if request.ref_audio:
            wav, sr = await server._resolve_ref_audio(request.ref_audio)
            prompt["ref_audio"] = (np.asarray(wav, dtype=np.float32), sr)
        if request.ref_text:
            prompt["ref_text"] = request.ref_text
        if request.voice:
            voice_lower = request.voice.lower()
            if voice_lower in server.uploaded_speakers and not has_inline_ref_audio:
                prompt["voice_name"] = voice_lower
                prompt["voice_created_at"] = server._voice_created_at(voice_lower)
        if request.language:
            prompt["lang"] = request.language
        if request.instructions:
            prompt["instruct"] = request.instructions
        return PreparedRequest(prompt=prompt, tts_params={}, model_type="omnivoice")
