# SPDX-License-Identifier: Apache-2.0
"""CoVo-Audio serving adapter."""

from typing import TYPE_CHECKING

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import ARTTSAdapter, PreparedRequest

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest


@register_tts_adapter
class CovoAudioAdapter(ARTTSAdapter):
    stage_keys = frozenset({"fused_thinker_talker"})
    name = "covo_audio"

    def validate(self, request: "OpenAICreateSpeechRequest") -> str | None:
        if not request.input or not request.input.strip():
            return "Input text cannot be empty"
        return None

    async def build(
        self, request: "OpenAICreateSpeechRequest", sampling_params_list: list, has_inline_ref_audio: bool
    ) -> PreparedRequest:
        prompt = self.ctx.server._build_covo_audio_prompt(request)
        return PreparedRequest(prompt=prompt, tts_params={}, model_type="covo_audio")
