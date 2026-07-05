# SPDX-License-Identifier: Apache-2.0
"""Higgs-Audio v3 serving adapter."""

from typing import TYPE_CHECKING

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import ARTTSAdapter, PreparedRequest

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest


@register_tts_adapter
class HiggsAudioV3Adapter(ARTTSAdapter):
    stage_keys = frozenset({"higgs_audio_v3"})
    name = "higgs_audio_v3"

    def validate(self, request: "OpenAICreateSpeechRequest") -> str | None:
        err = self.ctx.server._apply_uploaded_speaker(request)
        if err:
            return err
        return self.ctx.server._validate_higgs_audio_v3_request(request)

    async def build(
        self, request: "OpenAICreateSpeechRequest", sampling_params_list: list, has_inline_ref_audio: bool
    ) -> PreparedRequest:
        prompt = await self.ctx.server._build_higgs_audio_v3_params(request)
        return PreparedRequest(prompt=prompt, tts_params={}, model_type="higgs_audio_v3")
