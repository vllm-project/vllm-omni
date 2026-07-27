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
        """Validate higgs_audio_v3 request parameters."""
        server = self.ctx.server
        err = server._apply_uploaded_speaker(request)
        if err:
            return err
        if not request.input or not request.input.strip():
            return "higgs_audio_v3: input text cannot be empty"
        if request.ref_audio is not None and not request.ref_text:
            # Voice clone ref_text is optional for v3 (improves fidelity but not required)
            pass
        if request.max_new_tokens is not None:
            if request.max_new_tokens < self.max_new_tokens_min:
                return f"max_new_tokens must be at least {self.max_new_tokens_min}"
        return None

    async def build(
        self, request: "OpenAICreateSpeechRequest", sampling_params_list: list, has_inline_ref_audio: bool
    ) -> PreparedRequest:
        prompt = await self.ctx.server._build_higgs_audio_v3_params(request)
        return PreparedRequest(prompt=prompt, tts_params={}, model_type="higgs_audio_v3")
