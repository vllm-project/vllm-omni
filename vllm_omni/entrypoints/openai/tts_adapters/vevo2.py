# SPDX-License-Identifier: Apache-2.0
"""Vevo2 serving adapter.

Vevo2 (Amphion's unified AR + flow-matching TTS) runs as a single AR stage and
emits the full waveform as delta chunks, following the MOSS-TTS-Nano pattern.
Voice cloning requires a reference clip (``ref_audio``); ``ref_text`` (the
transcript of the reference) is recommended for prosody but not required.
The validation/param-building logic lives on the server
(``_validate_vevo2_request`` / ``_build_vevo2_params``); this adapter wires it
into the per-model adapter framework (RFC #4327).
"""

from typing import TYPE_CHECKING

from vllm.inputs import tokens_input

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import ARTTSAdapter, PreparedRequest, conditioning_cache_salt

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest


@register_tts_adapter
class Vevo2Adapter(ARTTSAdapter):
    stage_keys = frozenset({"vevo2"})
    name = "vevo2"

    def validate(self, request: "OpenAICreateSpeechRequest") -> str | None:
        err = self.ctx.server._apply_uploaded_speaker(request)
        if err:
            return err
        return self.ctx.server._validate_vevo2_request(request)

    async def build(
        self, request: "OpenAICreateSpeechRequest", sampling_params_list: list, has_inline_ref_audio: bool
    ) -> PreparedRequest:
        server = self.ctx.server
        tts_params = await server._build_vevo2_params(request)
        if request.voice:
            voice_lower = request.voice.lower()
            if voice_lower in server.uploaded_speakers and not has_inline_ref_audio:
                tts_params["voice_name"] = [voice_lower]
                tts_params["voice_created_at"] = [server._voice_created_at(voice_lower)]
        prompt = tokens_input(prompt_token_ids=[1])
        prompt["additional_information"] = tts_params
        prompt["cache_salt"] = conditioning_cache_salt(request, tts_params)
        return PreparedRequest(prompt=prompt, tts_params=tts_params, model_type=self.name)
