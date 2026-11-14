# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""OmniVoice serving adapter (AR engine_client path).

OmniVoice can also be served through the pure-diffusion engine via
``for_diffusion`` (``_create_diffusion_speech``); unifying that path is a
follow-up. This adapter covers the AR-stage deployment.
"""

from typing import TYPE_CHECKING

import numpy as np

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import ARTTSAdapter, PreparedRequest
from vllm_omni.model_executor.models.omnivoice.prompt_utils import prepare_instruct, validate_instruction
from vllm_omni.request_capabilities import RequestConstraints

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest


@register_tts_adapter
class OmniVoiceAdapter(ARTTSAdapter):
    stage_keys = frozenset({"omnivoice_generator"})
    name = "omnivoice"

    def get_request_constraints(self) -> RequestConstraints:
        from vllm_omni.request_capabilities import InputConstraint, ParameterConstraint

        request = super().get_request_constraints()
        return request.model_copy(
            update={
                "inputs": {
                    **request.inputs,
                    **{
                        role: InputConstraint(support="supported", min_count=0, max_count=1)
                        for role in ("reference_audio", "reference_text", "instructions", "voice")
                    },
                },
                "parameters": {
                    **request.parameters,
                    "language": ParameterConstraint(support="supported", type="string"),
                },
            }
        )

    def validate(self, request: "OpenAICreateSpeechRequest") -> str | None:
        if not request.input or not request.input.strip():
            return "Input text cannot be empty"

        err = self.ctx.server._apply_uploaded_speaker(request)
        if err is not None:
            return err

        return validate_instruction(request.instructions)

    async def build(
        self, request: "OpenAICreateSpeechRequest", sampling_params_list: list, has_inline_ref_audio: bool
    ) -> PreparedRequest:
        server = self.ctx.server
        prompt: dict = {"input": request.input}
        if request.ref_audio:
            wav, sr, _ = await server._resolve_ref_audio(request.ref_audio)
            prompt["ref_audio"] = (np.asarray(wav, dtype=np.float32), sr)
        if request.ref_text and request.ref_text.strip():
            prompt["ref_text"] = request.ref_text
        if request.voice:
            voice_lower = request.voice.lower()
            if voice_lower in server.uploaded_speakers and not has_inline_ref_audio:
                prompt["voice_name"] = voice_lower
                prompt["voice_created_at"] = server._voice_created_at(voice_lower)
        if request.language:
            prompt["lang"] = request.language
        if request.instructions:
            prompt["instruct"] = prepare_instruct(request.instructions)
        return PreparedRequest(prompt=prompt, tts_params={}, model_type="omnivoice")
