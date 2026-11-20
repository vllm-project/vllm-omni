# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Step-Audio2 serving adapter."""

from typing import TYPE_CHECKING

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import ARTTSAdapter, PreparedRequest

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest


@register_tts_adapter
class StepAudio2Adapter(ARTTSAdapter):
    """Adapter for Step-Audio2 (AR ``engine_client`` backend).

    Step-Audio2 runs a single thinker stage that emits audio tokens after a
    ``<tts_start>`` marker; prompt building is a synchronous chat-template
    construction with no uploaded-voice / ref-audio handling.
    """

    stage_keys = frozenset({"step_audio2_thinker"})
    name = "step_audio2"

    def validate(self, request: "OpenAICreateSpeechRequest") -> str | None:
        if not request.input or not request.input.strip():
            return "Input text cannot be empty"
        return None

    async def build(
        self, request: "OpenAICreateSpeechRequest", sampling_params_list: list, has_inline_ref_audio: bool
    ) -> PreparedRequest:
        """Build a chat prompt ending at ``<tts_start>``.

        The assistant turn deliberately omits ``<|EOT|>`` so the thinker
        continues by generating audio tokens.
        """
        system_prompt = request.instructions or "以自然的语速读出下面的文字。"
        prompt = {
            "prompt": (
                f"<|BOT|>system\n{system_prompt}<|EOT|>"
                f"<|BOT|>human\n{request.input}<|EOT|>"
                "<|BOT|>assistant\n<tts_start>"
            )
        }
        return PreparedRequest(prompt=prompt, tts_params={}, model_type="step_audio2")
