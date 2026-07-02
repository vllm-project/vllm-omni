# SPDX-License-Identifier: Apache-2.0
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
        # Dispatcher routes to the step_audio2 case (non-empty input check).
        if not request.input or not request.input.strip():
            return "Input text cannot be empty"
        return None

    async def build(
        self, request: "OpenAICreateSpeechRequest", sampling_params_list: list, has_inline_ref_audio: bool
    ) -> PreparedRequest:
        """Build prompt for Step-Audio2 TTS.

        Constructs the chat prompt with ``<tts_start>`` as the last token
        of the assistant turn (without ``<|im_end|>``), so the thinker
        continues generating audio tokens.

        Prompt format::
            <|im_start|>system\\n{system_prompt}<|im_end|>\\n
            <|im_start|>user\\n{input_text}<|im_end|>\\n
            <|im_start|>assistant\\n<tts_start>
        """
        system_prompt = getattr(request, "instructions", None) or "You are a voice assistant. Read the text aloud."
        text = request.input

        raw_prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{text}<|im_end|>\n"
            f"<|im_start|>assistant\n<tts_start>"
        )
        prompt = {"prompt": raw_prompt}
        return PreparedRequest(prompt=prompt, tts_params={}, model_type="step_audio2")
