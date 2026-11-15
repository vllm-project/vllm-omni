# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Miso TTS serving adapter (AR ``engine_client`` backend).

The talker reads text, speaker id and generation limits from
``additional_information``; the OpenAI ``voice`` field carries the numeric
speaker id (``"0"`` to ``"9"``).
"""

from typing import TYPE_CHECKING

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import ARTTSAdapter, PreparedRequest

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

MISO_SPEAKERS = tuple(str(i) for i in range(10))
MISO_DEFAULT_SPEAKER = "0"
# Talker prompt: the token ids are placeholders, the talker builds its own
# frame prompt from ``additional_information``.
MISO_PROMPT = "<|im_start|>assistant\n"


@register_tts_adapter
class MisoTTSAdapter(ARTTSAdapter):
    stage_keys = frozenset({"miso_tts"})
    model_archs = frozenset({"MisoTTSTalkerForConditionalGeneration"})
    name = "miso_tts"

    def _load_supported_speakers(self) -> list[str]:
        return list(MISO_SPEAKERS)

    def validate(self, request: "OpenAICreateSpeechRequest") -> str | None:
        if not request.input or not request.input.strip():
            return "Input text cannot be empty"
        if request.ref_audio is not None:
            return "Miso TTS does not support 'ref_audio'; use 'voice' with a speaker id"
        if request.voice is not None and request.voice.strip() not in MISO_SPEAKERS:
            return f"Invalid speaker '{request.voice}'. Supported: {', '.join(MISO_SPEAKERS)}"
        return self._validate_max_new_tokens(request)

    async def build(
        self, request: "OpenAICreateSpeechRequest", sampling_params_list: list, has_inline_ref_audio: bool
    ) -> PreparedRequest:
        speaker = int((request.voice or MISO_DEFAULT_SPEAKER).strip())
        additional: dict = {"text": [request.input], "speaker": [speaker]}
        if request.max_new_tokens is not None:
            additional["max_generation_frames"] = [request.max_new_tokens]
        prompt = {"prompt": MISO_PROMPT, "additional_information": additional}
        return PreparedRequest(prompt=prompt, tts_params={}, model_type="miso_tts")
