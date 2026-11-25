# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Miso TTS serving adapter (AR ``engine_client`` backend).

The talker reads text, speaker id, sampling options and generation limits from
``additional_information``; the OpenAI ``voice`` field carries the numeric
speaker id (``"0"`` to ``"9"``).
"""

import asyncio
import math
from typing import TYPE_CHECKING, Any

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import ARTTSAdapter, PreparedRequest
from vllm_omni.transformers_utils.configs.miso_tts import (
    MISO_AUDIO_VOCAB_SIZE,
    MISO_DEFAULT_MAX_FRAMES,
    MISO_MAX_SEQ_LEN,
    MISO_TEXT_TOKENIZER,
)

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

MISO_SPEAKERS = tuple(str(i) for i in range(10))
MISO_DEFAULT_SPEAKER = "0"
# Talker prompt: the token ids are placeholders, the talker builds its own
# frame prompt from ``additional_information``.
MISO_PROMPT = "<|im_start|>assistant\n"
# The talker's tokenizer template wraps each text segment in BOS and EOS.
_NUM_SPECIAL_TOKENS = 2


def _sampling_options(request: "OpenAICreateSpeechRequest") -> dict[str, Any]:
    return request.extra_params if isinstance(request.extra_params, dict) else {}


@register_tts_adapter
class MisoTTSAdapter(ARTTSAdapter):
    stage_keys = frozenset({"miso_tts"})
    model_archs = frozenset({"MisoTTSTalkerForConditionalGeneration"})
    name = "miso_tts"
    max_new_tokens_max = MISO_MAX_SEQ_LEN - 1

    async def warmup(self) -> None:
        await asyncio.to_thread(self._load_text_tokenizer)

    def _load_text_tokenizer(self) -> Any:
        server = self.ctx.server
        if server._tts_tokenizer is None:
            from transformers import AutoTokenizer

            server._tts_tokenizer = AutoTokenizer.from_pretrained(MISO_TEXT_TOKENIZER)
        return server._tts_tokenizer

    def _load_supported_speakers(self) -> list[str]:
        return list(MISO_SPEAKERS)

    def _validate_sampling(self, request: "OpenAICreateSpeechRequest") -> str | None:
        options = _sampling_options(request)
        if options.get("top_p") is not None:
            return "Miso TTS does not support extra_params.top_p"
        temperature = options.get("temperature")
        if temperature is not None and (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature)
            or temperature <= 0
        ):
            return "extra_params.temperature must be a finite number greater than 0"
        top_k = options.get("top_k")
        if top_k is not None and (type(top_k) is not int or not 1 <= top_k <= MISO_AUDIO_VOCAB_SIZE):
            return f"extra_params.top_k must be an integer in [1,{MISO_AUDIO_VOCAB_SIZE}]"
        return None

    def _validate_prompt_budget(self, request: "OpenAICreateSpeechRequest") -> str | None:
        speaker = (request.voice or MISO_DEFAULT_SPEAKER).strip()
        text = f"[{speaker}] {request.input.strip()}"
        prompt_len = len(self._load_text_tokenizer().encode(text, add_special_tokens=False)) + _NUM_SPECIAL_TOKENS
        frames = MISO_DEFAULT_MAX_FRAMES if request.max_new_tokens is None else request.max_new_tokens
        if prompt_len + frames >= MISO_MAX_SEQ_LEN:
            return (
                f"Input is too long: {prompt_len} prompt tokens plus {frames} audio frames "
                f"must be less than {MISO_MAX_SEQ_LEN}"
            )
        return None

    def validate(self, request: "OpenAICreateSpeechRequest") -> str | None:
        if not request.input or not request.input.strip():
            return "Input text cannot be empty"
        if request.ref_audio is not None:
            return "Miso TTS does not support 'ref_audio'; use 'voice' with a speaker id"
        if request.voice is not None and request.voice.strip() not in MISO_SPEAKERS:
            return f"Invalid speaker '{request.voice}'. Supported: {', '.join(MISO_SPEAKERS)}"
        return (
            self._validate_max_new_tokens(request)
            or self._validate_sampling(request)
            or self._validate_prompt_budget(request)
        )

    async def build(
        self, request: "OpenAICreateSpeechRequest", sampling_params_list: list, has_inline_ref_audio: bool
    ) -> PreparedRequest:
        speaker = int((request.voice or MISO_DEFAULT_SPEAKER).strip())
        additional: dict[str, list[Any]] = {"text": [request.input], "speaker": [speaker]}
        if request.max_new_tokens is not None:
            additional["max_generation_frames"] = [request.max_new_tokens]
        if request.seed is not None:
            additional["seed"] = [request.seed]
        options = _sampling_options(request)
        if options.get("temperature") is not None:
            additional["temperature"] = [float(options["temperature"])]
        if options.get("top_k") is not None:
            additional["topk"] = [options["top_k"]]
        prompt = {"prompt": MISO_PROMPT, "additional_information": additional}
        return PreparedRequest(prompt=prompt, tts_params={}, model_type="miso_tts")
