# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NeuTTS-Air TTS serving adapter."""

import copy
from typing import TYPE_CHECKING

import numpy as np

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import (
    ARTTSAdapter,
    PreparedRequest,
)

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import (
        OpenAICreateSpeechRequest,
    )


NEUTTS_MIN_NEW_TOKENS = 50
NEUTTS_DEFAULT_MAX_NEW_TOKENS = 512
NEUTTS_SPEECH_GENERATION_END_TOKEN_ID = 151670


@register_tts_adapter
class NeuTTSAirAdapter(ARTTSAdapter):
    """Translate OpenAI speech requests into the NeuTTS-Air pipeline input."""

    stage_keys = frozenset({"neucodec"})
    name = "neutts_air"
    max_new_tokens_min = NEUTTS_MIN_NEW_TOKENS

    def validate(self, request: "OpenAICreateSpeechRequest") -> str | None:
        if not request.input or not request.input.strip():
            return "Input text cannot be empty"
        if request.ref_audio is None:
            return "NeuTTS-Air requires 'ref_audio' (reference audio for voice cloning)"
        if isinstance(request.ref_audio, list):
            return "NeuTTS-Air requires exactly one 'ref_audio'"

        fmt_err = self.ctx.server._validate_ref_audio_format(request.ref_audio)
        if fmt_err:
            return fmt_err
        if not request.ref_text or not request.ref_text.strip():
            return "NeuTTS-Air requires 'ref_text' (transcript of the reference audio)"
        if request.max_new_tokens is not None:
            if request.max_new_tokens < self.max_new_tokens_min:
                return f"max_new_tokens must be at least {self.max_new_tokens_min}"
            if request.max_new_tokens > self.max_new_tokens_max:
                return f"max_new_tokens cannot exceed {self.max_new_tokens_max}"
        return None

    async def build(
        self,
        request: "OpenAICreateSpeechRequest",
        sampling_params_list: list,
        has_inline_ref_audio: bool,
    ) -> PreparedRequest:
        del sampling_params_list, has_inline_ref_audio
        assert isinstance(request.ref_audio, str)
        wav_list, sample_rate = await self.ctx.server._resolve_ref_audio(request.ref_audio)
        prompt = {
            "prompt": request.input,
            "multi_modal_data": {
                "audio": (np.asarray(wav_list, dtype=np.float32), sample_rate),
            },
            "mm_processor_kwargs": {
                "ref_text": request.ref_text,
            },
        }
        return PreparedRequest(
            prompt=prompt,
            tts_params={},
            model_type="neutts_air",
        )

    def apply_sampling_overrides(
        self,
        sampling_params_list: list,
        request: "OpenAICreateSpeechRequest",
    ) -> list:
        if not sampling_params_list:
            return sampling_params_list

        params = copy.deepcopy(sampling_params_list)
        stage0 = params[0]
        requested_max = request.max_new_tokens
        if requested_max is not None:
            stage0.max_tokens = int(requested_max)
        elif stage0.max_tokens is None or stage0.max_tokens < NEUTTS_MIN_NEW_TOKENS:
            stage0.max_tokens = NEUTTS_DEFAULT_MAX_NEW_TOKENS
        stage0.min_tokens = NEUTTS_MIN_NEW_TOKENS
        stage0.detokenize = False
        stage0.stop_token_ids = [NEUTTS_SPEECH_GENERATION_END_TOKEN_ID]
        stage0.ignore_eos = True

        if len(params) > 1:
            stage1 = params[1]
            stage1.max_tokens = 1
            stage1.detokenize = False
        return params
