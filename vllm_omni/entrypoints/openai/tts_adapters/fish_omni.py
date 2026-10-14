# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fish Audio Drama 3 serving adapter."""

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from vllm.inputs import tokens_input
from vllm.utils.async_utils import make_async

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import (
    ARTTSAdapter,
    PreparedRequest,
    apply_max_new_tokens,
    conditioning_cache_salt,
)

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest


@register_tts_adapter
class FishOmniAdapter(ARTTSAdapter):
    stage_keys = frozenset({"fish_omni"})
    name = "fish_omni_tts"

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self._tokenizer = None
        self._build_prompt_async = make_async(self._build_prompt, executor=getattr(ctx.server, "_tts_executor", None))

    def _get_tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            engine_client = self.ctx.engine_client
            assert engine_client is not None, "fish_omni serving needs an AR engine client"
            self._tokenizer = AutoTokenizer.from_pretrained(engine_client.model_config.model)
        return self._tokenizer

    def _build_prompt(
        self,
        request: "OpenAICreateSpeechRequest",
        ref_audio_data: tuple[list[float], int] | None,
        has_inline_ref_audio: bool,
    ) -> dict[str, Any]:
        """Additional information for stage-0 preprocess, with a placeholder prompt
        sized to the exact prompt row count."""
        from vllm_omni.model_executor.models.fish_omni.fish_omni_model import FishOmniForConditionalGeneration

        info: dict[str, Any] = {"text": request.input}
        if ref_audio_data is not None and request.ref_text:
            wav_samples, sr = ref_audio_data
            info["ref_text"] = request.ref_text
            info["ref_audio_wav"] = torch.from_numpy(np.asarray(wav_samples, dtype=np.float32))
            info["ref_audio_sr"] = int(sr)
            server = self.ctx.server
            if request.voice is not None:
                voice_lower = request.voice.lower()
                if voice_lower in server.uploaded_speakers and not has_inline_ref_audio:
                    info["voice_name"] = voice_lower
                    info["voice_created_at"] = server._voice_created_at(voice_lower)
        prompt_len = FishOmniForConditionalGeneration.estimate_prompt_len_from_additional_information(
            info, tokenizer=self._get_tokenizer()
        )
        prompt = tokens_input(prompt_token_ids=[1] * prompt_len)
        prompt["additional_information"] = info
        return prompt

    def validate(self, request: "OpenAICreateSpeechRequest") -> str | None:
        server = self.ctx.server
        err = server._apply_uploaded_speaker(request)
        if err:
            return err
        if not request.input or not request.input.strip():
            return "Input text cannot be empty"
        if request.ref_audio is not None:
            fmt_err = server._validate_ref_audio_format(request.ref_audio)
            if fmt_err:
                return fmt_err
            if not request.ref_text or not request.ref_text.strip():
                return "Voice cloning requires 'ref_text' (transcript of the reference audio)"
        return self._validate_max_new_tokens(request)

    async def build(
        self, request: "OpenAICreateSpeechRequest", sampling_params_list: list, has_inline_ref_audio: bool
    ) -> PreparedRequest:
        ref_audio_data = None
        tts_params: dict = {}
        if request.ref_audio is not None:
            wav_list, sr, cache_key = await self.ctx.server._resolve_ref_audio(request.ref_audio)
            ref_audio_data = (wav_list, sr)
            tts_params["ref_audio_cache_key"] = cache_key
        prompt = await self._build_prompt_async(request, ref_audio_data, has_inline_ref_audio)
        prompt["cache_salt"] = conditioning_cache_salt(request, tts_params)
        return PreparedRequest(prompt=prompt, tts_params=tts_params, model_type="fish_omni")

    def apply_sampling_overrides(
        self,
        sampling_params_list: list,
        request: "OpenAICreateSpeechRequest",
        prompt: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> list:
        return apply_max_new_tokens(sampling_params_list, request)
