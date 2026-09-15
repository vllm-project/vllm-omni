# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fish Speech serving adapter (retires the legacy ``_is_fish_speech`` flag)."""

import math
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
class FishSpeechAdapter(ARTTSAdapter):
    stage_keys = frozenset({"fish_speech_slow_ar"})
    name = "fish_tts"

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self._tokenizer = None
        self._build_prompt_async = make_async(self._build_prompt, executor=getattr(ctx.server, "_tts_executor", None))

    @staticmethod
    def _estimate_ref_code_len(ref_audio: object) -> int | None:
        from vllm_omni.model_executor.models.fish_speech.dac_utils import DAC_HOP_LENGTH, DAC_SAMPLE_RATE

        if not isinstance(ref_audio, (list, tuple)) or len(ref_audio) != 2:
            return None
        wav, sr = ref_audio
        sr = int(sr)
        if sr <= 0 or len(wav) <= 0:
            return None
        return max(1, math.ceil(max(1, math.ceil(len(wav) * DAC_SAMPLE_RATE / sr)) / DAC_HOP_LENGTH))

    def _get_tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.ctx.engine_client.model_config.model, trust_remote_code=True
            )
        return self._tokenizer

    def _estimate_prompt_len(self, text: str, ref_text: str, ref_audio: object) -> int:
        from vllm_omni.model_executor.models.fish_speech.prompt_utils import (
            estimate_fish_voice_clone_prompt_len_from_normalized,
        )

        try:
            semantic_len = self._estimate_ref_code_len(ref_audio)
            if semantic_len is None:
                raise ValueError("Failed to estimate Fish Speech semantic token length")
            return estimate_fish_voice_clone_prompt_len_from_normalized(
                self._get_tokenizer(), text, ref_text, semantic_len
            )
        except Exception as exc:
            import logging

            logging.getLogger(__name__).warning(
                "Failed to estimate Fish Speech prompt length, using fallback 2048: %s", exc
            )
            return 2048

    def _build_prompt(
        self,
        request: "OpenAICreateSpeechRequest",
        ref_audio_data: tuple[list[float], int] | None,
        has_inline_ref_audio: bool,
    ) -> dict[str, Any]:
        """Build either the text-only or structured Fish Speech clone prompt.

        Structured clone metadata uses concrete scalar fields because model-side
        preprocess consumes one request directly; text-only metadata retains the
        legacy single-item list representation used across EngineCore IPC.
        """
        from vllm_omni.model_executor.models.fish_speech.prompt_utils import (
            build_fish_text_only_prompt_ids,
            normalize_fish_voice_clone_texts,
        )

        tokenizer = self._get_tokenizer()
        if ref_audio_data is None or not request.ref_text:
            prompt_ids, normalized_text = build_fish_text_only_prompt_ids(tokenizer, request.input)
            info: dict[str, Any] = {"text": [normalized_text]}
            if request.max_new_tokens is not None:
                info["max_new_tokens"] = [request.max_new_tokens]
            prompt = tokens_input(prompt_token_ids=prompt_ids)
            prompt["additional_information"] = info
            return prompt

        wav_samples, sr = ref_audio_data
        normalized_text, normalized_ref_text = normalize_fish_voice_clone_texts(request.input, request.ref_text)
        ph_len = self._estimate_prompt_len(normalized_text, normalized_ref_text, ref_audio_data)
        info = {
            "text": normalized_text,
            "ref_text": normalized_ref_text,
            "ref_audio_wav": torch.from_numpy(np.asarray(wav_samples, dtype=np.float32)),
            "ref_audio_sr": int(sr),
            "fish_structured_voice_clone": True,
        }
        server = self.ctx.server
        if request.voice is not None:
            voice_lower = request.voice.lower()
            if voice_lower in server.uploaded_speakers and not has_inline_ref_audio:
                info["voice_name"] = voice_lower
                info["voice_created_at"] = server._voice_created_at(voice_lower)
        if request.max_new_tokens is not None:
            info["max_new_tokens"] = request.max_new_tokens
        prompt = tokens_input(prompt_token_ids=[1] * ph_len)
        prompt["additional_information"] = info
        return prompt

    def validate(self, request: "OpenAICreateSpeechRequest") -> str | None:
        """Validate Fish Speech request parameters. Returns error message or None."""
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

        if request.max_new_tokens is not None:
            if request.max_new_tokens < self.max_new_tokens_min:
                return f"max_new_tokens must be at least {self.max_new_tokens_min}"
            if request.max_new_tokens > self.max_new_tokens_max:
                return f"max_new_tokens cannot exceed {self.max_new_tokens_max}"

        return None

    async def build(
        self, request: "OpenAICreateSpeechRequest", sampling_params_list: list, has_inline_ref_audio: bool
    ) -> PreparedRequest:
        server = self.ctx.server
        ref_audio_data = None
        tts_params: dict = {}
        if request.ref_audio is not None:
            wav_list, sr, cache_key = await server._resolve_ref_audio(request.ref_audio)
            ref_audio_data = (wav_list, sr)
            tts_params["ref_audio_cache_key"] = cache_key
        prompt = await self._build_prompt_async(request, ref_audio_data, has_inline_ref_audio)
        prompt["cache_salt"] = conditioning_cache_salt(request, tts_params)
        return PreparedRequest(prompt=prompt, tts_params=tts_params, model_type="fish_speech")

    def apply_sampling_overrides(
        self,
        sampling_params_list: list,
        request: "OpenAICreateSpeechRequest",
        prompt: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> list:
        return apply_max_new_tokens(sampling_params_list, request)
