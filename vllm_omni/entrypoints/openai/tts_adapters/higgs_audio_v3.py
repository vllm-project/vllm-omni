# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Higgs-Audio v3 serving adapter."""

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from vllm.inputs import tokens_input

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import (
    ARTTSAdapter,
    PreparedRequest,
    apply_max_new_tokens,
    conditioning_cache_salt,
    resolve_stage_model_path,
    stage_additional_config,
)
from vllm_omni.utils.shared_reference_encoder import ReferenceEncoderSpec

# Encoded references kept across requests (delayed codes, int64).
_REF_CODE_CACHE_MAX_BYTES = 64 * 1024 * 1024

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest


@register_tts_adapter
class HiggsAudioV3Adapter(ARTTSAdapter):
    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        additional_config = stage_additional_config(getattr(ctx.server, "_tts_stage", None))
        # Codec encoder processes for reference clips, shared by the server's
        # API processes; 0 encodes in the hosting API process itself.
        self._reference_workers = int(additional_config.get("ref_audio_encode_workers", 1))

    @property
    def engine_client(self):
        return self.ctx.engine_client

    async def _resolve_ref_audio(self, ref_audio: str):
        return await self.ctx.server._resolve_ref_audio(ref_audio)

    async def _build_higgs_audio_v3_params(self, request: "OpenAICreateSpeechRequest"):
        """Build prompt_token_ids for higgs_audio_v3.

        Plain-text path: builds ``[tts, text, tokens, audio]``.
        Voice-clone path: encodes reference audio, applies delay pattern, and
        records reference-audio prompt positions while submitting valid token IDs.
        """
        adapter = await self._resolve_higgs_audio_v3_adapter()

        if request.ref_audio is None:
            prompt_ids = adapter.build_prompt(request.input)
            return tokens_input(prompt_token_ids=prompt_ids)

        # Voice clone
        wav_list, sr, cache_key = await self._resolve_ref_audio(cast(str, request.ref_audio))
        ref_codes_delayed = await self._higgs_audio_v3_ref_codes(np.asarray(wav_list, dtype=np.float32), int(sr))

        prompt_ids = adapter.build_prompt(
            request.input,
            num_ref_tokens=int(ref_codes_delayed.shape[0]),
            reference_text=request.ref_text or None,
        )
        prompt_ids, audio_placeholder_positions = adapter.prepare_prompt_for_engine(prompt_ids)
        prompt = tokens_input(prompt_token_ids=prompt_ids)
        import torch

        prompt["additional_information"] = {
            "audio_input_ids": ref_codes_delayed.to(torch.long),
            "audio_input_ids_mask": torch.ones(ref_codes_delayed.shape[0], dtype=torch.bool),
            "audio_placeholder_positions": audio_placeholder_positions,
            "ref_audio_cache_key": cache_key,
        }
        prompt["cache_salt"] = conditioning_cache_salt(request, prompt["additional_information"])
        return prompt

    async def _higgs_audio_v3_ref_codes(self, wav: np.ndarray, sr: int) -> torch.Tensor:
        """Delayed reference codes ``[T + N - 1, N]``, cached by clip content across requests."""
        return (await self.encode_reference(wav, sr)).clone()

    def reference_encoder_spec(self) -> ReferenceEncoderSpec:
        model_path = resolve_stage_model_path(self.engine_client)
        return ReferenceEncoderSpec(
            name="higgs-v3-ref",
            factory="vllm_omni.model_executor.models.higgs_audio_v3.reference_encoder:HiggsAudioV3ReferenceEncoder",
            kwargs={},
            wire="vllm_omni.model_executor.models.higgs_audio_v3.reference_encoder:HIGGS_AUDIO_V3_REFERENCE_WIRE",
            namespace=f"higgs_audio_v3:{model_path}",
            nbytes=lambda codes: int(codes.numel() * codes.element_size()),
            workers=self._reference_workers,
            cache_bytes=_REF_CODE_CACHE_MAX_BYTES,
        )

    async def _resolve_higgs_audio_v3_adapter(self):
        """Lazy-load the tokenizer adapter for higgs_audio_v3."""
        cached = getattr(self, "_higgs_audio_v3_adapter", None)
        if cached is not None:
            return cached

        from transformers import AutoTokenizer

        from vllm_omni.model_executor.models.higgs_audio_v3.higgs_audio_v3_tokenizer import (
            HiggsAudioV3TokenizerAdapter,
        )

        model_path = resolve_stage_model_path(self.engine_client)
        if model_path is None:
            raise RuntimeError("higgs_audio_v3 serving could not resolve model path")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        adapter = HiggsAudioV3TokenizerAdapter(tokenizer)
        self._higgs_audio_v3_adapter = adapter
        return adapter

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
        if request.max_new_tokens is not None:
            if request.max_new_tokens < self.max_new_tokens_min:
                return f"max_new_tokens must be at least {self.max_new_tokens_min}"
        return None

    async def build(
        self, request: "OpenAICreateSpeechRequest", sampling_params_list: list, has_inline_ref_audio: bool
    ) -> PreparedRequest:
        prompt = await self._build_higgs_audio_v3_params(request)
        return PreparedRequest(prompt=prompt, tts_params={}, model_type="higgs_audio_v3")

    def apply_sampling_overrides(
        self,
        sampling_params_list: list,
        request: "OpenAICreateSpeechRequest",
        prompt: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> list:
        return apply_max_new_tokens(sampling_params_list, request)
