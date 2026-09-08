# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Higgs-Audio v3 serving adapter."""

import asyncio
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from vllm.inputs import tokens_input

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import (
    ARTTSAdapter,
    PreparedRequest,
    apply_max_new_tokens,
    conditioning_cache_salt,
)

_REF_CODE_CACHE_MAX_ENTRIES = 256
_REF_CODE_CACHE_MAX_BYTES = 64 * 1024 * 1024

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest


@register_tts_adapter
class HiggsAudioV3Adapter(ARTTSAdapter):
    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self._higgs_audio_v3_ref_code_cache: OrderedDict[str, tuple[torch.Tensor, int]] = OrderedDict()
        self._higgs_audio_v3_ref_code_cache_bytes = 0
        self._higgs_audio_v3_ref_code_inflight: dict[str, asyncio.Task[torch.Tensor]] = {}

    @property
    def engine_client(self):
        return self.ctx.engine_client

    async def _resolve_ref_audio(self, ref_audio: str):
        return await self.ctx.server._resolve_ref_audio(ref_audio)

    def _get_resolved_ref_audio_artifact_key(self, cache_key: str):
        return self.ctx.server._get_resolved_ref_audio_artifact_key(cache_key)

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
        from vllm_omni.model_executor.models.higgs_audio_v3.higgs_audio_v3_tokenizer import (
            apply_delay_pattern,
            encode_reference_audio,
        )

        wav_list, sr, cache_key = await self._resolve_ref_audio(request.ref_audio)
        artifact_key = self._get_resolved_ref_audio_artifact_key(cache_key)
        wav = np.asarray(wav_list, dtype=np.float32)
        ref_codes_delayed, cache_hit, inflight_wait = await self._resolve_higgs_audio_v3_ref_codes(
            artifact_key,
            wav,
            int(sr),
            encode_reference_audio,
            apply_delay_pattern,
        )
        del cache_hit, inflight_wait

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

    async def _resolve_higgs_audio_v3_ref_codes(
        self,
        artifact_key: str | None,
        wav: np.ndarray,
        sr: int,
        encode_reference_audio,
        apply_delay_pattern,
    ) -> tuple[torch.Tensor, bool, bool]:
        ref_codes_delayed = self._get_higgs_audio_v3_ref_codes(artifact_key)
        if ref_codes_delayed is not None:
            return ref_codes_delayed, True, False
        if not artifact_key:
            ref_codes_raw = await asyncio.to_thread(encode_reference_audio, wav, sr)
            return apply_delay_pattern(ref_codes_raw), False, False

        task = self._higgs_audio_v3_ref_code_inflight.get(artifact_key)
        if task is not None:
            return (await task).clone(), False, True

        async def _encode_and_cache() -> torch.Tensor:
            ref_codes_raw = await asyncio.to_thread(encode_reference_audio, wav, sr)
            delayed = apply_delay_pattern(ref_codes_raw)
            self._put_higgs_audio_v3_ref_codes(artifact_key, delayed)
            cached = self._get_higgs_audio_v3_ref_codes(artifact_key)
            return cached if cached is not None else delayed.detach().to("cpu", dtype=torch.long).contiguous()

        task = asyncio.create_task(_encode_and_cache())
        self._higgs_audio_v3_ref_code_inflight[artifact_key] = task
        try:
            return (await task).clone(), False, False
        finally:
            if self._higgs_audio_v3_ref_code_inflight.get(artifact_key) is task:
                self._higgs_audio_v3_ref_code_inflight.pop(artifact_key, None)

    def _get_higgs_audio_v3_ref_codes(self, artifact_key: str | None) -> torch.Tensor | None:
        if not artifact_key:
            return None
        cached = self._higgs_audio_v3_ref_code_cache.get(artifact_key)
        if cached is None:
            return None
        self._higgs_audio_v3_ref_code_cache.move_to_end(artifact_key)
        return cached[0].clone()

    def _put_higgs_audio_v3_ref_codes(self, artifact_key: str, codes: torch.Tensor) -> None:
        if _REF_CODE_CACHE_MAX_ENTRIES <= 0 or _REF_CODE_CACHE_MAX_BYTES <= 0 or not artifact_key:
            return
        cached_codes = codes.detach().to("cpu", dtype=torch.long).contiguous()
        size = int(cached_codes.numel() * cached_codes.element_size())
        if size > _REF_CODE_CACHE_MAX_BYTES:
            return
        previous = self._higgs_audio_v3_ref_code_cache.pop(artifact_key, None)
        if previous is not None:
            self._higgs_audio_v3_ref_code_cache_bytes -= previous[1]
        self._higgs_audio_v3_ref_code_cache[artifact_key] = (cached_codes, size)
        self._higgs_audio_v3_ref_code_cache_bytes += size
        while len(self._higgs_audio_v3_ref_code_cache) > _REF_CODE_CACHE_MAX_ENTRIES:
            _, (_, old_size) = self._higgs_audio_v3_ref_code_cache.popitem(last=False)
            self._higgs_audio_v3_ref_code_cache_bytes -= old_size
        while self._higgs_audio_v3_ref_code_cache_bytes > _REF_CODE_CACHE_MAX_BYTES:
            _, (_, old_size) = self._higgs_audio_v3_ref_code_cache.popitem(last=False)
            self._higgs_audio_v3_ref_code_cache_bytes -= old_size

    async def _resolve_higgs_audio_v3_adapter(self):
        """Lazy-load the tokenizer adapter for higgs_audio_v3."""
        cached = getattr(self, "_higgs_audio_v3_adapter", None)
        if cached is not None:
            return cached

        from transformers import AutoTokenizer

        from vllm_omni.model_executor.models.higgs_audio_v3.higgs_audio_v3_tokenizer import (
            HiggsAudioV3TokenizerAdapter,
        )

        model_path = None
        for stage in self.engine_client.stage_configs:
            model_path = getattr(getattr(stage, "engine_args", None), "model", None)
            if model_path:
                break
        if model_path is None:
            model_path = getattr(self.engine_client, "model", None)
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
        if request.ref_audio is not None and not request.ref_text:
            # Voice clone ref_text is optional for v3 (improves fidelity but not required)
            pass
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
