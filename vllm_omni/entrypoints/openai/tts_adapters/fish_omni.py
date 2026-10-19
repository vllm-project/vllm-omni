# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fish Audio Drama 3 serving adapter.

A voice-clone reference arrives as audio (``ref_audio``, encoded in Stage 0)
or already encoded (``ref_audio_codes``, ``[codebooks, frames]`` from the
model's codec): then nothing is fetched or encoded.
"""

import hashlib
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
    supports_reference_codes = True

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

    def _model_config(self) -> Any:
        if self.ctx.engine_client is None:
            raise RuntimeError("fish_omni speech requests need the engine client")
        return self.ctx.engine_client.model_config

    def _codebook_sizes(self) -> tuple[int, ...]:
        """Entries of each codebook of the deployed model's codec (semantic, then residual)."""
        from vllm_omni.model_executor.models.fish_speech.dac_utils import dac_codebook_sizes

        sizes = dac_codebook_sizes()
        decoder = self._model_config().hf_config.audio_decoder_config
        if int(decoder.num_codebooks) != len(sizes):
            raise RuntimeError(f"fish_omni: the model has {decoder.num_codebooks} codebooks, its codec {len(sizes)}")
        return sizes

    def _reference_codes_error(self, rows: list[list[int]]) -> str | None:
        """Why ``ref_audio_codes`` rows do not fit the codec (each codebook's size), if they do not."""
        sizes = self._codebook_sizes()
        if len(rows) != len(sizes):
            return f"'ref_audio_codes' must be shaped [codebooks={len(sizes)}, frames], got {len(rows)} rows"
        for index, (row, size) in enumerate(zip(rows, sizes, strict=True)):
            if min(row) < 0 or max(row) >= size:
                kind = "semantic" if index == 0 else "residual"
                return f"'ref_audio_codes' row {index} ({kind} codebook) values must be in [0, {size - 1}]"
        return None

    def _build_prompt(
        self,
        request: "OpenAICreateSpeechRequest",
        ref_audio_data: tuple[list[float], int] | None,
        has_inline_ref_audio: bool,
        ref_codes: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Additional information for stage-0 preprocess, with a placeholder prompt
        sized to the exact prompt row count."""
        from vllm_omni.model_executor.models.fish_omni.fish_omni_model import FishOmniForConditionalGeneration

        info: dict[str, Any] = {"text": request.input}
        if ref_codes is not None:
            info["ref_text"] = request.ref_text
            # The model takes reference codes as [frames, codebooks].
            info["ref_codes"] = torch.from_numpy(np.ascontiguousarray(ref_codes.T))
        elif ref_audio_data is not None and request.ref_text:
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
        max_model_len = int(self._model_config().max_model_len)
        if prompt_len >= max_model_len:
            raise ValueError(f"the prompt is {prompt_len} rows; the model's limit is {max_model_len - 1}")
        prompt = tokens_input(prompt_token_ids=[1] * prompt_len)
        prompt["additional_information"] = info
        return prompt

    def validate(self, request: "OpenAICreateSpeechRequest") -> str | None:
        server = self.ctx.server
        if request.ref_audio_codes is not None:
            if request.voice:
                return "'ref_audio_codes' and an uploaded 'voice' are mutually exclusive"
            err = self._reference_codes_error(request.ref_audio_codes)
            if err:
                return err
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
        ref_codes = None
        tts_params: dict = {}
        if request.ref_audio_codes is not None:
            # validate() checked every value against its codebook, so the one
            # int64 conversion (for the prompt and the digest) cannot overflow.
            ref_codes = np.asarray(request.ref_audio_codes, dtype="<i8")
            digest = hashlib.blake2b(ref_codes.tobytes(), digest_size=16).hexdigest()
            tts_params["ref_codes_cache_key"] = f"v1:{ref_codes.shape[0]}x{ref_codes.shape[1]}:{digest}"
        elif request.ref_audio is not None:
            wav_list, sr, cache_key = await self.ctx.server._resolve_ref_audio(request.ref_audio)
            ref_audio_data = (wav_list, sr)
            tts_params["ref_audio_cache_key"] = cache_key
        try:
            prompt = await self._build_prompt_async(request, ref_audio_data, has_inline_ref_audio, ref_codes)
        except ValueError as exc:
            raise ValueError(f"Drama 3 request: {exc}") from exc
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
