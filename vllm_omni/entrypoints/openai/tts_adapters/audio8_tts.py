# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Audio8 TTS Preview serving adapter for ``/v1/audio/speech``."""

import copy
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
from vllm_omni.model_executor.models.audio8_tts.codec_utils import (
    estimate_reference_code_frames,
)
from vllm_omni.model_executor.models.audio8_tts.prompt_utils import (
    build_text_only_prompt_ids,
    estimate_voice_clone_prompt_len,
    normalize_text,
)

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest


@register_tts_adapter
class Audio8TTSAdapter(ARTTSAdapter):
    """Audio8 TTS Preview (0.6B and 0.1B): text-only synthesis and zero-shot
    voice cloning. Both variants share the serving path (same prompt tokens and
    44.1 kHz codec); only the Slow AR backbone differs."""

    stage_keys = frozenset({"audio8_tts_slow_ar", "audio8_tts_01b_slow_ar"})
    name = "audio8_tts"

    def __init__(self, ctx: Any) -> None:
        super().__init__(ctx)
        self._cached_tokenizer: Any = None

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

        if request.max_new_tokens is not None:
            if request.max_new_tokens < self.max_new_tokens_min:
                return f"max_new_tokens must be at least {self.max_new_tokens_min}"
            if request.max_new_tokens > self.max_new_tokens_max:
                return f"max_new_tokens cannot exceed {self.max_new_tokens_max}"

        return None

    def apply_sampling_overrides(
        self,
        sampling_params_list: list,
        request: "OpenAICreateSpeechRequest",
        prompt: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> list:
        sampling_params_list = apply_max_new_tokens(sampling_params_list, request)
        # Seed the Fast AR (talker_mtp) residual sampling per request. The Slow AR
        # is already seeded via sampling_metadata.generators, but the residual
        # codebooks read tts_local_seed; without it they fall back to global RNG,
        # so an unseeded request's audio would depend on batch composition. Default
        # it from the deploy seed; an explicit request.seed still overrides it
        # downstream. Mirrors how serving handles qwen3_tts.
        if sampling_params_list:
            default_seed = sampling_params_list[0].seed
            if default_seed is not None:
                sampling_params_list = copy.deepcopy(sampling_params_list)
                stage0 = sampling_params_list[0]
                if stage0.extra_args is None:
                    stage0.extra_args = {}
                stage0.extra_args.setdefault("tts_local_seed", int(default_seed))
        return sampling_params_list

    async def build(
        self,
        request: "OpenAICreateSpeechRequest",
        sampling_params_list: list,
        has_inline_ref_audio: bool,
    ) -> PreparedRequest:
        del sampling_params_list
        server = self.ctx.server
        ref_audio_data = None
        if request.ref_audio is not None:
            wav_list, sample_rate = await server._resolve_ref_audio(request.ref_audio)
            ref_audio_data = (wav_list, sample_rate)
        # Prompt building tokenizes and, for voice clone, allocates tensors; keep
        # it off the event loop via the server's single-worker TTS executor.
        build_prompt = make_async(self._build_prompt, executor=server._tts_executor)
        prompt = await build_prompt(
            request,
            ref_audio_data=ref_audio_data,
            has_inline_ref_audio=has_inline_ref_audio,
        )
        tts_params: dict = {}
        prompt["cache_salt"] = conditioning_cache_salt(request, tts_params)
        return PreparedRequest(prompt=prompt, tts_params=tts_params, model_type=self.name)

    def _tokenizer(self) -> Any:
        """The Audio8 tokenizer, loaded once per process."""
        if self._cached_tokenizer is None:
            from transformers import AutoTokenizer

            model_name = self.ctx.server.engine_client.model_config.model
            self._cached_tokenizer = AutoTokenizer.from_pretrained(model_name)
        return self._cached_tokenizer

    def _estimate_prompt_len(self, text: str, ref_text: str, ref_audio: object) -> int:
        """Exact clone-prompt length, without encoding the reference audio.

        Deliberately not defensive: the placeholder this sizes must match the
        length ``preprocess()`` actually builds, so a wrong guess corrupts the
        request instead of failing it. Let errors surface to the caller.
        """
        if not isinstance(ref_audio, (list, tuple)) or len(ref_audio) != 2:
            raise ValueError("Audio8 TTS reference audio must be a (samples, sample_rate) pair")
        wav, sample_rate = ref_audio
        ref_frames = estimate_reference_code_frames(len(wav), int(sample_rate))
        return estimate_voice_clone_prompt_len(self._tokenizer(), text, ref_text, ref_frames)

    def _build_prompt(
        self,
        request: "OpenAICreateSpeechRequest",
        ref_audio_data: tuple[list[float], int] | None = None,
        *,
        has_inline_ref_audio: bool = False,
    ) -> dict[str, Any]:
        """Build the engine prompt for Audio8 TTS Preview.

        Text-only prompts are tokenized here. Voice cloning cannot be: the
        prompt embeds the reference audio's own codec codes, so the model-side
        ``preprocess`` builds it and this path only reserves a placeholder of the
        exact final length.
        """
        server = self.ctx.server
        if ref_audio_data is None or not request.ref_text:
            prompt_ids, normalized_text = build_text_only_prompt_ids(self._tokenizer(), request.input)
            # Scalars are list-wrapped for the text-only path, matching the
            # other TTS entrypoints' additional_information shape.
            additional_information: dict[str, Any] = {"text": [normalized_text]}
            if request.max_new_tokens is not None:
                additional_information["max_new_tokens"] = [request.max_new_tokens]
            prompt = tokens_input(prompt_token_ids=prompt_ids)
            prompt["additional_information"] = additional_information
            return prompt

        wav_samples, sample_rate = ref_audio_data
        normalized_text = normalize_text(request.input)
        normalized_ref_text = normalize_text(request.ref_text, add_default_speaker=True)
        placeholder_len = self._estimate_prompt_len(normalized_text, normalized_ref_text, ref_audio_data)

        # Structured clone: scalars (not list-wrapped) because model-side
        # preprocess() consumes these fields directly.
        additional_information = {
            "text": normalized_text,
            "ref_text": normalized_ref_text,
            "ref_audio_wav": torch.from_numpy(np.asarray(wav_samples, dtype=np.float32)),
            "ref_audio_sr": int(sample_rate),
            "audio8_structured_voice_clone": True,
        }
        if request.voice is not None:
            voice_lower = request.voice.lower()
            if voice_lower in server.uploaded_speakers and not has_inline_ref_audio:
                # Lets the model cache this speaker's encoded codes.
                additional_information["voice_name"] = voice_lower
                additional_information["voice_created_at"] = server._voice_created_at(voice_lower)
        if request.max_new_tokens is not None:
            additional_information["max_new_tokens"] = request.max_new_tokens
        prompt = tokens_input(prompt_token_ids=[1] * placeholder_len)
        prompt["additional_information"] = additional_information
        return prompt


__all__ = ["Audio8TTSAdapter"]
