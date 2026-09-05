# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Breeze-TTS-2 serving adapter for synchronous full-payload generation."""

import asyncio
from typing import TYPE_CHECKING, Any

import numpy as np

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import (
    ARTTSAdapter,
    PreparedRequest,
    apply_max_new_tokens,
    conditioning_cache_salt,
)
from vllm_omni.model_executor.models.breeze_tts_2.audio_tokenizer import (
    BreezeReferenceAudioTokenizer,
)
from vllm_omni.model_executor.models.breeze_tts_2.prompt_builder import (
    BreezeTTS2PromptBuilder,
)

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest


@register_tts_adapter
class BreezeTTS2Adapter(ARTTSAdapter):
    """Build Breeze prompts before scheduler submission.

    The adapter owns only request-facing concerns. Model workers still own the
    stage-0 embeddings and stage-1 waveform decode.
    """

    stage_keys = frozenset({"breeze_tts_2"})
    model_archs = frozenset({"BreezeForConditionalGeneration"})
    name = "breeze_tts_2"
    # Breeze's architecture is unique, while its stage id is namespaced and not
    # shared.  Resolve architecture-only deployments ahead of generic/default
    # stage-key detectors without competing with VoxCPM's priority-10 rule.
    detect_priority = 9
    supported_output_sample_rates = frozenset({24000})

    def _load_supported_speakers(self) -> set[str]:
        # Breeze accepts arbitrary speaker tags such as S0/S1 in its prompt;
        # there is no finite checkpoint speaker inventory to advertise.
        return set()

    def _load_codec_frame_rate(self) -> float | None:
        config = self.ctx.engine_client.model_config.hf_config
        codec = getattr(config, "codec_config", None)
        if isinstance(codec, dict):
            frame_rate = codec.get("_frame_rate")
            if frame_rate is not None:
                return float(frame_rate)
        return 12.5

    def validate(self, request: "OpenAICreateSpeechRequest") -> str | None:
        if not request.input or not request.input.strip():
            return "Breeze-TTS-2 input text cannot be empty"
        if request.ref_text is not None and not request.ref_text.strip():
            return "Breeze-TTS-2 ref_text cannot be empty"
        if request.ref_text is not None and request.ref_audio is None:
            return "Breeze-TTS-2 ref_text requires ref_audio"
        if request.ref_audio is not None and (request.ref_text is None or not request.ref_text.strip()):
            return "Breeze-TTS-2 ref_audio requires ref_text"
        if request.task_type == "Base" and request.ref_audio is None:
            return "Breeze-TTS-2 Base task requires ref_audio and ref_text"
        if request.task_type == "VoiceDesign":
            return "Breeze-TTS-2 does not support task_type=VoiceDesign"
        if request.speed is not None and request.speed != 1.0:
            return "Breeze-TTS-2 does not support speed adjustment"
        extra_params = request.extra_params or {}
        guidance_scale = extra_params.get("guidance_scale", extra_params.get("cfg_scale", 1.0))
        try:
            unsupported_guidance = guidance_scale is not None and float(guidance_scale) != 1.0
        except (TypeError, ValueError):
            return "Breeze-TTS-2 guidance_scale must be a number"
        if unsupported_guidance:
            return "Breeze-TTS-2 currently supports only guidance_scale=1.0"
        if extra_params.get("negative_prompt"):
            return "Breeze-TTS-2 does not support negative_prompt/CFG yet"
        if request.max_new_tokens is not None:
            if request.max_new_tokens < self.max_new_tokens_min:
                return f"max_new_tokens must be at least {self.max_new_tokens_min}"
            if request.max_new_tokens > self.max_new_tokens_max:
                return f"max_new_tokens cannot exceed {self.max_new_tokens_max}"
        if request.ref_audio is not None:
            ref_audio = request.ref_audio[0] if isinstance(request.ref_audio, list) else request.ref_audio
            return self.ctx.server._validate_ref_audio_format(ref_audio)
        return None

    async def build(
        self,
        request: "OpenAICreateSpeechRequest",
        sampling_params_list: list,
        has_inline_ref_audio: bool,
    ) -> PreparedRequest:
        del has_inline_ref_audio
        server = self.ctx.server
        builder = await self._get_builder()
        template = self._resolve_template(request)
        payload: dict[str, Any] = {
            "text": request.input,
            "template": template,
            "speaker": request.voice or "S0",
        }
        if request.instructions:
            payload["instruction"] = request.instructions
        if request.ref_text:
            payload["ref_text"] = request.ref_text

        if request.ref_audio is not None:
            ref_audio = request.ref_audio[0] if isinstance(request.ref_audio, list) else request.ref_audio
            wav_list, sr, cache_key = await server._resolve_ref_audio(ref_audio)
            payload["ref_audio"] = np.asarray(wav_list, dtype=np.float32)
            payload["ref_audio_sample_rate"] = int(sr)
        else:
            cache_key = None

        prompt = await asyncio.to_thread(builder.build, payload, template=template)
        # The AR scheduler's completion budget is measured in codebook-0
        # frames for Breeze. Preserve it in the mutable request metadata so
        # the talker can emit its own EOS before a scheduler length cutoff.
        max_new_frames = request.max_new_tokens
        if max_new_frames is None and sampling_params_list:
            max_new_frames = getattr(sampling_params_list[0], "max_tokens", None)
        if max_new_frames is not None:
            prompt["additional_information"]["breeze_max_new_frames"] = int(max_new_frames)
        tts_params = {
            "template": [template],
            "text": [request.input],
        }
        if cache_key:
            tts_params["ref_audio_cache_key"] = cache_key
        prompt["cache_salt"] = conditioning_cache_salt(request, tts_params)
        return PreparedRequest(prompt=prompt, tts_params=tts_params, model_type=self.name)

    async def _get_builder(self) -> BreezeTTS2PromptBuilder:
        server = self.ctx.server
        cached = getattr(server, "_breeze_tts_2_prompt_builder", None)
        if cached is not None:
            return cached
        # Multiple speech requests can arrive while the first tokenizer load
        # is still in progress.  Serialize the one-time CPU initialization so
        # model files are not opened and decoded repeatedly.
        lock = getattr(server, "_breeze_tts_2_prompt_builder_lock", None)
        if lock is None:
            lock = asyncio.Lock()
            server._breeze_tts_2_prompt_builder_lock = lock
        async with lock:
            cached = getattr(server, "_breeze_tts_2_prompt_builder", None)
            if cached is not None:
                return cached
            model_path = server.engine_client.model_config.model
            config = server.engine_client.model_config.hf_config
            audio_tokenizer = await asyncio.to_thread(
                BreezeReferenceAudioTokenizer.from_pretrained,
                model_path,
                num_codebooks=int(getattr(config, "num_codebooks", 16)),
                codebook_size=self._codebook_size(config),
                device_map="cpu",
            )
            builder = await asyncio.to_thread(
                BreezeTTS2PromptBuilder.from_pretrained,
                model_path,
                config,
                reference_audio_encoder=audio_tokenizer,
            )
            server._breeze_tts_2_prompt_builder = builder
            return builder

    @staticmethod
    def _codebook_size(config: Any) -> int:
        codec = getattr(config, "codec_config", None)
        if isinstance(codec, dict):
            return int(codec.get("codebook_size", 2048))
        return int(getattr(codec, "codebook_size", 2048))

    @staticmethod
    def _resolve_template(request: "OpenAICreateSpeechRequest") -> str:
        if request.ref_audio is not None:
            return "ref_edit_tata" if request.instructions else "ref_clone_tata"
        return "tts_instruction" if request.instructions else "tts_plain"

    def apply_sampling_overrides(
        self,
        sampling_params_list: list,
        request: "OpenAICreateSpeechRequest",
        prompt: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> list:
        del prompt, request_id
        return apply_max_new_tokens(sampling_params_list, request)


__all__ = ["BreezeTTS2Adapter"]
