# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Higgs-Audio v2 serving adapter."""

import asyncio
from typing import TYPE_CHECKING, Any

import numpy as np
from vllm.inputs import tokens_input

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import ARTTSAdapter, PreparedRequest, apply_max_new_tokens

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest


@register_tts_adapter
class HiggsAudioV2Adapter(ARTTSAdapter):
    @property
    def engine_client(self):
        return self.ctx.engine_client

    async def _resolve_ref_audio(self, ref_audio: str):
        return await self.ctx.server._resolve_ref_audio(ref_audio)

    async def _build_higgs_audio_v2_params(self, request: "OpenAICreateSpeechRequest"):
        """Build prompt_token_ids for higgs_audio_v2 via the upstream processor.

        Plain-text path: runs ``build_plain_text_prompt`` and returns the
        token-only prompt. Voice-clone path (``ref_audio`` + ``ref_text``):
        resolves the reference clip via ``_resolve_ref_audio``, runs
        ``build_voice_clone_prompt`` (which encodes the clip through HF's
        ``HiggsAudioV2TokenizerModel`` loaded from the k2-fsa/OmniVoice
        ``audio_tokenizer/`` subdirectory), and attaches the encoded
        ``audio_input_ids`` + ``audio_input_ids_mask`` tensors via
        ``additional_information`` so the talker substitutes them at the
        prompt-side audio placeholders.
        """
        from vllm_omni.model_executor.models.higgs_audio_v2.higgs_audio_v2_tokenizer import (
            build_plain_text_prompt,
            build_voice_clone_prompt,
            input_ids_to_python_list,
        )

        processor = await self._resolve_higgs_audio_v2_processor()

        if request.ref_audio is None:
            inputs = await asyncio.to_thread(build_plain_text_prompt, processor, request.input)
            prompt_token_ids = input_ids_to_python_list(inputs)
            return tokens_input(prompt_token_ids=prompt_token_ids)

        wav_list, sr, _ = await self._resolve_ref_audio(request.ref_audio)
        wav = np.asarray(wav_list, dtype=np.float32)
        out = await asyncio.to_thread(
            build_voice_clone_prompt,
            processor,
            request.input,
            wav,
            int(sr),
            request.ref_text or "",
        )
        prompt = tokens_input(prompt_token_ids=out["prompt_token_ids"])
        # Pass tensors at the top level of additional_information (NOT list-
        # wrapped). ``vllm_omni.data_entry_keys.serialize_payload`` routes
        # bare ``torch.Tensor`` values through ``_serialize_tensor``; a list
        # containing tensors would fall into the ``list_data`` field which
        # msgspec cannot serialize and the tensors would be dropped over the
        # process boundary (silent voice-clone failure).
        prompt["additional_information"] = {
            "audio_input_ids": out["audio_input_ids"],
            "audio_input_ids_mask": out["audio_input_ids_mask"],
        }
        return prompt

    async def _resolve_higgs_audio_v2_processor(self):
        """Lazy-load the AutoProcessor for higgs_audio_v2 (once per serving instance)."""
        cached = getattr(self, "_higgs_audio_v2_processor", None)
        if cached is not None:
            return cached

        from transformers import AutoProcessor

        model_path = None
        for stage in self.engine_client.stage_configs:
            model_path = getattr(getattr(stage, "engine_args", None), "model", None)
            if model_path:
                break
        if model_path is None:
            # Fallback: the orchestrator stores the served model id on the engine
            # itself (set by AsyncOmniEngine.__init__). Stage-level engine_args
            # may not surface ``model`` when the deploy yaml doesn't set it per
            # stage (the CLI-passed model id is the single source of truth).
            model_path = getattr(self.engine_client, "model", None)
        if model_path is None:
            raise RuntimeError("higgs_audio_v2 serving could not resolve the model path from the engine stage configs")
        processor = AutoProcessor.from_pretrained(model_path)
        self._higgs_audio_v2_processor = processor
        return processor

    # ---- higgs-audio v3 ----

    stage_keys = frozenset({"higgs_audio_v2"})
    name = "higgs_audio_v2"

    def validate(self, request: "OpenAICreateSpeechRequest") -> str | None:
        """Validate higgs_audio_v2 request parameters. Returns error message or None.

        Accepted: plain text -> speech, or shallow voice clone via ``ref_audio``
        + ``ref_text`` (both required together). Still out of scope: preset
        ``voice``/``speaker`` selection, ``x_vector_only_mode`` /
        ``speaker_embedding`` helpers, ``task_type``/``language``/
        ``instructions``/``speed`` overrides, and multi-speaker ``[SPEAKERn]``
        tags inside the input body.
        """
        from vllm_omni.model_executor.models.higgs_audio_v2.higgs_audio_v2_tokenizer import (
            MULTI_SPEAKER_TAG_PATTERN,
        )

        server = self.ctx.server
        err = server._apply_uploaded_speaker(request)
        if err:
            return err

        if not request.input or not request.input.strip():
            return "higgs_audio_v2: input text cannot be empty"

        # Voice clone: ref_audio and ref_text must come together.
        if request.ref_audio is not None and not request.ref_text:
            return (
                "higgs_audio_v2 voice clone requires both 'ref_audio' and "
                "'ref_text'; received ref_audio without ref_text"
            )
        if request.ref_text and request.ref_audio is None:
            return (
                "higgs_audio_v2 voice clone requires both 'ref_audio' and "
                "'ref_text'; received ref_text without ref_audio"
            )

        if request.x_vector_only_mode is not None:
            return "higgs_audio_v2 v1 does not support 'x_vector_only_mode' (voice-cloning helper field)"
        if request.speaker_embedding is not None:
            return "higgs_audio_v2 v1 does not support 'speaker_embedding' (voice-cloning helper field)"
        if request.voice and request.ref_audio is None:
            # _apply_uploaded_speaker runs before this validator; if voice was
            # an uploaded speaker, ref_audio is now populated and ref_text is
            # backfilled from the speaker entry. A bare voice= with no
            # ref_audio means the name didn't resolve to an uploaded speaker
            # (and higgs has no built-in preset voices).
            return (
                "higgs_audio_v2 v1 does not support 'voice'/'speaker' selection for built-in voices; "
                f"upload a voice first via POST /v1/audio/voices, or use ref_audio + ref_text. "
                f"Got voice={request.voice!r}"
            )
        if request.instructions:
            return (
                "higgs_audio_v2 v1 does not support 'instructions' (voice "
                "style/emotion control); supply plain text instead"
            )
        if request.task_type is not None:
            return "higgs_audio_v2 v1 does not support 'task_type'; the model is single-mode plain text -> speech"
        if request.language is not None:
            return (
                "higgs_audio_v2 v1 does not accept 'language' overrides; the model infers language from the input text"
            )
        if request.speed is not None and request.speed != 1.0:
            return (
                "higgs_audio_v2 v1 does not support 'speed' adjustments; the audio is rendered at native rate (24 kHz)"
            )

        if MULTI_SPEAKER_TAG_PATTERN.search(request.input):
            return "higgs_audio_v2 v1 does not support multi-speaker [SPEAKERn] tags; remove the tag from the input"

        if request.max_new_tokens is not None:
            if request.max_new_tokens < self.max_new_tokens_min:
                return f"max_new_tokens must be at least {self.max_new_tokens_min}"

    async def build(
        self, request: "OpenAICreateSpeechRequest", sampling_params_list: list, has_inline_ref_audio: bool
    ) -> PreparedRequest:
        server = self.ctx.server
        prompt = await self._build_higgs_audio_v2_params(request)
        if request.voice:
            voice_lower = request.voice.lower()
            if voice_lower in server.uploaded_speakers and not has_inline_ref_audio:
                additional = prompt.setdefault("additional_information", {})
                additional["voice_name"] = voice_lower
                additional["voice_created_at"] = server._voice_created_at(voice_lower)
        return PreparedRequest(prompt=prompt, tts_params={}, model_type="higgs_audio_v2")

    def apply_sampling_overrides(
        self,
        sampling_params_list: list,
        request: "OpenAICreateSpeechRequest",
        prompt: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> list:
        return apply_max_new_tokens(sampling_params_list, request)
