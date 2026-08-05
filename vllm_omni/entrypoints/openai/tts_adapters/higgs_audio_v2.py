# SPDX-License-Identifier: Apache-2.0
"""Higgs-Audio v2 serving adapter."""

from typing import TYPE_CHECKING

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import ARTTSAdapter, PreparedRequest

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest


@register_tts_adapter
class HiggsAudioV2Adapter(ARTTSAdapter):
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
        prompt = await server._build_higgs_audio_v2_params(request)
        if request.voice:
            voice_lower = request.voice.lower()
            if voice_lower in server.uploaded_speakers and not has_inline_ref_audio:
                additional = prompt.setdefault("additional_information", {})
                additional["voice_name"] = voice_lower
                additional["voice_created_at"] = server._voice_created_at(voice_lower)
        return PreparedRequest(prompt=prompt, tts_params={}, model_type="higgs_audio_v2")
