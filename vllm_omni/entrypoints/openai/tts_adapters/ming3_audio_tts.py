# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Converted Ming3 Audio standalone TTS serving adapter."""

from typing import TYPE_CHECKING, Any

from vllm.inputs import tokens_input
from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.reference_audio import reference_audio_uri
from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import ARTTSAdapter, PreparedRequest
from vllm_omni.entrypoints.openai.tts_adapters.ming_tts import parse_ming_instruction_fields
from vllm_omni.model_executor.models.ming_flash_omni.reference_audio import (
    pack_reference_audio,
    validate_reference_audio,
    validate_reference_audio_count,
)

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

logger = init_logger(__name__)


@register_tts_adapter
class Ming3AudioTTSAdapter(ARTTSAdapter):
    # IP names are caption controls; clone references arrive with each request.
    stage_keys = frozenset({"ming3_tts"})
    name = "ming3_audio_tts"

    def validate(self, request: "OpenAICreateSpeechRequest") -> str | None:
        """Validate Ming3 Audio standalone-talker request parameters."""
        server = self.ctx.server
        if not request.input or not request.input.strip():
            return "Input text cannot be empty"
        if request.instructions is not None:
            if not isinstance(request.instructions, str):
                return "instructions must be a string"
            if len(request.instructions) > server._max_instructions_length:
                return f"instructions exceeds max length {server._max_instructions_length}"

        if request.task_type is not None:
            return "'task_type' is not supported for Ming3 Audio TTS"
        if request.initial_codec_chunk_frames is not None:
            return "'initial_codec_chunk_frames' is not supported for Ming3 Audio TTS"
        if request.voice and request.ref_audio is not None:
            return "Choose either an IP voice or ref_audio for voice cloning"
        from vllm_omni.model_executor.models.ming_flash_omni.omni_voice import DEFAULT_IP_VOICE, ip_voice_name

        # Match Chat/Realtime even if an old registration used a reserved IP name.
        ip_name = ip_voice_name(request.voice) if request.voice else None
        if ip_name is not None:
            request.voice = ip_name
        elif request.voice and request.voice.lower() in server.uploaded_speakers:
            error = server._apply_uploaded_speaker(request)
            if error:
                return error
            request.voice = None
        if request.ref_text is not None and request.ref_audio is None:
            return "ref_text requires ref_audio"

        if request.speaker_embedding is not None:
            vectors = request.speaker_embedding
            if not vectors:
                return "Ming speaker embeddings must contain 192 values per speaker"
            if isinstance(vectors[0], list):
                invalid_shape = any(not isinstance(vector, list) or len(vector) != 192 for vector in vectors)
            else:
                invalid_shape = len(vectors) != 192
            if invalid_shape:
                return "Ming speaker embeddings must contain 192 values per speaker"
        if (
            request.x_vector_only_mode
            and request.ref_audio is None
            and request.speaker_embedding is None
            and not request.voice
        ):
            return "x_vector_only_mode requires reference audio or speaker embeddings"

        if request.max_new_tokens is not None and request.max_new_tokens <= 0:
            return "'max_new_tokens' must be a positive integer"
        if not request.voice and request.ref_audio is None and request.speaker_embedding is None:
            request.voice = DEFAULT_IP_VOICE
        return None

    async def build(
        self, request: "OpenAICreateSpeechRequest", sampling_params_list: list, has_inline_ref_audio: bool
    ) -> PreparedRequest:
        """Build the standalone Ming3 Audio TTS prompt."""
        from vllm_omni.model_executor.models.ming_flash_omni.prompt_utils import (
            DEFAULT_PROMPT,
            create_instruction,
        )

        server = self.ctx.server
        # ``instructions`` accepts either plain style text or a JSON object.
        # Unknown caption keys are filtered by ``create_instruction``.
        caption_fields = parse_ming_instruction_fields(server, request, include_voice=False) or {}
        if request.voice:
            caption_fields["IP"] = request.voice
        has_spk_emb = request.speaker_embedding is not None
        reference_audio = None
        if request.ref_audio is not None:
            sources = request.ref_audio if isinstance(request.ref_audio, list) else [request.ref_audio]
            validate_reference_audio_count(len(sources))
            reference_audio = []
            for uri in sources:
                samples, sr, _ = await server._resolve_ref_audio_array(reference_audio_uri(uri))
                reference_audio.append((validate_reference_audio(samples, sr), sr))
        has_reference = has_spk_emb or bool(reference_audio)
        audio_mode = (
            ("speaker_only" if request.x_vector_only_mode or request.ref_text is None else "continuation")
            if has_reference
            else "none"
        )
        prompt_text = DEFAULT_PROMPT
        if request.language:
            language = {"zh": "Chinese", "en": "English"}.get(request.language.lower(), request.language)
            prompt_text += f"Target language: {language}.\n"
        # Standalone TTS uses the Ming ``instruct`` task. No model data/ preset
        # is loaded: a voice is an IP caption, or ref_audio supplies conditioning.
        info: dict[str, Any] = {
            "ming_task": "instruct",
            "prompt": prompt_text,
            "text": request.input,
            "instruction": create_instruction(caption_fields or None),
            "voice_name": None,
            "prompt_text": request.ref_text,
            "prompt_audio_mode": audio_mode,
            "speaker_conditioning": "prompt" if has_reference else "mute",
            "use_zero_spk_emb": not has_spk_emb,
            "cfg": 2.0,
            "sigma": 0.25,
            "temperature": 0.0,
        }
        if reference_audio:
            info.update(pack_reference_audio(reference_audio))
        if request.max_new_tokens is not None:
            info["max_decode_steps"] = request.max_new_tokens
        if request.seed is not None:
            info["seed"] = request.seed
        if request.speaker_embedding is not None:
            # Keep embeddings as plain floats for EngineCore serialization.
            info["spk_emb"] = list(request.speaker_embedding)
        prompt = tokens_input(prompt_token_ids=[0])
        prompt["additional_information"] = info
        return PreparedRequest(prompt=prompt, tts_params={}, model_type="ming3_audio_tts")

    def _load_supported_speakers(self) -> list[str]:
        from vllm_omni.model_executor.models.ming_flash_omni.omni_voice import ip_voice_names

        # The shared voice store lists these names and rejects colliding uploads.
        return [name.lower() for name in ip_voice_names()]
