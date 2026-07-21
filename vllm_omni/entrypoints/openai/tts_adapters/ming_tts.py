# SPDX-License-Identifier: Apache-2.0
"""Ming-TTS (dense) serving adapter."""

from typing import TYPE_CHECKING

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import ARTTSAdapter, PreparedRequest

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

from vllm.logger import init_logger

logger = init_logger(__name__)


@register_tts_adapter
class MingTTSAdapter(ARTTSAdapter):
    # Detected by model_arch (MingTTSForConditionalGeneration), not stage key.
    name = "ming_tts"

    def validate(self, request: "OpenAICreateSpeechRequest") -> str | None:
        """Validate Ming TTS request parameters. Returns error message or None."""
        if not request.input or not request.input.strip():
            return "Input text cannot be empty"

        if isinstance(request.ref_audio, list):
            return self._validate_ming_tts_podcast_request(request)
        return self._validate_ming_tts_single_speaker_request(request)

    async def build(
        self, request: "OpenAICreateSpeechRequest", sampling_params_list: list, has_inline_ref_audio: bool
    ) -> PreparedRequest:
        server = self.ctx.server
        ref_audio_source = request.ref_audio
        voice_lower = request.voice.lower() if isinstance(request.voice, str) else None
        if ref_audio_source is None and voice_lower in server.uploaded_speakers:
            speaker_info = server.uploaded_speakers[voice_lower]
            if speaker_info.get("embedding_source") == "direct":
                if request.speaker_embedding is None:
                    request.speaker_embedding = server._get_uploaded_speaker_embedding(request.voice)
                if request.speaker_embedding is None:
                    raise ValueError(f"Speaker embedding for uploaded voice '{request.voice}' is missing")
            else:
                ref_audio_source = server._get_uploaded_audio_data(request.voice)
                if not ref_audio_source:
                    raise ValueError(f"Audio file for uploaded voice '{request.voice}' is missing")
                if request.ref_text is None:
                    request.ref_text = speaker_info.get("ref_text")
        ref_audio_data = None
        if isinstance(ref_audio_source, list):
            ref_audio_data = await server._resolve_ref_audio_many(ref_audio_source)
            if request.speaker_embedding is None:
                request.speaker_embedding = server._extract_ming_speaker_embeddings_from_ref_audio(ref_audio_data)
        elif ref_audio_source is not None and isinstance(ref_audio_source, str):
            wav_list, sr = await server._resolve_ref_audio(ref_audio_source)
            ref_audio_data = (wav_list, sr)
            if request.speaker_embedding is None:
                request.speaker_embedding = server._extract_ming_speaker_embeddings_from_ref_audio([ref_audio_data])[0]
        prompt = server._build_ming_dense_prompt(request, ref_audio_data=ref_audio_data)
        tts_params = prompt.get("additional_information", {})
        # Ming stop-token / max_tokens sampling stays in the orchestrator tail.
        return PreparedRequest(prompt=prompt, tts_params=tts_params, model_type="ming_tts")

    def _validate_ming_tts_single_speaker_request(self, request: "OpenAICreateSpeechRequest") -> str | None:
        server = self.ctx.server
        if request.ref_audio is not None:
            fmt_err = server._validate_ref_audio_format(request.ref_audio)
            if fmt_err:
                return fmt_err

        if request.speaker_embedding is not None:
            if not request.speaker_embedding:
                return "'speaker_embedding' must be a non-empty list of floats"
            emb_len = len(request.speaker_embedding)
            if emb_len != 192:
                logger.warning(
                    "speaker_embedding has %d dimensions; Ming dense expects 192. "
                    "Wrong dimensions will likely fail or degrade output.",
                    emb_len,
                )

        voice_lower = request.voice.lower() if isinstance(request.voice, str) else None
        uploaded_voice = bool(voice_lower and voice_lower in server.uploaded_speakers)
        clone_source_present = request.ref_audio is not None or request.speaker_embedding is not None or uploaded_voice

        if request.task_type == "Base" and not clone_source_present:
            return "Base task requires 'ref_audio', 'speaker_embedding', or an uploaded voice sample"

        if request.ref_audio is not None and request.ref_text is not None and not request.ref_text.strip():
            return "'ref_text' must be non-empty when provided with 'ref_audio'"

        if request.ref_text is not None and request.ref_audio is None and not uploaded_voice:
            return "'ref_text' requires 'ref_audio' or an uploaded voice sample"

        if request.instructions and len(request.instructions) > server._max_instructions_length:
            return f"Instructions too long (max {server._max_instructions_length} characters)"

        if request.max_new_tokens is not None:
            if request.max_new_tokens < self.max_new_tokens_min:
                return f"max_new_tokens must be at least {self.max_new_tokens_min}"
            if request.max_new_tokens > self.max_new_tokens_max:
                return f"max_new_tokens cannot exceed {self.max_new_tokens_max}"

        return None

    def _validate_ming_tts_podcast_request(self, request: "OpenAICreateSpeechRequest") -> str | None:
        server = self.ctx.server
        if len(request.ref_audio) < 2:
            return "Podcast-style Ming requests require at least two 'ref_audio' clips"

        for ref_audio in request.ref_audio:
            fmt_err = server._validate_ref_audio_format(ref_audio)
            if fmt_err:
                return fmt_err

        if not request.ref_text or not request.ref_text.strip():
            return "Podcast-style Ming requests require non-empty 'ref_text'"

        if request.speaker_embedding is not None:
            embeddings = request.speaker_embedding
            embedding_count = len(embeddings) if embeddings and isinstance(embeddings[0], list) else 1
            if embedding_count != len(request.ref_audio):
                return (
                    "Podcast-style Ming requests require one speaker embedding per ref_audio clip; "
                    f"got {embedding_count} embeddings for {len(request.ref_audio)} clips"
                )
            if embeddings and isinstance(embeddings[0], list):
                for item in embeddings:
                    if len(item) != 192:
                        return "Podcast-style Ming speaker embeddings must each have 192 dimensions"

        if request.instructions and len(request.instructions) > server._max_instructions_length:
            return f"Instructions too long (max {server._max_instructions_length} characters)"

        if request.max_new_tokens is not None:
            if request.max_new_tokens < self.max_new_tokens_min:
                return f"max_new_tokens must be at least {self.max_new_tokens_min}"
            if request.max_new_tokens > self.max_new_tokens_max:
                return f"max_new_tokens cannot exceed {self.max_new_tokens_max}"

        return None
