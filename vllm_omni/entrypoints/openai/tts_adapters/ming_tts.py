# SPDX-License-Identifier: Apache-2.0
"""Ming-TTS (dense) serving adapter."""

from typing import TYPE_CHECKING

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import ARTTSAdapter, PreparedRequest

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest


@register_tts_adapter
class MingTTSAdapter(ARTTSAdapter):
    # Detected by model_arch (MingTTSForConditionalGeneration), not stage key.
    name = "ming_tts"

    def validate(self, request: "OpenAICreateSpeechRequest") -> str | None:
        return self.ctx.server._validate_ming_tts_request(request)

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
