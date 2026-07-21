# SPDX-License-Identifier: Apache-2.0
"""VoxCPM2 serving adapter (AR base-LM + diffusion side-computation)."""

from typing import TYPE_CHECKING

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import ARTTSAdapter, PreparedRequest

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest


@register_tts_adapter
class VoxCPM2Adapter(ARTTSAdapter):
    """VoxCPM2 shares ``latent_generator`` with VoxCPM; selected when no ``vae``
    stage is present (and/or via ``model_arch``)."""

    stage_keys = frozenset({"latent_generator"})
    name = "voxcpm2"

    def validate(self, request: "OpenAICreateSpeechRequest") -> str | None:
        """Validate VoxCPM2 request parameters. Returns error message or None."""
        server = self.ctx.server
        if not request.input or not request.input.strip():
            return "Input text cannot be empty"

        if request.voice is not None:
            request.voice = request.voice.lower()
            available_voices = set(server.uploaded_speakers) | set(server.precomputed_speakers) | {"default"}
            if request.voice not in available_voices:
                supported = ", ".join(sorted(available_voices)) or "none"
                return f"Invalid voice '{request.voice}'. Supported: {supported}"

        if request.max_new_tokens is not None:
            if request.max_new_tokens < self.max_new_tokens_min:
                return f"max_new_tokens must be at least {self.max_new_tokens_min}"
            if request.max_new_tokens > self.max_new_tokens_max:
                return f"max_new_tokens cannot exceed {self.max_new_tokens_max}"

        return None

    async def build(
        self, request: "OpenAICreateSpeechRequest", sampling_params_list: list, has_inline_ref_audio: bool
    ) -> PreparedRequest:
        server = self.ctx.server
        # VoxCPM2 needs the raw waveform tuple for prefill-length accounting, so
        # it loads uploaded audio directly rather than via _apply_uploaded_speaker.
        uploaded_ref = None
        if request.voice:
            voice_lower = request.voice.lower()
            if voice_lower in server.uploaded_speakers and not has_inline_ref_audio:
                if server.uploaded_speakers[voice_lower].get("embedding_source") == "direct":
                    raise ValueError(
                        f"Uploaded voice '{request.voice}' uses a speaker embedding (Qwen3-only). "
                        f"Re-upload with an audio file for VoxCPM2."
                    )
                if request.ref_audio is None:
                    uploaded_ref = server._load_uploaded_audio(voice_lower)
        prompt = await server._build_voxcpm2_prompt(request, uploaded_ref=uploaded_ref)
        tts_params = {}
        if request.voice:
            voice_lower = request.voice.lower()
            if voice_lower in server.uploaded_speakers or voice_lower in server.precomputed_speakers:
                additional = prompt.setdefault("additional_information", {})
                additional["voice_name"] = voice_lower
                additional["voice_created_at"] = server._voice_created_at(voice_lower)
        return PreparedRequest(prompt=prompt, tts_params=tts_params, model_type="voxcpm2")
