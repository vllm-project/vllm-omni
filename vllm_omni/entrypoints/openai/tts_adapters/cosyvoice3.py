# SPDX-License-Identifier: Apache-2.0
"""CosyVoice3 TTS serving adapter."""

from typing import TYPE_CHECKING, Any

import numpy as np

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import ARTTSAdapter, PreparedRequest

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

_COSYVOICE3_PROMPT_DELIMITER = "<|endofprompt|>"
_COSYVOICE3_PROMPT_PREFIX = f"You are a helpful assistant.{_COSYVOICE3_PROMPT_DELIMITER}"


@register_tts_adapter
class CosyVoice3Adapter(ARTTSAdapter):
    stage_keys = frozenset({"cosyvoice3_talker"})
    name = "cosyvoice3"

    def validate(self, request: "OpenAICreateSpeechRequest") -> str | None:
        """Validate CosyVoice3 request parameters. Returns error message or None."""
        server = self.ctx.server
        err = server._apply_uploaded_speaker(request)
        if err:
            return err
        if not request.input or not request.input.strip():
            return "Input text cannot be empty"

        # CosyVoice3 requires reference audio for voice cloning
        if request.ref_audio is None:
            return "CosyVoice3 requires 'ref_audio' (reference audio for voice cloning)"

        fmt_err = server._validate_ref_audio_format(request.ref_audio)
        if fmt_err:
            return fmt_err

        if not request.ref_text or not request.ref_text.strip():
            return "CosyVoice3 requires 'ref_text' (transcript of the reference audio)"

        if request.max_new_tokens is not None:
            if request.max_new_tokens < self.max_new_tokens_min:
                return f"max_new_tokens must be at least {self.max_new_tokens_min}"
            if request.max_new_tokens > self.max_new_tokens_max:
                return f"max_new_tokens cannot exceed {self.max_new_tokens_max}"

        return None

    async def build(
        self, request: "OpenAICreateSpeechRequest", sampling_params_list: list, has_inline_ref_audio: bool
    ) -> PreparedRequest:
        """Build prompt for CosyVoice3.

        CosyVoice3 uses multimodal input with reference audio for voice cloning.
        The prompt format matches the offline example: text prompt + audio data
        + mm_processor_kwargs with prompt_text.
        """
        server = self.ctx.server
        # Resolve reference audio
        wav_samples, sr = await server._resolve_ref_audio(request.ref_audio)
        audio_data = (np.asarray(wav_samples, dtype=np.float32), sr)

        # Wrap the reference transcript in the CosyVoice3 instruction template
        # so the talker emits target-only speech (see _COSYVOICE3_PROMPT_PREFIX).
        # Skip if the caller already supplied a formatted prompt_text.
        ref_text = request.ref_text or ""
        if _COSYVOICE3_PROMPT_DELIMITER not in ref_text:
            ref_text = f"{_COSYVOICE3_PROMPT_PREFIX}{ref_text}"
        mm_kwargs: dict[str, Any] = {
            "prompt_text": ref_text,
            "sample_rate": sr,
        }
        # Pass voice metadata for caching in the processor
        if request.voice:
            voice_lower = request.voice.lower()
            if voice_lower in server.uploaded_speakers and not has_inline_ref_audio:
                mm_kwargs["voice_name"] = voice_lower
                mm_kwargs["voice_created_at"] = server._voice_created_at(voice_lower)

        prompt = {
            "prompt": request.input,
            "multi_modal_data": {
                "audio": audio_data,
            },
            "mm_processor_kwargs": mm_kwargs,
        }
        # NOTE: CosyVoice3 dynamic-token sampling stays in the orchestrator tail
        # (keyed on _tts_model_type) during this incremental migration.
        return PreparedRequest(prompt=prompt, tts_params={}, model_type="cosyvoice3")
