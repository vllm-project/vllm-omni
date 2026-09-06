# SPDX-License-Identifier: Apache-2.0
"""Voxtral TTS serving adapter."""

from typing import TYPE_CHECKING, Any

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import ARTTSAdapter, PreparedRequest, apply_max_new_tokens
from vllm_omni.entrypoints.openai.tts_adapters.capabilities import load_supported_speakers

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest


@register_tts_adapter
class VoxtralTTSAdapter(ARTTSAdapter):
    stage_keys = frozenset({"audio_generation"})
    name = "voxtral_tts"

    def validate(self, request: "OpenAICreateSpeechRequest") -> str | None:
        """Validate Voxtral TTS request parameters. Returns error message or None."""
        server = self.ctx.server
        if not request.input or not request.input.strip():
            return "Input text cannot be empty"

        # Voxtral TTS requires either a preset voice or ref_audio for voice cloning.
        if request.voice is None and request.ref_audio is None:
            return "Either 'voice' (preset speaker) or 'ref_audio' (voice cloning) must be provided"

        if request.ref_audio is not None:
            fmt_err = server._validate_ref_audio_format(request.ref_audio)
            if fmt_err:
                return fmt_err

            if not self._has_encoder_weights():
                return (
                    "Voice cloning with 'ref_audio' requires the full Voxtral checkpoint "
                    "with encoder weights. The open-source variant only supports "
                    "preset voices via the 'voice' parameter."
                )

        if request.voice is not None:
            request.voice = request.voice.lower()
            available_speakers = server._get_available_speakers()
            if available_speakers and request.voice not in available_speakers:
                return f"Invalid speaker '{request.voice}'. Supported: {', '.join(sorted(available_speakers))}"

        if request.max_new_tokens is not None:
            if request.max_new_tokens < self.max_new_tokens_min:
                return f"max_new_tokens must be at least {self.max_new_tokens_min}"
            if request.max_new_tokens > self.max_new_tokens_max:
                return f"max_new_tokens cannot exceed {self.max_new_tokens_max}"

        return None

    async def build(
        self, request: "OpenAICreateSpeechRequest", sampling_params_list: list, has_inline_ref_audio: bool
    ) -> PreparedRequest:
        prompt = await self.ctx.server._build_voxtral_prompt_async(request)
        return PreparedRequest(prompt=prompt, tts_params={}, model_type="voxtral_tts")

    def _load_supported_speakers(self) -> set[str]:
        config = self.ctx.engine_client.model_config.hf_config.audio_config
        return load_supported_speakers(self.ctx.engine_client, config)

    def apply_sampling_overrides(
        self,
        sampling_params_list: list,
        request: "OpenAICreateSpeechRequest",
        prompt: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> list:
        return apply_max_new_tokens(sampling_params_list, request)

    def _has_encoder_weights(self) -> bool:
        """Check if the loaded checkpoint contains encoder weights.

        The open-source Voxtral checkpoint omits encoder weights (input_proj,
        encoder_blocks), which are required for ref_audio voice cloning.
        """
        from vllm.transformers_utils.repo_utils import get_hf_file_to_dict

        model_name = self.ctx.engine_client.model_config.model
        config_dict = get_hf_file_to_dict("params.json", model_name)

        for key in config_dict.keys():
            if key.startswith(("input_proj.", "encoder_blocks.")):
                return True

        return False
