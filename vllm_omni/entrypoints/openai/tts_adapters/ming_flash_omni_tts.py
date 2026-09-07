# SPDX-License-Identifier: Apache-2.0
"""MingFlashOmniTTS serving adapter."""

from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import ARTTSAdapter, PreparedRequest

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

logger = init_logger(__name__)


@register_tts_adapter
class MingFlashOmniTTSAdapter(ARTTSAdapter):
    # Ming-flash-omni drives speaker selection via the caption JSON
    # (audio_sequence[0]["说话人"]) rather than a spk_id table, so there
    # is no static speaker list to surface here.
    stage_keys = frozenset({"ming_tts"})
    name = "ming_flash_omni_tts"

    def validate(self, request: "OpenAICreateSpeechRequest") -> str | None:
        """Validate Ming-flash-omni standalone-talker request parameters."""
        server = self.ctx.server
        if not request.input or not request.input.strip():
            return "Input text cannot be empty"
        if request.instructions is not None:
            if not isinstance(request.instructions, str):
                return "instructions must be a string"
            if len(request.instructions) > server._max_instructions_length:
                return f"instructions exceeds max length {server._max_instructions_length}"

        if request.task_type is not None:
            return "'task_type' is not supported for Ming-flash-omni TTS"
        if request.language is not None:
            return "'language' is not supported for Ming-flash-omni TTS (language is inferred from input text)"
        if request.x_vector_only_mode is not None:
            return "'x_vector_only_mode' is not supported for Ming-flash-omni TTS"
        if request.initial_codec_chunk_frames is not None:
            return "'initial_codec_chunk_frames' is not supported for Ming-flash-omni TTS"

        # Per-request voice cloning from raw audio is not yet wired up: Ming
        # extracts spk_emb / prompt_wav_lat / prompt_wav_emb model-side via
        # register_prompt_wav() at engine init. For ad-hoc cloning, callers
        # should pre-compute speaker_embedding and pass it directly.
        if request.ref_audio is not None:
            return (
                "'ref_audio' is not yet supported for Ming-flash-omni TTS; "
                "use a preset 'voice' or 'speaker_embedding' instead"
            )
        if request.ref_text is not None:
            return "'ref_text' is not yet supported for Ming-flash-omni TTS"

        if request.max_new_tokens is not None and request.max_new_tokens <= 0:
            return "'max_new_tokens' must be a positive integer"
        return None

    async def build(
        self, request: "OpenAICreateSpeechRequest", sampling_params_list: list, has_inline_ref_audio: bool
    ) -> PreparedRequest:
        server = self.ctx.server
        prompt = server._build_ming_flash_omni_prompt(request)
        return PreparedRequest(prompt=prompt, tts_params={}, model_type="ming_flash_omni_tts")

    def apply_sampling_overrides(
        self,
        sampling_params_list: list,
        request: "OpenAICreateSpeechRequest",
        prompt: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> list:
        """Size Stage-0 for the native-paged talker.

        The paged Qwen2 backbone runs one AR step per audio frame, so
        ``max_tokens`` must cover the full decode budget (mirrored into
        ``max_decode_steps`` on the prompt) and generation stops on the talker
        EOS (token id 1).
        """
        import copy

        from vllm_omni.entrypoints.openai.serving_speech import _TTS_MAX_NEW_TOKENS_MAX

        if not sampling_params_list:
            return sampling_params_list

        sampling_params_list = copy.deepcopy(sampling_params_list)
        max_tokens = request.max_new_tokens or getattr(sampling_params_list[0], "max_tokens", None)
        max_tokens = int(max_tokens or _TTS_MAX_NEW_TOKENS_MAX)
        sampling_params_list[0].max_tokens = max_tokens
        if isinstance(prompt, dict):
            prompt.setdefault("additional_information", {})["max_decode_steps"] = max_tokens
        stop_token_ids = set(getattr(sampling_params_list[0], "stop_token_ids", None) or [])
        stop_token_ids.add(1)
        sampling_params_list[0].stop_token_ids = sorted(stop_token_ids)
        return sampling_params_list

    def _load_supported_speakers(self) -> set[str]:
        # Speaker selection is driven by caption JSON rather than a static table.
        return set()
