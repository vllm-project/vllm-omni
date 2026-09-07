# SPDX-License-Identifier: Apache-2.0
"""GLM-TTS serving adapter."""

from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import ARTTSAdapter, PreparedRequest

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

logger = init_logger(__name__)


@register_tts_adapter
class GlmTTSAdapter(ARTTSAdapter):
    stage_keys = frozenset({"glm_tts"})
    name = "glm_tts"

    def validate(self, request: "OpenAICreateSpeechRequest") -> str | None:
        """Validate GLM-TTS request — requires ref_audio for voice cloning."""
        server = self.ctx.server
        err = server._apply_uploaded_speaker(request)
        if err:
            return err
        if not request.input or not request.input.strip():
            return "Input text cannot be empty"

        if request.ref_audio is None:
            return "GLM-TTS requires 'ref_audio' for zero-shot voice cloning"
        fmt_err = server._validate_ref_audio_format(request.ref_audio)
        if fmt_err:
            return fmt_err
        if not request.ref_text or not request.ref_text.strip():
            return "GLM-TTS voice cloning requires 'ref_text' (transcript of the reference audio)"

        if request.max_new_tokens is not None:
            if request.max_new_tokens < self.max_new_tokens_min:
                return f"max_new_tokens must be >= {self.max_new_tokens_min}"
            if request.max_new_tokens > self.max_new_tokens_max:
                return f"max_new_tokens cannot exceed {self.max_new_tokens_max}"
        return None

    async def build(
        self, request: "OpenAICreateSpeechRequest", sampling_params_list: list, has_inline_ref_audio: bool
    ) -> PreparedRequest:
        prompt = await self.ctx.server._build_glm_tts_prompt(request, has_inline_ref_audio=has_inline_ref_audio)
        return PreparedRequest(prompt=prompt, tts_params={}, model_type="glm_tts")

    def _load_supported_speakers(self) -> set[str]:
        return set()

    def apply_sampling_overrides(
        self,
        sampling_params_list: list,
        request: "OpenAICreateSpeechRequest",
        prompt: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> list:
        # GLM-TTS: set dynamic min/max tokens based on text length.
        import copy

        server = self.ctx.server
        sampling_params_list = copy.deepcopy(sampling_params_list)
        glm_metadata = prompt.get("additional_information") if isinstance(prompt, dict) else None
        text_len_value = None
        if isinstance(glm_metadata, dict):
            text_len_value = glm_metadata.get("glm_tts_text_token_len")
            if isinstance(text_len_value, list) and text_len_value:
                text_len_value = text_len_value[0]
        text_token_len = (
            int(text_len_value)
            if text_len_value is not None
            else server._estimate_glm_tts_text_token_len(request.input)
        )
        hf_cfg = server.model_config.hf_config
        min_ratio = getattr(hf_cfg, "min_token_text_ratio", 2)
        max_ratio = getattr(hf_cfg, "max_token_text_ratio", 20)
        stage_min_tokens = getattr(sampling_params_list[0], "min_tokens", None)
        stage_max_tokens = getattr(sampling_params_list[0], "max_tokens", None)
        cap_candidates = [int(cap) for cap in (stage_max_tokens, request.max_new_tokens) if cap is not None]
        hard_cap = min(cap_candidates) if cap_candidates else None

        min_tokens = max(1, int(text_token_len * min_ratio))
        if stage_min_tokens is not None:
            min_tokens = max(min_tokens, int(stage_min_tokens))
        if hard_cap is not None:
            min_tokens = min(min_tokens, hard_cap)

        max_tokens = max(min_tokens, int(text_token_len * max_ratio))
        if hard_cap is not None:
            max_tokens = min(max_tokens, hard_cap)
        sampling_params_list[0].min_tokens = min_tokens
        sampling_params_list[0].max_tokens = max_tokens
        seed = getattr(request, "seed", None)
        if seed is not None:
            sampling_params_list[0].seed = seed
        logger.info(
            "GLM-TTS dynamic tokens: text_tokens=%d, min_ratio=%s, max_ratio=%s, "
            "stage_min=%s, stage_max=%s, request_max=%s, min_tokens=%d, max_tokens=%d",
            text_token_len,
            min_ratio,
            max_ratio,
            stage_min_tokens,
            stage_max_tokens,
            request.max_new_tokens,
            min_tokens,
            max_tokens,
        )
        return sampling_params_list
