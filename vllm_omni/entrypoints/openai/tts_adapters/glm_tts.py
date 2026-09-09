# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""GLM-TTS serving adapter."""

from typing import TYPE_CHECKING, Any

import numpy as np
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

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self._text_tokenizer = None
        self._text_frontend = None

    def _text_tokenizer_and_frontend(self):
        from vllm_omni.model_executor.models.glm_tts.glm_tts import (
            load_glm_tts_tokenizer,
            resolve_glm_tts_tokenizer_path,
        )
        from vllm_omni.model_executor.models.glm_tts.text_frontend import GLMTTSTextFrontend

        if self._text_tokenizer is None:
            config = self.ctx.engine_client.model_config
            tokenizer_path = getattr(config, "tokenizer", None) or resolve_glm_tts_tokenizer_path(config.model)
            self._text_tokenizer = load_glm_tts_tokenizer(
                tokenizer_path,
                model_name_or_path=config.model,
                trust_remote_code=bool(getattr(config, "trust_remote_code", False)),
            )
        if self._text_frontend is None:
            self._text_frontend = GLMTTSTextFrontend()
        return self._text_tokenizer, self._text_frontend

    def _estimate_text_token_len(self, text: str | None, *, add_trailing_space: bool = False) -> int:
        tokenizer, frontend = self._text_tokenizer_and_frontend()
        normalized = (frontend.text_normalize(text or "") or text or "").strip()
        if add_trailing_space and normalized:
            normalized = f"{normalized} "
        return max(1, len(tokenizer.encode(normalized)))

    def _build_prefill_metadata(self, text: str, prompt_text: str | None) -> dict[str, Any]:
        text_len = self._estimate_text_token_len(text)
        prompt_len = self._estimate_text_token_len(prompt_text, add_trailing_space=True) if prompt_text else 0
        return {
            "glm_tts_text_token_len": [text_len],
            "glm_tts_prompt_text_token_len": [prompt_len],
            "input_len": [prompt_len + text_len + 1],
        }

    async def _build_prompt(
        self, request: "OpenAICreateSpeechRequest", *, has_inline_ref_audio: bool = False
    ) -> dict[str, Any]:
        """Build GLM-TTS multimodal voice-cloning input and prefill metadata.

        AR preprocess constructs PromptText, Text, BOA and prompt speech tokens;
        the DiT consumes the resulting prompt tokens, features and embedding as
        conditioning.
        """
        server = self.ctx.server
        if request.ref_audio is None or not request.ref_text:
            raise ValueError("GLM-TTS requires ref_audio and ref_text for voice cloning.")
        wav_samples, sr, _ = await server._resolve_ref_audio(request.ref_audio)
        mm_kwargs: dict[str, Any] = {"prompt_text": request.ref_text}
        if request.voice:
            voice_lower = request.voice.lower()
            if voice_lower in server.uploaded_speakers and not has_inline_ref_audio:
                mm_kwargs["voice_name"] = voice_lower
                mm_kwargs["voice_created_at"] = server._voice_created_at(voice_lower)
        return {
            "prompt": request.input,
            "multi_modal_data": {"audio": (np.asarray(wav_samples, dtype=np.float32), int(sr))},
            "mm_processor_kwargs": mm_kwargs,
            "additional_information": self._build_prefill_metadata(request.input, request.ref_text),
        }

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
        prompt = await self._build_prompt(request, has_inline_ref_audio=has_inline_ref_audio)
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
            int(text_len_value) if text_len_value is not None else self._estimate_text_token_len(request.input)
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
