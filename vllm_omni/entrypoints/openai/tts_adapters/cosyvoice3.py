# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CosyVoice3 TTS serving adapter."""

import os
from typing import TYPE_CHECKING, Any

import numpy as np
from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import ARTTSAdapter, PreparedRequest

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

logger = init_logger(__name__)

# Without this delimiter the talker re-speaks the reference transcript instead
# of emitting target-only speech (issue #4644). Matches the offline example and
# upstream demo.
_PROMPT_DELIMITER = "<|endofprompt|>"
_PROMPT_PREFIX = f"You are a helpful assistant.{_PROMPT_DELIMITER}"


@register_tts_adapter
class CosyVoice3Adapter(ARTTSAdapter):
    stage_keys = frozenset({"cosyvoice3_talker"})
    name = "cosyvoice3"

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        # Cache the resolved Qwen tokenizer used for dynamic-token sizing.
        # Rebuilding it would repeat snapshot resolution and add ~100 ms to the
        # TTFP-critical sampling override on every request.
        self._tokenizer = None

    async def _build_prompt(
        self, request: "OpenAICreateSpeechRequest", *, has_inline_ref_audio: bool = False
    ) -> dict[str, Any]:
        """Build the multimodal CosyVoice3 voice-cloning prompt."""
        server = self.ctx.server
        wav_samples, sr, _ = await server._resolve_ref_audio(request.ref_audio)
        ref_text = request.ref_text or ""
        if _PROMPT_DELIMITER not in ref_text:
            ref_text = f"{_PROMPT_PREFIX}{ref_text}"
        mm_kwargs: dict[str, Any] = {"prompt_text": ref_text, "sample_rate": sr}
        if request.voice:
            voice_lower = request.voice.lower()
            if voice_lower in server.uploaded_speakers and not has_inline_ref_audio:
                mm_kwargs["voice_name"] = voice_lower
                mm_kwargs["voice_created_at"] = server._voice_created_at(voice_lower)
        return {
            "prompt": request.input,
            "multi_modal_data": {"audio": (np.asarray(wav_samples, dtype=np.float32), sr)},
            "mm_processor_kwargs": mm_kwargs,
        }

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
        prompt = await self._build_prompt(request, has_inline_ref_audio=has_inline_ref_audio)
        return PreparedRequest(prompt=prompt, tts_params={}, model_type="cosyvoice3")

    def apply_sampling_overrides(
        self,
        sampling_params_list: list,
        request: "OpenAICreateSpeechRequest",
        prompt: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> list:
        """Set dynamic min/max tokens based on tokenized text length.

        The official model uses ``min_token_text_ratio`` to prevent early EOS
        and ``max_token_text_ratio`` to cap the generated token count.
        """
        import copy

        from vllm_omni.model_executor.models.cosyvoice3.tokenizer import get_qwen_tokenizer
        from vllm_omni.model_executor.models.cosyvoice3.utils import extract_text_token

        server = self.ctx.server
        sampling_params_list = copy.deepcopy(sampling_params_list)
        hf_cfg = server.model_config.hf_config
        # Build the Qwen tokenizer once per process (resolving the model dir via
        # snapshot_download at most once) and reuse it across requests.
        tokenizer = self._tokenizer
        if tokenizer is None:
            model_path = server.engine_client.model_config.model
            if not os.path.isdir(model_path):
                from huggingface_hub import snapshot_download

                model_path = snapshot_download(model_path)
            tokenizer = get_qwen_tokenizer(
                token_path=os.path.join(model_path, hf_cfg.qwen_pretrain_path),
                skip_special_tokens=hf_cfg.skip_special_tokens,
                version=hf_cfg.version,
            )
            self._tokenizer = tokenizer
        _, text_token_len = extract_text_token(
            request.input,
            tokenizer,
            hf_cfg.allowed_special,
        )
        min_ratio = getattr(hf_cfg, "min_token_text_ratio", 2)
        max_ratio = getattr(hf_cfg, "max_token_text_ratio", 20)
        sampling_params_list[0].min_tokens = max(1, int(text_token_len * min_ratio))
        sampling_params_list[0].max_tokens = min(2048, int(text_token_len * max_ratio))

        if request.max_new_tokens is not None:
            sampling_params_list[0].max_tokens = request.max_new_tokens
            sampling_params_list[0].min_tokens = min(
                getattr(sampling_params_list[0], "min_tokens", 0),
                request.max_new_tokens,
            )
        logger.info(
            "CosyVoice3 dynamic tokens: text_tokens=%d, min_tokens=%d, max_tokens=%d",
            text_token_len,
            sampling_params_list[0].min_tokens,
            sampling_params_list[0].max_tokens,
        )
        return sampling_params_list
