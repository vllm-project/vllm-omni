# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CoVo-Audio serving adapter."""

from typing import TYPE_CHECKING

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import ARTTSAdapter, PreparedRequest

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest


@register_tts_adapter
class CovoAudioAdapter(ARTTSAdapter):
    stage_keys = frozenset({"fused_thinker_talker"})
    name = "covo_audio"

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self._tokenizer = None

    @classmethod
    def matches(cls, model_stage: str | None, model_arch: str | None) -> bool:
        """``fused_thinker_talker`` is a generic stage key that non-CoVo fused
        deployments also use, so the architecture has to confirm it. A fused
        stage from another model falls through to "not a TTS model"."""
        return super().matches(model_stage, model_arch) and bool(model_arch) and "CovoAudio" in model_arch

    def validate(self, request: "OpenAICreateSpeechRequest") -> str | None:
        if not request.input or not request.input.strip():
            return "Input text cannot be empty"
        return None

    async def build(
        self, request: "OpenAICreateSpeechRequest", sampling_params_list: list, has_inline_ref_audio: bool
    ) -> PreparedRequest:
        """Build the tokenized chat prompt expected by Covo-Audio-Chat.

        The model requires a specific system prompt instructing it to
        interleave text and audio tokens. Passing token IDs avoids a second
        engine-side tokenization step.
        """
        from transformers import AutoTokenizer

        from vllm_omni.model_executor.models.covo_audio.prompt_utils import build_covo_audio_prompt_token_ids

        if self._tokenizer is None:
            model_name = self.ctx.engine_client.model_config.model
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            except Exception as exc:
                raise RuntimeError(f"Failed to load Covo-Audio tokenizer from '{model_name}': {exc}") from exc
        prompt_ids = build_covo_audio_prompt_token_ids(self._tokenizer, request.input)
        return PreparedRequest(prompt={"prompt_token_ids": prompt_ids}, tts_params={}, model_type="covo_audio")
