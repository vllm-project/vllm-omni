# SPDX-License-Identifier: Apache-2.0
"""dots.tts serving adapter."""

from typing import TYPE_CHECKING

from transformers import AutoTokenizer
from vllm.utils.async_utils import make_async

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import ARTTSAdapter, PreparedRequest, apply_max_new_tokens

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest


@register_tts_adapter
class DotsTTSAdapter(ARTTSAdapter):
    """Adapter for dots.tts (AR ``engine_client`` backend).

    The current integration supports text-only, no-reference speech synthesis.
    Voice cloning and reference-audio conditioning are not supported yet.
    """

    stage_keys = frozenset()
    model_archs = frozenset({"DotsTTSForConditionalGeneration"})
    name = "dots_tts"
    detect_priority = 5

    def __init__(self, ctx):
        super().__init__(ctx)
        self.tokenizer = None
        self._build_prompt_async = None

    def _build_prompt(self, text: str) -> dict:
        from vllm_omni.model_executor.models.dots_tts.dots_tts_prompt import build_dots_tts_prompt

        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.ctx.engine_client.model_config.model,
                trust_remote_code=True,
            )
        return build_dots_tts_prompt(self.tokenizer, text)

    def validate(self, request: "OpenAICreateSpeechRequest") -> str | None:
        if not request.input or not request.input.strip():
            return "Input text cannot be empty"

        if request.voice not in (None, "default"):
            return "'voice' is not supported for dots.tts"

        if request.ref_audio is not None:
            return "'ref_audio' is not supported for dots.tts"

        if request.ref_text is not None:
            return "'ref_text' is not supported for dots.tts"

        if request.speaker_embedding is not None:
            return "'speaker_embedding' is not supported for dots.tts"

        if request.x_vector_only_mode is not None:
            return "'x_vector_only_mode' is not supported for dots.tts"
        return None

    async def build(
        self, request: "OpenAICreateSpeechRequest", sampling_params_list: list, has_inline_ref_audio: bool
    ) -> PreparedRequest:
        if self._build_prompt_async is None:
            self._build_prompt_async = make_async(
                self._build_prompt,
                executor=self.ctx.server._tts_executor,
            )
        prompt = await self._build_prompt_async(request.input)
        return PreparedRequest(prompt=prompt, tts_params={}, model_type="dots_tts")

    def apply_sampling_overrides(
        self,
        sampling_params_list: list,
        request: "OpenAICreateSpeechRequest",
        prompt: dict | None = None,
        request_id: str | None = None,
    ) -> list:
        return apply_max_new_tokens(sampling_params_list, request)
