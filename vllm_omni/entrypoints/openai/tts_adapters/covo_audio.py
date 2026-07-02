# SPDX-License-Identifier: Apache-2.0
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

    def __init__(self, ctx):
        super().__init__(ctx)
        self._covo_audio_tokenizer = None

    async def build(
        self, request: "OpenAICreateSpeechRequest", sampling_params_list: list, has_inline_ref_audio: bool
    ) -> PreparedRequest:
        """Build a chat-style prompt for Covo-Audio-Chat.

        Covo-Audio requires a specific system prompt that instructs the model
        to interleave text and audio tokens in its output.  We render the
        messages through the chat template and pass prompt_token_ids so that
        the engine does not need to re-tokenize.
        """
        server = self.ctx.server
        from transformers import AutoTokenizer

        from vllm_omni.model_executor.models.covo_audio.prompt_utils import (
            build_covo_audio_prompt_token_ids,
        )

        if self._covo_audio_tokenizer is None:
            model_name = server.engine_client.model_config.model
            try:
                self._covo_audio_tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                )
            except Exception as exc:
                raise RuntimeError(f"Failed to load Covo-Audio tokenizer from '{model_name}': {exc}") from exc

        prompt_ids = build_covo_audio_prompt_token_ids(
            self._covo_audio_tokenizer,
            request.input,
        )
        prompt = {"prompt_token_ids": prompt_ids}
        return PreparedRequest(prompt=prompt, tts_params={}, model_type="covo_audio")
