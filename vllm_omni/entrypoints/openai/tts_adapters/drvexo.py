# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Caller-text entry for the shared DRVEXO Talker, including reference voices."""

from threading import Lock

from vllm.utils.async_utils import make_async

from vllm_omni.entrypoints.openai.drvexo_speech import load_speech_tokenizer
from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import PreparedRequest, SpeechServingContext
from vllm_omni.entrypoints.openai.tts_adapters.qwen3_tts import Qwen3TTSAdapter
from vllm_omni.model_executor.models.drvexo_speech_8b.lexical import fixed_text_ids


@register_tts_adapter
class DRVEXOTTSAdapter(Qwen3TTSAdapter):
    name = "drvexo"
    stage_keys = frozenset()
    model_archs = frozenset({"DRVEXOSpeech8BTalker", "DRVEXOUnifiedTalker"})
    detect_priority = 90
    entry_stage_id = 1

    def __init__(self, ctx: SpeechServingContext) -> None:
        super().__init__(ctx)
        self._tokenizer_load_lock = Lock()
        self._validate_text_async = make_async(self._validate_text, executor=getattr(ctx.server, "_tts_executor", None))

    def _load_text_tokenizer(self) -> None:
        server = self.ctx.server
        if server._tts_tokenizer is None:
            # Concurrent cold requests share one load. Only worker threads use
            # this lock; request handlers never wait for disk on the event loop.
            with self._tokenizer_load_lock:
                if server._tts_tokenizer is None:
                    assert self.ctx.engine_client is not None
                    # The root tokenizer belongs to Thinker. Match the native
                    # Talker for admission and placeholder length/IDs.
                    server._tts_tokenizer = load_speech_tokenizer(self.ctx.engine_client.model_config.model)

    def _validate_text(self, text: str) -> None:
        self._load_text_tokenizer()
        fixed_text_ids(text, self.ctx.server._tts_tokenizer)

    async def build(
        self, request: OpenAICreateSpeechRequest, sampling_params_list: list, has_inline_ref_audio: bool
    ) -> PreparedRequest:
        # Parameter validation remains synchronous and cheap. Both the fallback
        # disk load and lexical validation run off-loop, before Qwen's builder
        # can resolve reference audio or submit anything to the engine.
        await self._validate_text_async(request.input)
        return await super().build(request, sampling_params_list, has_inline_ref_audio)

    def apply_sampling_overrides(self, sampling_params_list, request, prompt=None, request_id=None):
        # Serving passes all stages indexed from zero, not an entry-stage slice.
        # Both complete DRVEXO pipelines have Talker followed by Codec.
        if len(sampling_params_list) == self.entry_stage_id:
            raise ValueError(
                "Speech output is unavailable in this DRVEXO Thinker-only deployment; "
                "use a deployment with Talker and Codec"
            )
        if len(sampling_params_list) != self.entry_stage_id + 2:
            raise ValueError("DRVEXO speech requires the full stage-0-indexed sampling list through Talker and Codec")
        # Qwen3-TTS owns the Talker budget. Keep Thinker's params independent.
        return sampling_params_list[: self.entry_stage_id] + super().apply_sampling_overrides(
            sampling_params_list[self.entry_stage_id :], request, prompt, request_id
        )

    def _build_tts_params(self, request):
        params = super()._build_tts_params(request)
        # This entry starts at Talker with explicit caller text. It cannot
        # claim the Thinker-generated token provenance required by chat edges.
        params["drvexo_fixed_reply"] = True
        return params
