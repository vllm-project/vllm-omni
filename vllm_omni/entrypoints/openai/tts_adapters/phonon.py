# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import (
    ARTTSAdapter,
    OutputPolicy,
    PreparedRequest,
    conditioning_cache_salt,
)

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest


@register_tts_adapter
class PhononAdapter(ARTTSAdapter):
    name = "phonon"
    stage_keys = frozenset({"phonon"})
    model_archs = frozenset({"PhononTalkerForConditionalGeneration"})

    def __init__(self, ctx):
        super().__init__(ctx)
        self.tokenizer, self.voice_frames = self._load_assets()

    def _load_supported_speakers(self) -> list[str]:
        return ["default"]

    def _load_supported_languages(self) -> frozenset[str]:
        return frozenset({"English", "en", "Auto"})

    def validate(self, request: OpenAICreateSpeechRequest) -> str | None:
        if not request.input or not request.input.strip():
            return "Input text cannot be empty"
        if request.voice not in (None, "default"):
            return "Phonon initial release supports the default pretrained voice"
        if request.ref_audio is not None or request.speaker_embedding is not None:
            return "Phonon initial release supports precomputed speaker conditioning"
        return None

    async def build(
        self,
        request: OpenAICreateSpeechRequest,
        sampling_params_list: list,
        has_inline_ref_audio: bool,
    ) -> PreparedRequest:
        import asyncio

        return await asyncio.to_thread(self._build, request)

    def _load_assets(self):
        import sentencepiece
        from safetensors import safe_open
        from transformers.utils.hub import cached_file

        if self.ctx is None:
            raise RuntimeError("Phonon adapter must be bound before building requests")
        config = self.ctx.engine_client.model_config
        revision = getattr(config, "revision", None)
        tokenizer = sentencepiece.SentencePieceProcessor(
            model_file=cached_file(config.model, "tokenizer.model", revision=revision)
        )
        from vllm_omni.model_executor.models.phonon.prompt import voice_layout

        voice_path = cached_file(config.model, "default-voice.safetensors", revision=revision)
        with safe_open(voice_path, framework="pt", device="cpu") as voice:
            _, frames, _ = voice_layout(voice)
        return tokenizer, frames

    def _build(self, request: OpenAICreateSpeechRequest) -> PreparedRequest:
        from vllm_omni.model_executor.models.phonon.prompt import prepare_text

        text, tail = prepare_text(request.input)
        ids = self.tokenizer.encode(text)
        prompt = {
            "prompt_token_ids": [0] * (self.voice_frames + len(ids) + 1),
            "additional_information": {
                "phonon_text_ids": ids,
                "phonon_tail_frames": tail,
            },
            "cache_salt": conditioning_cache_salt(request),
        }
        return PreparedRequest(
            prompt=prompt, model_type="phonon", output_policy=OutputPolicy(accumulate_nonstreaming=True)
        )
