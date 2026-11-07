# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import json
from functools import lru_cache
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


@lru_cache(maxsize=8)
def _load_model_assets(model_id: str, revision: str | None, tokenizer_path: str, config_json: str):
    import sentencepiece

    from vllm_omni.model_executor.models.phonon.prompt import load_voice

    tokenizer = sentencepiece.SentencePieceProcessor(model_file=tokenizer_path)
    voice, project = load_voice(model_id, json.loads(config_json), revision)
    return tokenizer, voice, project


@register_tts_adapter
class PhononAdapter(ARTTSAdapter):
    name = "phonon"
    stage_keys = frozenset({"phonon"})
    model_archs = frozenset({"PhononTalkerForConditionalGeneration"})

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

    def _build(self, request: OpenAICreateSpeechRequest) -> PreparedRequest:
        from transformers.utils.hub import cached_file

        from vllm_omni.model_executor.models.phonon.prompt import prepare_text

        if self.ctx is None:
            raise RuntimeError("Phonon adapter must be bound to its serving context before building requests")
        config = self.ctx.engine_client.model_config
        revision = getattr(config, "revision", None)
        model_id = config.model
        config_dict = config.hf_config.to_dict()
        tokenizer_path = cached_file(model_id, "tokenizer.model", revision=revision)

        tokenizer, voice, project = _load_model_assets(
            model_id,
            revision,
            tokenizer_path,
            json.dumps(config_dict, sort_keys=True),
        )
        text, tail = prepare_text(request.input)
        ids = tokenizer.encode(text)
        frames = voice.shape[-1] if project else voice.shape[-2]
        prompt = {
            "prompt_token_ids": [0] * (frames + len(ids) + 1),
            "additional_information": {
                "phonon_text_ids": ids,
                "phonon_voice": voice,
                "phonon_project_voice": project,
                "phonon_tail_frames": tail,
            },
            "cache_salt": conditioning_cache_salt(request),
        }
        return PreparedRequest(
            prompt=prompt, model_type="phonon", output_policy=OutputPolicy(accumulate_nonstreaming=True)
        )
