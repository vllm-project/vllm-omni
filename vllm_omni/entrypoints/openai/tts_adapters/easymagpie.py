# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import ARTTSAdapter, PreparedRequest, apply_max_new_tokens
from vllm_omni.model_executor.models.easymagpie.config import EasyMagpieOmniArch
from vllm_omni.model_executor.models.easymagpie.tokenizer import EasyMagpieTextTokenizer

MODEL_TYPE = "easymagpie"
_TALKER_STAGE = "easymagpie"
_DEFAULT_SPEAKER = "eng"
_DEFAULT_CONTEXT_TEXT = "[EN]"
_DEFAULT_TEMPERATURE = 0.7
_DEFAULT_TOP_K = 80
_TEXT_EOS_OFFSET_FROM_VOCAB = 2


@register_tts_adapter
class EasyMagpieTTSAdapter(ARTTSAdapter):
    """Build speaker-conditioned prompts for ``/v1/audio/speech`` requests."""

    name = MODEL_TYPE
    stage_keys = frozenset({_TALKER_STAGE})

    def __init__(self, ctx: Any) -> None:
        super().__init__(ctx)
        self._tokenizer: Any = None
        self._model_path_cache: str | None = None
        self._prompt_len_cache: dict[tuple[str, str], int] = {}
        self._model_config_cache: dict[str, Any] | None = None
        self._arch_cache: EasyMagpieOmniArch | None = None

    def _model_path(self) -> str:
        if self._model_path_cache is not None:
            return self._model_path_cache
        engine_client = getattr(self.ctx, "engine_client", None)
        model_config = getattr(engine_client, "model_config", None)
        path = getattr(model_config, "model", None)
        if not path:
            for stage in getattr(engine_client, "stage_configs", None) or []:
                stage_path = getattr(getattr(stage, "engine_args", None), "model", None)
                if stage_path:
                    path = stage_path
                    break
        if not path:
            raise RuntimeError("EasyMagpie serving adapter could not resolve the model path.")
        self._model_path_cache = path
        return path

    def _tokenize(self) -> Callable[[str], Any]:
        if self._tokenizer is None:
            self._tokenizer = EasyMagpieTextTokenizer.from_pretrained(self._model_path())
        return self._tokenizer.encode_context

    def _model_tokenizer(self):
        self._tokenize()
        return self._tokenizer

    def _prompt_len(self, speaker_id: str, context_text: str = _DEFAULT_CONTEXT_TEXT) -> int:
        key = (speaker_id, context_text)
        cached = self._prompt_len_cache.get(key)
        if cached is not None:
            return cached
        from vllm_omni.model_executor.models.easymagpie.easymagpie import EasyMagpieTTSForConditionalGeneration

        plen = int(
            EasyMagpieTTSForConditionalGeneration.get_prompt_len(
                speaker_id, self._model_path(), tokenize=self._tokenize(), context_text=context_text
            )
        )
        self._prompt_len_cache[key] = plen
        return plen

    def _model_config(self) -> dict[str, Any]:
        if self._model_config_cache is None:
            path = Path(self._model_path()) / "config.json"
            self._model_config_cache = json.loads(path.read_text())
        return self._model_config_cache

    def _arch(self) -> EasyMagpieOmniArch:
        if self._arch_cache is None:
            self._arch_cache = EasyMagpieOmniArch.from_hf_config(SimpleNamespace(**self._model_config()))
        return self._arch_cache

    def _text_stream_metadata(self) -> tuple[int, int]:
        config = self._model_config()
        text_vocab_size = int(config.get("text_vocab_size", config.get("vocab_size", 0)))
        if text_vocab_size <= _TEXT_EOS_OFFSET_FROM_VOCAB:
            raise ValueError("EasyMagpie config must define text_vocab_size")
        configured_text_eos_id = config.get("text_eos_id")
        text_eos_id = (
            text_vocab_size - _TEXT_EOS_OFFSET_FROM_VOCAB
            if configured_text_eos_id is None
            else int(configured_text_eos_id)
        )
        return (text_eos_id, self._arch().text_prefill_num)

    def validate(self, request: OpenAICreateSpeechRequest) -> str | None:
        if error := self._validate_max_new_tokens(request):
            return error
        if not request.input or not request.input.strip():
            return "Input text cannot be empty"
        extra = request.extra_params
        if extra is not None and (not isinstance(extra, dict)):
            return "extra_params must be a JSON object/dict"
        context_text = (extra or {}).get("context_text")
        if context_text is not None and (not isinstance(context_text, str)):
            return "context_text must be a string or null"
        ref_audio = request.ref_audio
        if ref_audio is None:
            speaker_id = (request.voice or _DEFAULT_SPEAKER).strip()
            if not speaker_id or Path(speaker_id).name != speaker_id or speaker_id in {".", ".."}:
                return "voice must be a bundled speaker name"
            return None
        if request.voice and request.voice.strip():
            return "'voice' and 'ref_audio' are mutually exclusive; provide a known voice or raw reference audio"
        if not isinstance(ref_audio, str):
            return "EasyMagpie supports a single reference in 'ref_audio', not a list"
        format_error = self.ctx.server._validate_ref_audio_format(ref_audio)
        if format_error:
            return format_error
        try:
            self._arch().ensure_reference_audio_available()
        except RuntimeError as error:
            return str(error)
        return None

    def apply_sampling_overrides(
        self,
        sampling_params_list: list,
        request: OpenAICreateSpeechRequest,
        prompt: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> list:
        return apply_max_new_tokens(sampling_params_list, request)

    async def build(
        self, request: OpenAICreateSpeechRequest, sampling_params_list: list, has_inline_ref_audio: bool
    ) -> PreparedRequest:
        del sampling_params_list, has_inline_ref_audio
        extra = request.extra_params or {}
        context_text = extra.get("context_text") or _DEFAULT_CONTEXT_TEXT
        (text_eos_id, text_prefill_num) = self._text_stream_metadata()
        text_tokens = self._model_tokenizer().encode(request.input, add_special_tokens=False)
        text_tokens.append(text_eos_id)
        prompt: dict[str, Any] = {
            "additional_information": {
                "context_text": context_text,
                "text_tokens": text_tokens,
                "prefill_text_tokens": text_tokens[:text_prefill_num],
                "text_prefill_num": text_prefill_num,
                "temperature": float(extra.get("temperature", _DEFAULT_TEMPERATURE)),
                "top_k": int(extra.get("top_k", _DEFAULT_TOP_K)),
            }
        }
        ref_audio = request.ref_audio
        if ref_audio is None:
            speaker_id = (request.voice or _DEFAULT_SPEAKER).strip()
            prompt["prompt_token_ids"] = [0] * (self._prompt_len(speaker_id, context_text) + text_prefill_num)
            prompt["additional_information"]["speaker_id"] = speaker_id
            return PreparedRequest(prompt=prompt, tts_params={}, model_type=MODEL_TYPE)
        arch = self._arch()
        arch.ensure_reference_audio_available()
        (waveform, sample_rate) = await self.ctx.server._resolve_ref_audio(ref_audio)
        if sample_rate != arch.codec_input_sample_rate:
            raise ValueError(
                f"EasyMagpie reference audio must be {arch.codec_input_sample_rate} Hz; received {sample_rate} Hz"
            )
        context_text = prompt["additional_information"]["context_text"]
        context_len = len(self._tokenize()(context_text))
        task_len = int(arch.num_task_embeddings > 0)
        prompt["prompt_token_ids"] = (
            [0] * task_len + [arch.audio_input_token_id] + [0] * context_len + [0] * text_prefill_num
        )
        prompt["multi_modal_data"] = {"audio": [(waveform, sample_rate)]}
        return PreparedRequest(prompt=prompt, tts_params={}, model_type=MODEL_TYPE)
