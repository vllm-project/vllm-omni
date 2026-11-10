# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""OpenAI speech adapter for inclusionAI Ming-3.0-TTS."""

from __future__ import annotations

import asyncio
import copy
import json
import math
import threading
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import regex as re
import torch
from transformers import AutoTokenizer
from vllm.inputs import tokens_input
from vllm.logger import init_logger
from vllm.platforms import current_platform

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import ARTTSAdapter, PreparedRequest
from vllm_omni.model_executor.models.ming3_tts.constants import (
    AUDIO_END_TOKEN_ID,
    KEY_CFG,
    KEY_MAX_DECODE_STEPS,
    KEY_REQUEST_ID,
    KEY_TEMPERATURE,
    MAX_DECODE_STEPS,
)
from vllm_omni.model_executor.models.ming3_tts.prompt_assembly import (
    build_ming3_prompt,
    create_ming3_instruction,
    normalize_target_language,
    resolve_podcast_decode_window,
)
from vllm_omni.model_executor.models.ming_tts.audio_prep import coerce_speaker_embeddings

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

logger = init_logger(__name__)

# Without CUDA, extract CAM++ speaker embeddings in the API process (on a
# worker thread) instead of inside the Stage-0 engine step, where the CPU
# inference stalls every active stream. On CUDA, Stage 0 embeds the reference
# clips of a step's new requests together on the GPU (batched_campplus).
_SPEAKER_CACHE_SIZE = 256
_speaker_lock = threading.Lock()
_speaker_slots = threading.BoundedSemaphore(4)
_speaker_extractors: dict[str, Any] = {}
_speaker_embeddings: OrderedDict[tuple[str, str], torch.Tensor] = OrderedDict()


def _extract_speaker_embeddings(
    model: str,
    waveforms: list[torch.Tensor],
    sample_rates: list[int],
    audio_keys: list[str | None],
) -> torch.Tensor:
    from vllm_omni.model_executor.models.common.ming.speaker_extractor import SpeakerEmbeddingExtractor

    with _speaker_lock:
        extractor = _speaker_extractors.get(model)
        if extractor is None:
            extractor = SpeakerEmbeddingExtractor(model, target_sr=16000)
            _speaker_extractors[model] = extractor
    rows = []
    for waveform, sample_rate, audio_key in zip(waveforms, sample_rates, audio_keys, strict=True):
        cache_key = (model, audio_key) if audio_key else None
        with _speaker_lock:
            embedding = _speaker_embeddings.get(cache_key) if cache_key is not None else None
            if embedding is not None and cache_key is not None:
                _speaker_embeddings.move_to_end(cache_key)
        if embedding is None:
            # The ONNX session runs concurrent calls; a few slots keep the CPU
            # work bounded beside the server without serializing requests.
            with _speaker_slots:
                extracted = coerce_speaker_embeddings(extractor.extract_from_waveform(waveform, sample_rate))
                if extracted is None:
                    raise ValueError("Speaker extractor returned no embedding")
                [embedding] = extracted
            if cache_key is not None:
                with _speaker_lock:
                    _speaker_embeddings[cache_key] = embedding
                    if len(_speaker_embeddings) > _SPEAKER_CACHE_SIZE:
                        _speaker_embeddings.popitem(last=False)
        rows.append(embedding)
    return torch.stack(rows)


@register_tts_adapter
class Ming3TTSAdapter(ARTTSAdapter):
    name = "ming3_tts"
    model_archs = frozenset({"Ming3TTSForConditionalGeneration", "LingV3TTSForConditionalGeneration"})
    arch_identifies_entry_stage = True
    detect_priority = 190
    max_new_tokens_max = MAX_DECODE_STEPS

    @property
    def tts_config(self) -> dict[str, Any]:
        return dict(getattr(self.ctx.server.engine_client.model_config.hf_config, "tts_config", {}) or {})

    @staticmethod
    def _is_emergent(request: OpenAICreateSpeechRequest) -> bool:
        return (
            request.ref_audio is None
            and request.ref_text is None
            and request.instructions is None
            and request.voice is None
            and request.speaker_embedding is None
            and request.task_type is None
        )

    def validate(self, request: OpenAICreateSpeechRequest) -> str | None:
        server = self.ctx.server
        error = server._apply_uploaded_speaker(request)
        if error:
            return error
        if not request.input or not request.input.strip():
            return "Input text cannot be empty"
        if request.ref_audio is None:
            if request.ref_text is not None:
                return "'ref_text' requires reference audio"
            if not self._is_emergent(request) and not request.instructions:
                return "Ming-3.0-TTS requires reference audio or 'instructions'"
        references = (
            request.ref_audio
            if isinstance(request.ref_audio, list)
            else ([request.ref_audio] if request.ref_audio is not None else [])
        )
        if len(references) > 1:
            if self.tts_config.get("task_decode_alignment", {}).get("podcast") != "old_shift":
                return "Ming3 Podcast requires task_decode_alignment.podcast=old_shift"
            if not request.ref_text or not request.ref_text.strip():
                return "Ming3 Podcast requires an ordered speaker_N reference transcript in 'ref_text'"
            if request.instructions:
                return "Ming3 Podcast does not support 'instructions'"
            speakers = set()
            for line in request.input.splitlines():
                if not line.strip():
                    continue
                match = re.fullmatch(r"\s*speaker_([1-9][0-9]*):\s*(\S.*)", line)
                if match is None:
                    return "Multi-speaker input requires one 'speaker_N: text' turn per line"
                speakers.add(int(match[1]))
            if speakers != set(range(1, len(references) + 1)):
                return "Multi-speaker input must reference each supplied speaker_1 ... speaker_N"
            reference_speakers = re.findall(r"(?m)^\s*speaker_([1-9][0-9]*):\s*\S", request.ref_text)
            if reference_speakers != [str(index) for index in range(1, len(references) + 1)]:
                return "Podcast ref_text must contain one ordered speaker_N turn per reference clip"
        for reference in references:
            format_error = server._validate_ref_audio_format(reference)
            if format_error:
                return format_error
        if request.ref_text is not None and not request.ref_text.strip():
            return "'ref_text' must be non-empty when provided"
        try:
            normalize_target_language(self._target_language(request))
        except ValueError as error:
            return str(error)
        try:
            self._instruction(request.instructions)
        except ValueError as error:
            return str(error)
        extra = request.extra_params or {}
        if extra.get(KEY_CFG) is not None and extra.get("lingv3_cfg") is not None:
            if extra[KEY_CFG] != extra["lingv3_cfg"]:
                return "ming3_cfg and its legacy alias lingv3_cfg must agree"
        for key in (KEY_CFG, "lingv3_cfg", "temperature"):
            value = (request.extra_params or {}).get(key)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
            ):
                return f"Ming-3.0-TTS {key} must be a finite non-negative number"
        if request.speaker_embedding is not None:
            return "Ming-3.0-TTS v1 derives speaker conditioning from 'ref_audio'"
        if request.max_new_tokens is not None:
            minimum = int(self.tts_config.get("minimum_decode_steps", 5))
            if len(references) <= 1 and request.max_new_tokens < minimum:
                return f"max_new_tokens must be at least {minimum}"
            if request.max_new_tokens > MAX_DECODE_STEPS:
                return f"max_new_tokens cannot exceed {MAX_DECODE_STEPS}"
        if len(references) > 1:
            try:
                resolve_podcast_decode_window(request.input, max_decode_steps=request.max_new_tokens)
            except ValueError as error:
                return str(error)
        return None

    @staticmethod
    def _instruction(instructions: str | None) -> str | dict[str, str] | None:
        if instructions is None:
            return None
        instruction = instructions.strip()
        if not instruction:
            raise ValueError("Ming-3.0-TTS instructions must be non-empty")
        try:
            controls = json.loads(instruction)
        except json.JSONDecodeError:
            if instruction.startswith(("{", "[")):
                raise ValueError("Ming-3.0-TTS structured instructions must be a valid JSON object") from None
            if not instruction.startswith("Speaking instruction:"):
                instruction = f"Speaking instruction: Speak in this style: {instruction}"
            return instruction
        allowed = {"语速", "基频", "音量", "情感", "方言", "风格", "IP"}
        if isinstance(controls, dict) and set(controls) == {"audio_sequence"}:
            sequence = controls["audio_sequence"]
            if not isinstance(sequence, list) or len(sequence) != 1 or not isinstance(sequence[0], dict):
                raise ValueError("Ming-3.0-TTS structured instructions require one audio_sequence item")
            item = sequence[0]
            if set(item) != allowed | {"序号", "说话人", "BGM"}:
                raise ValueError("Ming-3.0-TTS structured instructions require the complete single-speaker schema")
            if type(item["序号"]) is not int or item["序号"] != 1 or item["说话人"] != "speaker_1":
                raise ValueError("Ming-3.0-TTS structured instructions require sequence 1 and speaker_1")
            if any(
                item[key] is not None and (not isinstance(item[key], str) or not item[key].strip()) for key in allowed
            ):
                raise ValueError("Ming-3.0-TTS instruction values must be non-empty strings or null")
            bgm = item["BGM"]
            if (
                not isinstance(bgm, dict)
                or set(bgm) != {"Genre", "Mood", "Instrument", "Theme", "ENV", "SNR"}
                or any(value is not None for value in bgm.values())
            ):
                raise ValueError("Ming-3.0-TTS structured instructions require inactive BGM fields")
            # Field order is part of the model's token input. Preserve archived
            # full captions instead of rebuilding them from the short controls.
            return instruction
        if not isinstance(controls, dict) or not controls or not set(controls) <= allowed:
            raise ValueError("Ming-3.0-TTS instructions support 语速, 基频, 音量, 情感, 方言, 风格 and IP")
        if any(not isinstance(value, str) or not value.strip() for value in controls.values()):
            raise ValueError("Ming-3.0-TTS instruction values must be non-empty strings")
        return controls

    def _target_language(self, request: OpenAICreateSpeechRequest) -> str | None:
        if request.language is not None:
            return request.language
        multiple_references = isinstance(request.ref_audio, list) and len(request.ref_audio) > 1
        if not multiple_references or not self.tts_config.get("enable_tts_target_language"):
            return None
        if not self.tts_config.get("infer_tts_language_from_text"):
            raise ValueError("This Ming3 Podcast checkpoint requires an explicit language")
        text = re.sub(r"\[[^\]\n]{1,128}\]|\([A-Za-z][A-Za-z _-]{0,127}\)", "", request.input)
        return "Chinese" if re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", text) else "English"

    def _tokenizer(self) -> Any:
        server = self.ctx.server
        if server._tts_tokenizer is None:
            model_name = server.engine_client.model_config.model
            trust_remote_code = bool(getattr(server.engine_client.model_config, "trust_remote_code", False))
            server._tts_tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
                fix_mistral_regex=True,
            )
        return server._tts_tokenizer

    async def warmup(self) -> None:
        """Load the tokenizer (and the CPU CAM++ session) in each API process at startup.

        On first use they take about 0.7 s and 0.5 s; the tokenizer load also
        blocks the event loop, so the first requests of every API process
        would queue behind both.
        """
        await asyncio.to_thread(self._tokenizer)
        if current_platform.is_cuda():
            return
        try:
            await asyncio.to_thread(
                _extract_speaker_embeddings,
                self.ctx.server.engine_client.model_config.model,
                [torch.zeros(1, 16000)],
                [16000],
                [None],
            )
        except Exception:  # noqa: BLE001 - requests still load the extractor on demand
            logger.warning("Ming3 speaker extractor warmup failed", exc_info=True)

    async def build(
        self,
        request: OpenAICreateSpeechRequest,
        sampling_params_list: list,
        has_inline_ref_audio: bool,
    ) -> PreparedRequest:
        del sampling_params_list
        server = self.ctx.server
        tokenizer = self._tokenizer()
        speaker_audio_keys = []
        waveform, sample_rate = None, None
        if request.ref_audio is not None:
            waveform, sample_rate = [], []
            references = request.ref_audio if isinstance(request.ref_audio, list) else [request.ref_audio]
            for source in references:
                # The array resolver skips the per-sample Python list of the list-based one;
                # the array may be shared with its cache, so it is copied, not wrapped.
                samples, rate, resolved_key = await server._resolve_ref_audio_array(source)
                waveform.append(torch.tensor(samples, dtype=torch.float32).reshape(1, -1))
                sample_rate.append(int(rate))
                speaker_audio_keys.append(server._get_resolved_ref_audio_artifact_key(resolved_key))
        speaker_embedding = None
        if waveform and not current_platform.is_cuda():
            speaker_embedding = await asyncio.to_thread(
                _extract_speaker_embeddings,
                server.engine_client.model_config.model,
                waveform,
                sample_rate,
                speaker_audio_keys,
            )
        max_steps = request.max_new_tokens
        if max_steps is None and len(speaker_audio_keys) <= 1:
            max_steps = int(self.tts_config.get("maximum_decode_steps", MAX_DECODE_STEPS))
        prompt_dict = build_ming3_prompt(
            tokenizer,
            target_text=request.input,
            reference_waveform=waveform,
            reference_sample_rate=sample_rate,
            target_language=self._target_language(request),
            prompt_text=request.ref_text,
            speaker_embedding=speaker_embedding,
            instruction=create_ming3_instruction(self._instruction(request.instructions)),
            max_decode_steps=max_steps,
            speaker_slot=not self._is_emergent(request),
        )
        if speaker_audio_keys:
            prompt_dict["additional_information"]["speaker_audio_keys"] = speaker_audio_keys
        if request.voice and not has_inline_ref_audio and not isinstance(request.ref_audio, list):
            voice_name = request.voice.lower()
            if voice_name in server.uploaded_speakers:
                prompt_dict["additional_information"]["voice_name"] = voice_name
                prompt_dict["additional_information"]["voice_created_at"] = server._voice_created_at(voice_name)
        prompt = tokens_input(prompt_token_ids=prompt_dict["prompt_token_ids"])
        prompt["prompt"] = prompt_dict["prompt"]
        prompt["text"] = prompt_dict["text"]
        prompt["additional_information"] = prompt_dict["additional_information"]
        return PreparedRequest(
            prompt=prompt,
            tts_params=prompt_dict["additional_information"],
            model_type=self.name,
        )

    def apply_sampling_overrides(
        self,
        sampling_params_list: list,
        request: OpenAICreateSpeechRequest,
        prompt: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> list:
        updated = copy.deepcopy(sampling_params_list)
        updated[0].stop_token_ids = [AUDIO_END_TOKEN_ID]
        latent_cap = int(request.max_new_tokens or self.tts_config.get("maximum_decode_steps", MAX_DECODE_STEPS))
        if prompt is not None:
            info = prompt.get("additional_information")
            if isinstance(info, dict):
                latent_cap = int(info.get(KEY_MAX_DECODE_STEPS, latent_cap))
                cfg = (request.extra_params or {}).get(KEY_CFG)
                if cfg is None:
                    cfg = (request.extra_params or {}).get("lingv3_cfg")
                if cfg is not None:
                    info[KEY_CFG] = float(cfg)
                temperature = (request.extra_params or {}).get("temperature")
                if temperature is not None:
                    info[KEY_TEMPERATURE] = float(temperature)
                if request_id is not None:
                    info[KEY_REQUEST_ID] = request_id
                if request.seed is not None:
                    info["tts_local_seed"] = int(request.seed)
        updated[0].max_tokens = latent_cap + 1
        return updated


__all__ = ["Ming3TTSAdapter"]
