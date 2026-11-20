# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
from contextlib import aclosing
from typing import Any

import numpy as np
from vllm.entrypoints.speech_to_text.transcription.serving import OpenAIServingTranscription
from vllm.exceptions import VLLMError
from vllm.logger import init_logger
from vllm.renderers.inputs.preprocess import parse_model_prompt

from vllm_omni.entrypoints.transcription import SameModelTranscriber, TranscriptionSettings

logger = init_logger(__name__)


class _ChunkedTranscriptionPrompt(dict[str, Any]):
    """Private adapter envelope, never submitted to the model or serialized."""

    def __init__(self, chunks: list[np.ndarray], settings: TranscriptionSettings):
        # Keep waveforms out of the upstream request logger's prompt dictionary.
        super().__init__(type="token", prompt_token_ids=[])
        self.chunks = chunks
        self.settings = settings


class _TranscriptionEngineClient:
    """Handler-local view: upstream STT must not use conversational audio routing."""

    def __init__(self, transcriber: SameModelTranscriber):
        self.transcriber = transcriber
        self.model_config = transcriber.stage.model_config
        self.renderer = transcriber.renderer

    def __getattr__(self, name):
        return getattr(self.transcriber.engine, name)

    async def generate(self, prompt, sampling_params, request_id, *, lora_request=None, trace_headers=None):
        if isinstance(prompt, _ChunkedTranscriptionPrompt):
            yield await self.transcriber.transcribe_chunks(
                prompt.chunks,
                prompt.settings,
                request_id,
                sampling_params,
                lora_request=lora_request,
                trace_headers=trace_headers,
            )
            return
        async with aclosing(
            self.transcriber.generate(
                prompt, sampling_params, request_id, lora_request=lora_request, trace_headers=trace_headers
            )
        ) as generator:
            async for output in generator:
                yield output


class OmniServingTranscription(OpenAIServingTranscription):
    def __init__(self, transcriber: SameModelTranscriber, models, **kwargs):
        super().__init__(_TranscriptionEngineClient(transcriber), models, **kwargs)
        self.transcriber = transcriber
        # No conversational generation defaults (speech budget, stops, tools).
        self.default_sampling_params = {}

    def _decode_and_chunk_speech(self, audio_data: bytes) -> tuple[list[np.ndarray], float]:
        waveform, sample_rate = self.transcriber._prepare_encoded(audio_data)
        return [waveform], waveform.size / sample_rate

    async def _preprocess_speech_to_text(self, request, audio_data: bytes, request_id: str):
        request.language = self.model_cls.validate_language(request.language)
        recordings, duration = await self._decode_and_chunk_speech_async(audio_data)
        (waveform,) = recordings
        max_clip_s = self.transcriber.config.max_audio_clip_s
        is_long = max_clip_s is not None and duration > max_clip_s
        chunks = (
            await asyncio.to_thread(self.transcriber._split_recording, waveform, int(self.asr_config.sample_rate))
            if is_long
            else recordings
        )
        if request.language is None and getattr(self.model_cls, "supports_explicit_language_detection", False):
            request.language = await self._detect_language(chunks[0], f"{request_id}-lang_detect")
        if is_long:
            settings = self.transcriber.validate_settings(
                {"model": request.model, "language": request.language, "prompt": request.prompt}
            )
            # One serving-level result preserves HTTP sampling/usage/SSE handling,
            # without rendering (and truncating) the full recording first.
            return [_ChunkedTranscriptionPrompt(chunks, settings)], duration, [0.0]

        stt_params = request.build_stt_params(
            audio=waveform, stt_config=self.asr_config, model_config=self.model_config, task_type=self.task_type
        )
        prompt = self.model_cls.get_generation_prompt(stt_params)
        parsed = parse_model_prompt(self.model_config, prompt)
        engine_inputs = await self.renderer.render_cmpl_async([parsed])
        return engine_inputs, duration, [0.0]

    async def create_transcription(self, audio_data, request, raw_request=None):
        if not request.model:
            request.model = self.models.model_name()
        # The base schema covers models with capabilities the Qwen path does not
        # provide. Reject them rather than returning invented alignment/metadata.
        if (
            request.response_format not in ("json", "text")
            or request.timestamp_granularities
            or request.to_language
            or request.hotwords
            or request.use_beam_search
            or request.vllm_xargs
        ):
            logger.warning("HTTP ASR request rejected: unsupported transcription options")
            return self.create_error_response("Only plain text transcription is supported by this pipeline")
        try:
            self.transcriber.validate_settings(
                {"model": request.model, "language": request.language, "prompt": request.prompt}
            )
        except ValueError as exc:
            logger.warning("HTTP ASR request rejected: %s", exc)
            return self.create_error_response(str(exc))
        try:
            return await super().create_transcription(audio_data, request, raw_request)
        except VLLMError:
            # Engine/client errors belong to the existing HTTP exception handlers.
            raise
        except Exception as exc:
            # Upstream logs streaming failures, but not ordinary non-stream exceptions.
            metadata = getattr(raw_request.state, "request_metadata", None) if raw_request is not None else None
            logger.exception("HTTP ASR failed request_id=%s", getattr(metadata, "request_id", None))
            if isinstance(exc, ValueError):
                return self.create_error_response(
                    "Input transcription failed; check the recording and model audio/context/output limits"
                )
            raise
