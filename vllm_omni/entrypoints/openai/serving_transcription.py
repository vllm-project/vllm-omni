# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Transcription with an optional forced-aligner pipeline stage."""

import asyncio
import json
import math
from collections.abc import AsyncGenerator, Mapping
from copy import copy
from typing import Annotated

from fastapi import APIRouter, Form, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from transformers.models.whisper.tokenization_whisper import LANGUAGES
from vllm.entrypoints.generate.base.protocol import RequestResponseMetadata
from vllm.entrypoints.serve.engine.protocol import ErrorResponse, UsageInfo
from vllm.entrypoints.serve.utils.api_utils import get_max_tokens, load_aware_call, with_cancellation
from vllm.entrypoints.speech_to_text.base.serving import asr_inter_chunk_separator
from vllm.entrypoints.speech_to_text.base.utils import read_upload_with_limit
from vllm.entrypoints.speech_to_text.transcription.protocol import (
    TranscriptionRequest,
    TranscriptionResponse,
    TranscriptionResponseVariant,
    TranscriptionResponseVerbose,
    TranscriptionWord,
)
from vllm.entrypoints.speech_to_text.transcription.serving import OpenAIServingTranscription
from vllm.inputs import split_enc_dec_input
from vllm.outputs import RequestOutput
from vllm.renderers.inputs.preprocess import parse_model_prompt
from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.utils.async_utils import merge_async_iterators

from vllm_omni.model_executor.models.interfaces import get_transcription_language
from vllm_omni.model_executor.stage_input_processors.forced_aligner import parse_asr_output
from vllm_omni.utils.forced_aligner import extract_word_timestamps
from vllm_omni.utils.qwen3_force_align_processor import segment_words

router = APIRouter()


def _resolve_language_code(language: str, supported_languages: Mapping[str, str]) -> str:
    """Map detected language names to ISO codes, preserving explicit codes."""
    return next(
        (code for code, name in {**LANGUAGES, **supported_languages}.items() if name.lower() == language.lower()),
        language,
    )


class TextOnlyEngineClient:
    """Keep native ASR and translation requests on the transcript stage."""

    def __init__(self, engine_client):
        self._engine_client = engine_client

    def __getattr__(self, name):
        return getattr(self._engine_client, name)

    def generate(self, *args, **kwargs):
        kwargs.setdefault("output_modalities", ["text"])
        return self._engine_client.generate(*args, **kwargs)


@router.post("/v1/audio/transcriptions")
@with_cancellation
@load_aware_call
async def create_transcriptions(raw_request: Request, request: Annotated[TranscriptionRequest, Form()]):
    handler = raw_request.app.state.openai_serving_transcription
    audio_data = await read_upload_with_limit(request.file)
    result = await handler.create_transcription(audio_data, request, raw_request)
    if isinstance(result, ErrorResponse):
        return JSONResponse(content=result.model_dump(), status_code=result.error.code)
    if isinstance(result, TranscriptionResponseVariant):
        if request.response_format == "text":
            return PlainTextResponse(result.text)
        return JSONResponse(content=result.model_dump())
    return StreamingResponse(content=result, media_type="text/event-stream")


class OmniOpenAIServingTranscription(OpenAIServingTranscription):
    """Expose native transcription and optional forced word timestamps.

    The upstream verbose path decodes native segment timestamp tokens. Forced
    word alignment instead consumes the transcript in a separate pooling stage.
    """

    def __init__(self, engine_client, models, *, has_asr_aligner: bool = False, **kwargs):
        super().__init__(TextOnlyEngineClient(engine_client), models, **kwargs)
        self.has_asr_aligner = has_asr_aligner

    async def transcription_stream_generator(
        self,
        request: TranscriptionRequest,
        result_generator: list[AsyncGenerator[RequestOutput, None]],
        request_id: str,
        request_metadata: RequestResponseMetadata,
        audio_duration_s: float,
        separator: str,
    ) -> AsyncGenerator[str, None]:
        """Join decoded chunks and aggregate usage while reusing vLLM SSE."""
        completed = False
        failed = False
        has_text = False
        prompt_tokens = 0
        last_prompt_tokens = 0
        include_usage = self.enable_force_include_usage or request.stream_include_usage
        language = request.language
        model_cls = self.model_cls
        model_post_processor_cls = self.streaming_post_processor_cls

        class ChunkJoiningPostProcessor:
            def __init__(self):
                self.processor = model_post_processor_cls()
                self.prefix = ""
                self.beginning = True

            def process_delta(self, text, finished):
                nonlocal has_text, language, separator
                if self.beginning:
                    self.prefix += text
                text = self.processor.process_delta(text, finished)
                if self.beginning and (text or finished):
                    if language is None:
                        language = get_transcription_language(model_cls, self.prefix)
                        if language:
                            language_code = _resolve_language_code(language, model_cls.supported_languages)
                            separator = asr_inter_chunk_separator(language_code, model_cls.no_space_languages)
                    self.prefix = ""
                    if text:
                        if has_text:
                            text = separator + text
                        self.beginning = False
                        has_text = True
                return text

        request_ids = [
            request_id if len(result_generator) == 1 else f"{request_id}-{index}"
            for index in range(len(result_generator))
        ]

        async def checked(source):
            nonlocal failed, prompt_tokens, last_prompt_tokens
            chunk_prompt_tokens = 0
            try:
                async for result in source:
                    if result.error:
                        raise RuntimeError(result.error)
                    if len(result.outputs) != 1:
                        raise RuntimeError("ASR streaming requires exactly one transcription output.")
                    if result.outputs[0].finish_reason == "error":
                        raise RuntimeError("ASR generation failed.")
                    if result.prompt_token_ids is not None:
                        last_prompt_tokens = len(result.prompt_token_ids)
                        prompt_tokens += last_prompt_tokens - chunk_prompt_tokens
                        chunk_prompt_tokens = last_prompt_tokens
                    yield result
            except Exception:
                failed = True
                raise

        sources = [checked(source) for source in result_generator]
        # The native generator creates a postprocessor for each audio chunk.
        # Override it on a request-local view so concurrent streams never share
        # joining state. Separators belong after model-specific prefix removal.
        stream_serving = copy(self)
        stream_serving.streaming_post_processor_cls = ChunkJoiningPostProcessor
        stream = OpenAIServingTranscription.transcription_stream_generator(
            stream_serving, request, sources, request_id, request_metadata, audio_duration_s, ""
        )
        try:
            async for event in stream:
                # Native usage retains only the latest chunk's prompt count.
                # Add the other chunks without changing its completion count
                # or model-specific audio-token estimate for the full audio.
                prompt_token_offset = prompt_tokens - last_prompt_tokens
                if event == "data: [DONE]\n\n":
                    usage = request_metadata.final_usage_info
                    if usage is not None:
                        usage.prompt_tokens += prompt_token_offset
                        usage.total_tokens = usage.prompt_tokens + usage.completion_tokens
                elif include_usage and prompt_token_offset:
                    chunk = json.loads(event.removeprefix("data: "))
                    if chunk_usage := chunk.get("usage"):
                        chunk_usage["prompt_tokens"] += prompt_token_offset
                        chunk_usage["total_tokens"] = chunk_usage["prompt_tokens"] + chunk_usage["completion_tokens"]
                        event = f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                yield event
            completed = True
        finally:
            if failed or not completed:
                await asyncio.gather(self.engine_client.abort(request_ids), return_exceptions=True)
            await asyncio.gather(stream.aclose(), return_exceptions=True)
            await asyncio.gather(*(source.aclose() for source in [*sources, *result_generator]), return_exceptions=True)

    async def create_transcription(
        self,
        audio_data: bytes,
        request: TranscriptionRequest,
        raw_request: Request | None = None,
    ):
        is_qwen3_asr = getattr(getattr(self.model_config, "hf_config", None), "model_type", None) == "qwen3_asr"
        # Qwen retains its verbose_json word-alignment default. Other ASR
        # models opt into forced alignment by explicitly requesting words.
        if not is_qwen3_asr and not (self.has_asr_aligner and "word" in request.timestamp_granularities):
            # Preserve native prompt handling, beam search, segment timestamps
            # and diarization whenever forced word alignment is not requested.
            return await super().create_transcription(audio_data, request, raw_request)

        error = await self._check_model(request)
        if error is not None:
            return error
        if not request.model:
            request.model = self.models.model_name()
        self._preflight()

        verbose = request.response_format == "verbose_json"
        if request.response_format not in {"json", "text", "verbose_json"}:
            return self.create_error_response("ASR forced alignment supports json, text and verbose_json responses.")
        if request.use_beam_search:
            return self.create_error_response("Beam search is not supported with ASR forced word alignment.")
        if request.timestamp_granularities and not verbose:
            return self.create_error_response("timestamp_granularities requires response_format=verbose_json.")
        if "segment" in request.timestamp_granularities:
            return self.create_error_response("ASR forced alignment currently supports word timestamps only.")
        if verbose and request.stream:
            return self.create_error_response("verbose_json word timestamps do not support streaming.")
        if verbose and not self.has_asr_aligner:
            return self.create_error_response("ASR word timestamps require an ASR stage connected to a forced aligner.")

        request_id = f"transcribe-{self._base_request_id(raw_request)}"
        metadata = RequestResponseMetadata(request_id=request_id)
        if raw_request is not None:
            raw_request.state.request_metadata = metadata
        engine_request_ids: list[str] = []

        try:
            request.language = self.model_cls.validate_language(request.language)
            if request.to_language:
                request.to_language = self.model_cls.validate_language(request.to_language)
            chunks, duration = await self._decode_and_chunk_speech_async(audio_data)
            if not chunks or any(chunk.size == 0 for chunk in chunks):
                return self.create_error_response("The audio file must contain audio samples.")
            if request.language is None and getattr(self.model_cls, "supports_explicit_language_detection", False):
                request.language = await self._detect_language(chunks[0], f"{request_id}-lang_detect")
            lora_request = self._maybe_get_adapters(request)
            trace_headers = None if raw_request is None else await self._get_trace_headers(raw_request.headers)
            generators = []
            offsets = []
            offset = 0.0
            for index, chunk in enumerate(chunks):
                stt_params = request.build_stt_params(
                    audio=chunk,
                    stt_config=self.asr_config,
                    model_config=self.model_config,
                    task_type="transcribe",
                )
                parsed_prompt = parse_model_prompt(self.model_config, self.model_cls.get_generation_prompt(stt_params))
                (engine_input,) = await self.renderer.render_cmpl_async([parsed_prompt])
                prompt = dict(engine_input)
                # Keep a raw waveform sidecar for downstream alignment without
                # altering the model's rendered encoder/decoder or MM features.
                # InputProcessor accepts the typed EngineInput without rendering
                # it again; AsyncOmni retains this dict as the original prompt.
                prompt["multi_modal_data"] = {"audio": (chunk, int(self.asr_config.sample_rate))}
                if request.language:
                    prompt["additional_information"] = {
                        **(prompt.get("additional_information") or {}),
                        "language": [request.language],
                    }
                _, decoder_input = split_enc_dec_input(prompt)
                max_tokens = get_max_tokens(
                    self.model_config.max_model_len,
                    request.max_completion_tokens,
                    length_from_prompt_token_ids_or_embeds(
                        decoder_input.get("prompt_token_ids"), decoder_input.get("prompt_embeds")
                    ),
                    self.default_sampling_params,
                )
                params = request.to_sampling_params(max_tokens, self.default_sampling_params)
                chunk_id = request_id if len(chunks) == 1 else f"{request_id}-{index}"
                engine_request_ids.append(chunk_id)
                self._log_inputs(chunk_id, prompt, params=params, lora_request=lora_request)
                generators.append(
                    self.engine_client.generate(
                        prompt,
                        params,
                        chunk_id,
                        output_modalities=["text", "timestamps"] if verbose else ["text"],
                        lora_request=lora_request,
                        trace_headers=trace_headers,
                    )
                )
                offsets.append(offset)
                offset += len(chunk) / self.asr_config.sample_rate

            separator = asr_inter_chunk_separator(request.language, self.model_cls.no_space_languages)
            if request.stream:
                return self.transcription_stream_generator(
                    request, generators, request_id, metadata, duration, separator
                )

            texts = [""] * len(chunks)
            languages: list[str | None] = [None] * len(chunks)
            alignments = [None] * len(chunks)
            seen_text = [False] * len(chunks)
            prompt_tokens = [0] * len(chunks)
            completion_tokens = [0] * len(chunks)
            async for index, result in merge_async_iterators(*generators):
                if result.error:
                    raise RuntimeError(result.error)
                if result.final_output_type == "timestamps":
                    alignments[index] = result
                    continue
                if result.final_output_type != "text":
                    continue
                output = result.outputs[0]
                if output.finish_reason == "error":
                    raise RuntimeError("ASR generation failed.")
                texts[index], languages[index] = parse_asr_output(
                    output.text, model_config=self.model_config, model_cls=self.model_cls
                )
                seen_text[index] = True
                prompt_tokens[index] = len(result.prompt_token_ids or [])
                completion_tokens[index] = len(output.token_ids)

            if not all(seen_text):
                raise RuntimeError("ASR did not return a transcription for every audio chunk.")
            language = request.language or next((lang for lang in languages if lang), "")
            language_code = _resolve_language_code(language, self.model_cls.supported_languages)
            separator = asr_inter_chunk_separator(language_code, self.model_cls.no_space_languages)
            text = separator.join(text for text in texts if text)
            metadata.final_usage_info = UsageInfo(
                prompt_tokens=sum(prompt_tokens),
                completion_tokens=sum(completion_tokens),
                total_tokens=sum(prompt_tokens) + sum(completion_tokens),
            )
            if not verbose:
                return TranscriptionResponse(text=text, usage={"type": "duration", "seconds": math.ceil(duration)})

            words: list[TranscriptionWord] = []
            for index, transcript in enumerate(texts):
                if alignments[index] is None:
                    raise RuntimeError("The forced-aligner stage did not return word timestamps.")
                chunk_language = request.language or languages[index]
                timestamps = extract_word_timestamps(alignments[index], transcript, chunk_language)
                if timestamps is None:
                    if not segment_words(transcript, chunk_language):
                        continue
                    raise RuntimeError("The forced-aligner output did not match the ASR transcript.")
                words.extend(
                    TranscriptionWord(
                        word=item["word"],
                        start=min(duration, offsets[index] + item["start_ms"] / 1000),
                        end=min(duration, offsets[index] + item["end_ms"] / 1000),
                    )
                    for item in timestamps
                )
            return TranscriptionResponseVerbose(duration=duration, language=language, text=text, words=words)
        except asyncio.CancelledError:
            await asyncio.gather(self.engine_client.abort(engine_request_ids), return_exceptions=True)
            raise
        except ValueError as exc:
            if engine_request_ids:
                await asyncio.gather(self.engine_client.abort(engine_request_ids), return_exceptions=True)
            return self.create_error_response(str(exc))
        except Exception:
            await asyncio.gather(self.engine_client.abort(engine_request_ids), return_exceptions=True)
            raise
