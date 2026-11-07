# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Same-model speech recognition, independent of conversational generation."""

from __future__ import annotations

import asyncio
import io
from collections.abc import AsyncIterator
from contextlib import aclosing
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import BaseModel, ConfigDict, ValidationError
from vllm import envs
from vllm.config import ModelConfig
from vllm.config.speech_to_text import SpeechToTextParams
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsTranscription, supports_transcription
from vllm.multimodal.audio import split_audio
from vllm.multimodal.media.audio import load_audio
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.renderers.inputs.preprocess import parse_model_prompt
from vllm.sampling_params import RequestOutputKind, SamplingParams

from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.utils.audio_resample import StreamingAudioResampler

if TYPE_CHECKING:
    from vllm_omni.entrypoints.async_omni import AsyncOmni

logger = init_logger(__name__)


class TranscriptionSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str
    language: str | None = None
    prompt: str | None = None


@dataclass(frozen=True)
class TranscriptionStage:
    stage_id: int
    model_config: ModelConfig
    model_cls: type[SupportsTranscription]


def resolve_transcription_stage(engine: AsyncOmni) -> TranscriptionStage | None:
    """Only expose a reachable transcription-capable comprehension/text endpoint."""
    from vllm.model_executor.model_loader import get_model_cls

    stage_id = engine._get_comprehension_stage_index()
    if stage_id is None or "text" not in engine.output_modalities:
        return None
    stage = engine.engine.stage_configs[stage_id]
    if not stage.is_comprehension or not stage.final_output or stage.final_output_type != "text":
        return None
    if engine._compute_final_output_stage_ids(["text"]) != [stage_id]:
        return None
    if stage_id:
        try:
            engine.get_stage_renderer(stage_id)
        except KeyError:
            return None
    model_config = engine.engine.stage_vllm_configs[stage_id].model_config
    model_cls = get_model_cls(model_config)
    if not supports_transcription(model_cls):
        return None
    return TranscriptionStage(stage_id, model_config, model_cls)


class SameModelTranscriber:
    def __init__(self, engine: AsyncOmni, stage: TranscriptionStage, model_names: set[str]):
        self.engine = engine
        self.stage = stage
        self.model_names = frozenset(model_names)
        self.config = stage.model_cls.get_speech_to_text_config(stage.model_config, "transcribe")
        self.renderer = engine.get_stage_renderer(stage.stage_id) if stage.stage_id else engine.renderer
        self.max_file_bytes = envs.VLLM_MAX_AUDIO_CLIP_FILESIZE_MB * 1024 * 1024
        logger.info(
            "[transcription] Same-model ASR available models=%s stage_id=%s", sorted(model_names), stage.stage_id
        )

    def validate_settings(self, settings: Any) -> TranscriptionSettings:
        values = settings.model_dump(exclude_none=True) if isinstance(settings, BaseModel) else settings
        try:
            accepted = TranscriptionSettings.model_validate(values)
        except ValidationError as exc:
            raise ValueError("Unsupported input transcription settings; use model, language and prompt only") from exc
        if accepted.model not in self.model_names:
            raise ValueError("Requested transcription model is not a served base model")
        if accepted.language and accepted.language not in self.stage.model_cls.supported_languages:
            raise ValueError("Requested transcription language is not supported")
        return accepted

    def _normalize_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if sample_rate <= 0 or audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
            raise ValueError("Invalid transcription audio")
        logger.debug(
            "[transcription] Audio samples=%d duration_s=%.3f source_rate=%s target_rate=%s",
            audio.size,
            audio.size / sample_rate,
            sample_rate,
            self.config.sample_rate,
        )
        if sample_rate != self.config.sample_rate:
            audio = StreamingAudioResampler(sample_rate, int(self.config.sample_rate)).process(audio, final=True)
        return audio

    def _prepare_audio(self, audio: np.ndarray, sample_rate: int, settings: TranscriptionSettings):
        audio = self._normalize_audio(audio, sample_rate)
        # The feature extractor can truncate oversized clips. Every private
        # inference still has to fit the model's unchanged per-clip limit.
        if self.config.max_audio_clip_s and audio.size > self.config.sample_rate * self.config.max_audio_clip_s:
            raise ValueError("Audio exceeds this model's transcription duration limit")
        return self.stage.model_cls.get_generation_prompt(
            SpeechToTextParams(
                audio=audio,
                stt_config=self.config,
                model_config=self.stage.model_config,
                language=settings.language,
                task_type="transcribe",
                request_prompt=settings.prompt or "",
            )
        )

    async def transcribe_pcm(
        self, audio: bytes, sample_rate: int, settings: TranscriptionSettings, request_id: str
    ) -> str:
        settings = self.validate_settings(settings)
        return await self._transcribe_waveform(
            np.frombuffer(audio, dtype="<i2").astype(np.float32) / 32768.0, sample_rate, settings, request_id
        )

    def _prepare_encoded(self, audio: bytes) -> tuple[np.ndarray, int]:
        if len(audio) > self.max_file_bytes:
            raise ValueError("Transcription recording exceeds the upload size limit")
        return load_audio(
            io.BytesIO(audio),
            sr=int(self.config.sample_rate),
            mono=True,
            max_duration_s=envs.VLLM_MAX_AUDIO_DECODE_DURATION_S,
            max_decode_bytes=envs.VLLM_MAX_AUDIO_DECODE_BYTES,
        )

    async def transcribe_encoded(self, audio: bytes, settings: TranscriptionSettings, request_id: str) -> str:
        settings = self.validate_settings(settings)
        waveform, sample_rate = await asyncio.to_thread(self._prepare_encoded, audio)
        return await self._transcribe_waveform(waveform, sample_rate, settings, request_id)

    def _split_recording(self, audio: np.ndarray, sample_rate: int) -> list[np.ndarray]:
        audio = self._normalize_audio(audio, sample_rate)
        sample_rate = int(self.config.sample_rate)
        max_clip_s = self.config.max_audio_clip_s
        if max_clip_s is not None and int(sample_rate * max_clip_s) < 1:
            raise ValueError("Invalid transcription clip duration")
        if max_clip_s is None or audio.size <= sample_rate * max_clip_s:
            return [audio]
        # The overlap is a boundary-search region, not duplicated audio. Use
        # 100 ms energy windows locally; do not enable the model's global chunker.
        return split_audio(
            audio,
            sample_rate=sample_rate,
            max_clip_duration_s=max_clip_s,
            overlap_duration_s=min(1.0, max_clip_s / 2),
            min_energy_window_size=max(1, int(sample_rate * min(0.1, max_clip_s / 2))),
        )

    async def _transcribe_waveform(
        self, audio: np.ndarray, sample_rate: int, settings: TranscriptionSettings, request_id: str
    ) -> str:
        max_clip_s = self.config.max_audio_clip_s
        if max_clip_s is None or audio.size <= sample_rate * max_clip_s:
            prompt = await asyncio.to_thread(self._prepare_audio, audio, sample_rate, settings)
            return await self.transcribe_prompt(prompt, request_id)

        chunks = await asyncio.to_thread(self._split_recording, audio, sample_rate)
        # The views retain the normalized waveform, not an extra source-rate copy.
        del audio
        result = await self.transcribe_chunks(chunks, settings, request_id)
        return result.outputs[0].text

    async def transcribe_chunks(
        self,
        chunks: list[np.ndarray],
        settings: TranscriptionSettings,
        request_id: str,
        sampling_params: SamplingParams | None = None,
        *,
        lora_request=None,
        trace_headers=None,
    ) -> RequestOutput:
        """Run private clips sequentially; publish nothing until every clip succeeds."""
        params = (
            sampling_params.clone() if sampling_params is not None else SamplingParams(temperature=0, max_tokens=None)
        )
        params.output_kind = RequestOutputKind.CUMULATIVE
        parts = []
        prompt_token_ids = []
        token_ids = []
        for index, chunk in enumerate(chunks):
            prompt = await asyncio.to_thread(self._prepare_audio, chunk, int(self.config.sample_rate), settings)
            parsed = parse_model_prompt(self.stage.model_config, prompt)
            (engine_input,) = await self.renderer.render_cmpl_async([parsed])
            completion = None
            async with aclosing(
                self.generate(
                    engine_input,
                    params,
                    f"{request_id}-chunk-{index}",
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                )
            ) as generator:
                async for output in generator:
                    if output.outputs:
                        (completion,) = output.outputs
            if completion is None:
                raise ValueError("Transcription ended without a complete recognition result")
            text = self.stage.model_cls.post_process_output(completion.text).strip()
            # Only digital silence is provably safe to omit without a speech
            # detector. Empty recognition of a nonzero clip cannot certify completeness.
            if not text and np.any(chunk):
                raise ValueError("Transcription returned no words for a non-silent chunk")
            if text:
                parts.append(text)
            prompt_token_ids.extend(engine_input["prompt_token_ids"])
            token_ids.extend(completion.token_ids)
            del prompt, parsed, engine_input, completion, output
        separator = "" if settings.language in self.stage.model_cls.no_space_languages else " "
        return RequestOutput(
            request_id=request_id,
            prompt=None,
            prompt_token_ids=prompt_token_ids,
            prompt_logprobs=None,
            outputs=[
                CompletionOutput(
                    index=0,
                    text=separator.join(parts),
                    token_ids=token_ids,
                    cumulative_logprob=None,
                    logprobs=None,
                    finish_reason="stop",
                )
            ],
            finished=True,
        )

    async def transcribe_prompt(self, prompt, request_id: str) -> str:
        # None lets the stage's real context window bound generation, not an old
        # fixed short ASR budget. A length stop below is a failure, never success.
        params = SamplingParams(temperature=0, max_tokens=None, output_kind=RequestOutputKind.CUMULATIVE)
        parsed = parse_model_prompt(self.stage.model_config, prompt)
        (engine_input,) = await self.renderer.render_cmpl_async([parsed])
        text = ""
        async with aclosing(self.generate(engine_input, params, request_id)) as generator:
            async for output in generator:
                for completion in output.outputs:
                    text = completion.text  # CUMULATIVE is explicit above.
        return self.stage.model_cls.post_process_output(text).strip()

    async def generate(
        self, prompt, params: SamplingParams, request_id: str, *, lora_request=None, trace_headers=None
    ) -> AsyncIterator[OmniRequestOutput]:
        params = params.clone()
        available = self.stage.model_config.max_model_len - len(prompt["prompt_token_ids"])
        if available <= 0:
            raise ValueError("Transcription audio exceeds the model context window")
        params.max_tokens = min(params.max_tokens or available, available)
        stage_params = list(self.engine.default_sampling_params_list)
        stage_params[self.stage.stage_id] = params
        logger.debug(
            "[transcription] Generate request_id=%s stage_id=%s prompt_tokens=%d max_tokens=%s",
            request_id,
            self.stage.stage_id,
            len(prompt["prompt_token_ids"]),
            params.max_tokens,
        )
        finish_reason = None
        generator = self.engine.generate(
            prompt=prompt,
            request_id=request_id,
            sampling_params_list=stage_params,
            output_modalities=["text"],
            entry_stage_id=self.stage.stage_id,
            lora_request=lora_request,
            trace_headers=trace_headers,
        )
        # Closing on cancellation/early exit invokes AsyncOmni's bounded abort
        # path; recognition never becomes the session's active assistant response.
        async with aclosing(generator):
            async for output in generator:
                if output.stage_id != self.stage.stage_id or output.error:
                    raise RuntimeError("Unexpected transcription stage output")
                for completion in output.outputs:
                    if completion.finish_reason is not None:
                        finish_reason = completion.finish_reason
                        if finish_reason != "stop":
                            raise ValueError("Transcription did not finish within the model's output budget")
                yield output
        if finish_reason != "stop":
            raise ValueError("Transcription ended without a complete recognition result")
