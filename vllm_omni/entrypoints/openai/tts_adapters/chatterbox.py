# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Chatterbox speech serving with the built-in voice or a reference recording."""

import asyncio
import math
from collections import OrderedDict
from threading import Lock

import numpy as np
import torch
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerBase
from vllm.utils.async_utils import make_async

from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import (
    ARTTSAdapter,
    PreparedRequest,
    SpeechServingContext,
    apply_max_new_tokens,
    resolve_stage_model_path,
)
from vllm_omni.model_executor.models.chatterbox.conditioning import (
    CONDITIONING_BATCH_SIZE,
    VoiceConditioner,
    VoiceConditioning,
    build_prompt,
    load_text_tokenizer,
    tokenize_text,
)
from vllm_omni.transformers_utils.configs.chatterbox import (
    ORIGINAL_DEFAULT_CFG_WEIGHT,
    ORIGINAL_DEFAULT_EXAGGERATION,
    ORIGINAL_MAX_NEW_TOKENS,
    ChatterboxConfig,
)

CONDITIONING_MAX_PENDING = 64
CONDITIONING_CACHE_ENTRIES = 128


@register_tts_adapter
class ChatterboxAdapter(ARTTSAdapter):
    """Build the same conditioned prompts used by offline inference."""

    name = "chatterbox"
    stage_keys = frozenset({"chatterbox_t3"})
    supported_output_sample_rates = frozenset({24000})
    preprocessing_workers = 4
    variant = "turbo"

    def __init__(self, ctx: SpeechServingContext) -> None:
        super().__init__(ctx)
        self.config = ChatterboxConfig(self.variant)
        # The deploy file's stage 0 ``additional_config`` names where the
        # reference encoders run; the API process must see that device.
        self.conditioning_device = torch.device(self.stage_additional_config().get("conditioning_device", "cpu"))
        if self.conditioning_device.type not in {"cpu", "cuda"}:
            raise ValueError("Chatterbox conditioning_device must be cpu or cuda")
        self.tokenizer: PreTrainedTokenizerBase | Tokenizer | None = None
        self.builtin_voice: VoiceConditioning | None = None
        self.conditioner: VoiceConditioner | None = None
        # Publish fully initialized components before concurrent requests use them.
        self.initialization_lock = Lock()
        self.build_async = make_async(self.prepare_prompt, executor=ctx.server._tts_executor)
        self.condition_batch_async = make_async(self.prepare_voices, executor=ctx.server._tts_executor)
        self.conditioning_slots = asyncio.Semaphore(CONDITIONING_MAX_PENDING)
        self.conditioning_queue: asyncio.Queue = asyncio.Queue(maxsize=CONDITIONING_MAX_PENDING)
        self.conditioning_pending: dict[str, asyncio.Future] = {}
        self.conditioning_cache: OrderedDict[str, VoiceConditioning] = OrderedDict()
        self.conditioning_task: asyncio.Task | None = None
        self.closed = False

    def shutdown(self) -> None:
        """Cancel queued encoder work and wake its waiters before executor shutdown."""
        self.closed = True
        if self.conditioning_task is not None:
            self.conditioning_task.cancel()
        for pending in self.conditioning_pending.values():
            pending.cancel()
        self.conditioning_pending.clear()
        while not self.conditioning_queue.empty():
            self.conditioning_queue.get_nowait()
        self.conditioning_cache.clear()

    def prepare_voices(self, references: list[tuple[np.ndarray, int]]) -> list[VoiceConditioning | Exception]:
        """Run one encoder batch on the server-owned executor."""
        model = resolve_stage_model_path(self.ctx.engine_client)
        if model is None:
            raise ValueError("Chatterbox requires a stage model path")
        with self.initialization_lock:
            if self.conditioner is None:
                self.conditioner = VoiceConditioner(model, self.config, self.conditioning_device)
        results: dict[int, VoiceConditioning | Exception] = {}
        samples = []
        indices = []
        for index, reference in enumerate(references):
            try:
                samples.append(self.conditioner.resample(*reference))
            except ValueError as error:
                results[index] = error
            else:
                indices.append(index)
        if samples:
            for index, voice in zip(indices, self.conditioner.from_resampled_batch(samples), strict=True):
                results[index] = voice
        return [results[index] for index in range(len(references))]

    async def drain_conditioning(self) -> None:
        """Own encoder execution; finish each reference's waiters independently."""
        while not self.conditioning_queue.empty():
            batch = [self.conditioning_queue.get_nowait()]
            while len(batch) < CONDITIONING_BATCH_SIZE and not self.conditioning_queue.empty():
                batch.append(self.conditioning_queue.get_nowait())
            try:
                voices = await self.condition_batch_async([reference for _, reference in batch])
            except Exception as error:
                for key, _ in batch:
                    self.conditioning_pending.pop(key).set_exception(error)
            else:
                for (key, _), voice in zip(batch, voices, strict=True):
                    if isinstance(voice, Exception):
                        self.conditioning_pending.pop(key).set_exception(voice)
                        continue
                    self.conditioning_cache[key] = voice
                    self.conditioning_cache.move_to_end(key)
                    while len(self.conditioning_cache) > CONDITIONING_CACHE_ENTRIES:
                        self.conditioning_cache.popitem(last=False)
                    self.conditioning_pending.pop(key).set_result(voice)

    async def resolve_voice(self, source: str) -> VoiceConditioning:
        """Bound decoded references, coalesce identical misses, and cache artifacts."""
        await self.conditioning_slots.acquire()
        release_slot = True
        try:
            if self.closed:
                raise RuntimeError("Chatterbox conditioning is shut down")
            wav, sample_rate, locator_key = await self.ctx.server._resolve_ref_audio_array(source)
            key = self.ctx.server._get_resolved_ref_audio_artifact_key(locator_key)
            if key is None:
                key = self.ctx.server._make_ref_audio_artifact_cache_key(wav, sample_rate)
            cached = self.conditioning_cache.get(key)
            if cached is not None:
                self.conditioning_cache.move_to_end(key)
                return cached
            pending = self.conditioning_pending.get(key)
            if pending is None:
                pending = asyncio.get_running_loop().create_future()
                # The permit belongs to encoder work, not its HTTP waiter:
                # disconnects cannot release capacity while shielded work remains.
                pending.add_done_callback(lambda future: self.conditioning_slots.release())
                pending.add_done_callback(lambda future: future.exception() if not future.cancelled() else None)
                release_slot = False
                self.conditioning_pending[key] = pending
                self.conditioning_queue.put_nowait((key, (wav, sample_rate)))
                if self.conditioning_task is None or self.conditioning_task.done():
                    self.conditioning_task = asyncio.create_task(self.drain_conditioning())
            return await asyncio.shield(pending)
        finally:
            if release_slot:
                self.conditioning_slots.release()

    def validate(self, request: OpenAICreateSpeechRequest) -> str | None:
        """Reject requests whose controls the Turbo checkpoint cannot honor."""
        error = self.ctx.server._apply_uploaded_speaker(request)
        if error:
            return error
        if not request.input.strip():
            return "Input text cannot be empty"
        if request.max_new_tokens is not None and request.max_new_tokens > self.max_new_tokens_max:
            return f"Chatterbox Turbo supports at most {self.max_new_tokens_max} new speech tokens"
        if request.language is not None and request.language.lower() not in {"auto", "en", "english"}:
            return "Chatterbox Turbo supports English only"
        if request.instructions:
            return "Chatterbox Turbo does not support instructions"
        if request.task_type == "VoiceDesign":
            return "Chatterbox Turbo does not support VoiceDesign"
        if request.x_vector_only_mode:
            return "Chatterbox Turbo requires full reference conditioning"
        if self.variant == "turbo" and {"exaggeration", "cfg_weight"}.intersection(request.extra_params or {}):
            return "Exaggeration and CFG controls require Chatterbox Original"
        if request.ref_audio_2 is not None or request.speaker_embedding is not None:
            return "Chatterbox Turbo accepts one reference recording"
        if request.ref_audio is None:
            if request.voice not in (None, "", "default"):
                return "Chatterbox Turbo supports voice='default' or an uploaded reference voice"
        else:
            reference = request.ref_audio
            if isinstance(reference, list):
                if len(reference) != 1:
                    return "Chatterbox Turbo accepts one reference recording"
                reference = reference[0]
            error = self.ctx.server._validate_ref_audio_format(reference)
            if error:
                return error
        return None

    def encode_text(self, model: str, text: str) -> list[int]:
        """Load the variant tokenizer once and encode outside the event loop."""
        with self.initialization_lock:
            if self.tokenizer is None:
                self.tokenizer = load_text_tokenizer(model, self.variant)
        return tokenize_text(text, self.tokenizer, self.variant)

    def prepare_prompt(
        self,
        text: str,
        exaggeration: float = ORIGINAL_DEFAULT_EXAGGERATION,
        cfg_weight: float = ORIGINAL_DEFAULT_CFG_WEIGHT,
        voice: VoiceConditioning | None = None,
    ) -> dict:
        """Load and condition on a worker thread, outside the API event loop."""
        model = resolve_stage_model_path(self.ctx.engine_client)
        if model is None:
            raise ValueError("Chatterbox requires a stage model path")
        if voice is None:
            with self.initialization_lock:
                if self.builtin_voice is None:
                    self.builtin_voice = VoiceConditioning.from_builtin(model)
                voice = self.builtin_voice
        text_ids = self.encode_text(model, text)
        return build_prompt(text_ids, voice, self.config, exaggeration=exaggeration, cfg_weight=cfg_weight)

    async def build(
        self, request: OpenAICreateSpeechRequest, sampling_params_list: list, has_inline_ref_audio: bool
    ) -> PreparedRequest:
        """Resolve reference audio and build the engine's conditioned token prompt."""
        voice = None
        if request.ref_audio is not None:
            source = request.ref_audio[0] if isinstance(request.ref_audio, list) else request.ref_audio
            voice = await self.resolve_voice(source)
        controls = request.extra_params or {}
        prompt = await self.build_async(
            request.input,
            controls.get("exaggeration", ORIGINAL_DEFAULT_EXAGGERATION),
            controls.get("cfg_weight", ORIGINAL_DEFAULT_CFG_WEIGHT),
            voice,
        )
        return PreparedRequest(prompt=prompt, model_type=self.name)

    def apply_sampling_overrides(
        self, sampling_params_list: list, request: OpenAICreateSpeechRequest, prompt=None, request_id=None
    ) -> list:
        """Honor a request's speech-token cap through the shared adapter helper."""
        return apply_max_new_tokens(sampling_params_list, request)

    def _load_supported_speakers(self) -> list[str]:
        return ["default"]

    def _load_supported_languages(self) -> frozenset[str]:
        return frozenset({"English"})

    def _load_codec_frame_rate(self) -> float:
        return self.config.token_rate


@register_tts_adapter
class ChatterboxOriginalAdapter(ChatterboxAdapter):
    """English Original with its released tokenizer and per-request CFG controls."""

    name = "chatterbox_original"
    stage_keys = frozenset({"chatterbox_original_t3"})
    variant = "original"
    max_new_tokens_max = ORIGINAL_MAX_NEW_TOKENS

    def validate(self, request: OpenAICreateSpeechRequest) -> str | None:
        error = super().validate(request)
        if error:
            return error.replace("Chatterbox Turbo", "Chatterbox Original")
        for name, default in (
            ("exaggeration", ORIGINAL_DEFAULT_EXAGGERATION),
            ("cfg_weight", ORIGINAL_DEFAULT_CFG_WEIGHT),
        ):
            value = (request.extra_params or {}).get(name, default)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                return f"{name} must be a finite nonnegative number"
        return None
