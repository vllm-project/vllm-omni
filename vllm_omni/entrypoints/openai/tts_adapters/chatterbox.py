# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Chatterbox speech serving with the built-in voice or a reference recording."""

import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
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
    VoiceConditioner,
    VoiceConditioning,
    build_prompt,
    load_text_tokenizer,
    tokenize_text,
)
from vllm_omni.transformers_utils.configs.chatterbox import ChatterboxConfig

# Original's per-request controls, read from ``extra_params``.
CONTROLS = frozenset({"exaggeration", "cfg_weight"})


@register_tts_adapter
class ChatterboxAdapter(ARTTSAdapter):
    """Build the same conditioned prompts used by offline inference."""

    name = "chatterbox"
    stage_keys = frozenset({"chatterbox_t3"})
    supported_output_sample_rates = frozenset({24000})
    variant = "turbo"
    # Stage 0 sequences one request holds: Original's guided requests run in pairs.
    sequences_per_request = 1
    # The request controls the checkpoint honors.
    controls: frozenset[str] = frozenset()

    def __init__(self, ctx: SpeechServingContext) -> None:
        super().__init__(ctx)
        self.config = ChatterboxConfig(self.variant)
        self.max_new_tokens_max = self.config.max_new_tokens
        # The deploy file's stage 0 ``additional_config`` names where the
        # reference encoders run (the API process must see that device) and
        # the controls' request defaults.
        additional_config = self.stage_setting("runtime_config", "additional_config") or {}
        if "conditioning_device" not in additional_config:
            raise ValueError("Chatterbox deploy files set stage 0 additional_config.conditioning_device")
        self.conditioning_device = torch.device(additional_config["conditioning_device"])
        if not self.controls <= additional_config.keys():
            raise ValueError(
                f"Chatterbox {self.variant.capitalize()} deploy files set stage 0 "
                f"additional_config.{' and '.join(sorted(self.controls))}"
            )
        self.default_controls = {name: additional_config[name] for name in self.controls}
        # As many references are encoded together as stage 0 admits requests:
        # the deploy profile sizes that for its card, and nothing here divides it.
        max_num_seqs = self.stage_setting("scheduler_config", "max_num_seqs")
        if max_num_seqs is None:
            raise ValueError("Chatterbox deploy files set stage 0 max_num_seqs")
        self.conditioning_capacity = max_num_seqs // self.sequences_per_request
        if self.conditioning_device.type not in {"cpu", "cuda"}:
            raise ValueError("Chatterbox conditioning_device must be cpu or cuda")
        model = resolve_stage_model_path(ctx.engine_client)
        if model is None:
            raise ValueError("Chatterbox requires a stage model path")
        # Loaded with the server, so no request waits on a first-use load.
        self.tokenizer = load_text_tokenizer(model, self.variant)
        self.builtin_voice = VoiceConditioning.from_builtin(model)
        self.conditioner = VoiceConditioner(model, self.config, self.conditioning_device)
        self.build_async = make_async(self.prepare_prompt, executor=ctx.server._tts_executor)
        # Encoder batches get a thread of their own, so a prompt build (the
        # built-in voice, a cached reference) never waits behind one; the
        # drain task already runs one batch at a time.
        self.encoder_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="chatterbox-encoder")
        self.condition_batch_async = make_async(self.prepare_voices, executor=self.encoder_executor)
        self.conditioning_slots = asyncio.Semaphore(self.conditioning_capacity)
        self.conditioning_queue: asyncio.Queue = asyncio.Queue(maxsize=self.conditioning_capacity)
        self.conditioning_pending: dict[str, asyncio.Future] = {}
        self.conditioning_task: asyncio.Task | None = None
        self.closed = False

    def shutdown(self) -> None:
        """Cancel queued encoder work, wake its waiters and stop the encoder thread."""
        self.closed = True
        if self.conditioning_task is not None:
            self.conditioning_task.cancel()
        for pending in self.conditioning_pending.values():
            pending.cancel()
        self.conditioning_pending.clear()
        while not self.conditioning_queue.empty():
            self.conditioning_queue.get_nowait()
        self.encoder_executor.shutdown(wait=False, cancel_futures=True)

    def prepare_voices(self, references: list[tuple[np.ndarray, int]]) -> list[VoiceConditioning | ValueError]:
        """Run one encoder batch on the adapter's encoder thread."""
        return self.conditioner.prepare_batch(references)

    async def drain_conditioning(self) -> None:
        """Own encoder execution; every queued reference goes through the encoders together."""
        while not self.conditioning_queue.empty():
            batch = []
            while not self.conditioning_queue.empty():
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
                    self.ctx.server._speaker_cache.put(self.cache_key(key), dict(vars(voice)))
                    self.conditioning_pending.pop(key).set_result(voice)

    def cache_key(self, artifact_key: str) -> tuple[str, str, int]:
        """A reference's key in the server's speaker cache, which ``speech_cache.speaker_max_bytes`` bounds."""
        return self.ctx.server._speaker_cache.make_cache_key(f"ref:{artifact_key}", model_type=self.name)

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
            cached = self.ctx.server._speaker_cache.get(self.cache_key(key))
            if cached is not None:
                return VoiceConditioning(**cached)
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
        """Reject requests the checkpoint cannot honor."""
        checkpoint = f"Chatterbox {self.variant.capitalize()}"
        error = self.ctx.server._apply_uploaded_speaker(request)
        if error:
            return error
        if not request.input.strip():
            return "Input text cannot be empty"
        error = self._validate_max_new_tokens(request)
        if error:
            return error
        if request.language is not None and request.language.lower() not in {"auto", "en", "english"}:
            return f"{checkpoint} supports English only"
        if request.instructions:
            return f"{checkpoint} does not support instructions"
        if request.task_type == "VoiceDesign":
            return f"{checkpoint} does not support VoiceDesign"
        if request.x_vector_only_mode:
            return f"{checkpoint} requires full reference conditioning"
        if (CONTROLS - self.controls).intersection(request.extra_params or {}):
            return "Exaggeration and CFG controls require Chatterbox Original"
        if request.ref_audio_2 is not None or request.speaker_embedding is not None:
            return f"{checkpoint} accepts one reference recording"
        if request.ref_audio is None:
            if request.voice not in (None, "", "default"):
                return f"{checkpoint} supports voice='default' or an uploaded reference voice"
        else:
            reference = request.ref_audio
            if isinstance(reference, list):
                if len(reference) != 1:
                    return f"{checkpoint} accepts one reference recording"
                reference = reference[0]
            error = self.ctx.server._validate_ref_audio_format(reference)
            if error:
                return error
        return None

    def prepare_prompt(self, text: str, voice: VoiceConditioning | None, controls: dict[str, float]) -> dict:
        """Build the conditioned prompt on a worker thread, outside the API event loop."""
        if voice is None:
            voice = self.builtin_voice
        return build_prompt(tokenize_text(text, self.tokenizer, self.variant), voice, self.config, **controls)

    async def build(
        self, request: OpenAICreateSpeechRequest, sampling_params_list: list, has_inline_ref_audio: bool
    ) -> PreparedRequest:
        """Resolve reference audio and build the engine's conditioned token prompt."""
        voice = None
        if request.ref_audio is not None:
            source = request.ref_audio[0] if isinstance(request.ref_audio, list) else request.ref_audio
            voice = await self.resolve_voice(source)
        requested = request.extra_params or {}
        controls = {name: requested.get(name, default) for name, default in self.default_controls.items()}
        prompt = await self.build_async(request.input, voice, controls)
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
    sequences_per_request = 2
    controls = CONTROLS
