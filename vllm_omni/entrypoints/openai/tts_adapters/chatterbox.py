# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Chatterbox speech serving with the built-in voice or a reference recording."""

import math
import os
from threading import Lock

import numpy as np
import torch
from tokenizers import Tokenizer
from transformers import AutoTokenizer, PreTrainedTokenizerBase
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
    punc_norm,
)
from vllm_omni.transformers_utils.configs.chatterbox import ChatterboxConfig
from vllm_omni.transformers_utils.repo_utils import hf_api


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
        self.tokenizer: PreTrainedTokenizerBase | None = None
        self.builtin_voice: VoiceConditioning | None = None
        self.conditioner: VoiceConditioner | None = None
        # Publish fully initialized components before concurrent requests use them.
        self.initialization_lock = Lock()
        self.build_async = make_async(self.prepare_prompt, executor=ctx.server._tts_executor)

    def validate(self, request: OpenAICreateSpeechRequest) -> str | None:
        """Reject requests whose controls the Turbo checkpoint cannot honor."""
        error = self.ctx.server._apply_uploaded_speaker(request)
        if error:
            return error
        if not request.input.strip():
            return "Input text cannot be empty"
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
        """Load Turbo's explicitly selected tokenizer for its config-less checkpoint."""
        with self.initialization_lock:
            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(model, tokenizer_type="gpt2")
        return self.tokenizer.encode(punc_norm(text), add_special_tokens=False)

    def prepare_prompt(
        self, text: str, reference: tuple[np.ndarray, int] | None, exaggeration: float = 0.5, cfg_weight: float = 0.5
    ) -> dict:
        """Load and condition on a worker thread, outside the API event loop."""
        model = resolve_stage_model_path(self.ctx.engine_client)
        if model is None:
            raise ValueError("Chatterbox requires a stage model path")
        with self.initialization_lock:
            if reference is None:
                if self.builtin_voice is None:
                    self.builtin_voice = VoiceConditioning.from_builtin(model)
                voice = self.builtin_voice
            else:
                if self.conditioner is None:
                    self.conditioner = VoiceConditioner(model, self.config, torch.device("cpu"))
                conditioner = self.conditioner
        if reference is not None:
            wav, sample_rate = reference
            voice = conditioner.prepare(np.asarray(wav, dtype=np.float32), sample_rate)
        text_ids = self.encode_text(model, text)
        return build_prompt(text_ids, voice, self.config, exaggeration=exaggeration, cfg_weight=cfg_weight)

    async def build(
        self, request: OpenAICreateSpeechRequest, sampling_params_list: list, has_inline_ref_audio: bool
    ) -> PreparedRequest:
        """Resolve reference audio and build the engine's conditioned token prompt."""
        reference = None
        if request.ref_audio is not None:
            source = request.ref_audio[0] if isinstance(request.ref_audio, list) else request.ref_audio
            wav, sample_rate, _ = await self.ctx.server._resolve_ref_audio(source)
            reference = (wav, sample_rate)
        controls = request.extra_params or {}
        prompt = await self.build_async(
            request.input, reference, controls.get("exaggeration", 0.5), controls.get("cfg_weight", 0.5)
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

    def __init__(self, ctx: SpeechServingContext) -> None:
        super().__init__(ctx)
        self.original_tokenizer: Tokenizer | None = None

    def validate(self, request: OpenAICreateSpeechRequest) -> str | None:
        error = super().validate(request)
        if error:
            return error.replace("Chatterbox Turbo", "Chatterbox Original")
        for name in ("exaggeration", "cfg_weight"):
            value = (request.extra_params or {}).get(name, 0.5)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                return f"{name} must be a finite nonnegative number"
        return None

    def encode_text(self, model: str, text: str) -> list[int]:
        with self.initialization_lock:
            if self.original_tokenizer is None:
                model_dir = model
                if not os.path.isdir(model):
                    model_dir = hf_api().snapshot_download(model, allow_patterns=["tokenizer.json"])
                self.original_tokenizer = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        return self.original_tokenizer.encode(punc_norm(text, "original").replace(" ", "[SPACE]")).ids
