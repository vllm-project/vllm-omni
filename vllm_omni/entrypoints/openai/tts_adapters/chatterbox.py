# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Chatterbox speech serving with the built-in voice or a reference recording."""

from threading import Lock

import numpy as np
import torch
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


@register_tts_adapter
class ChatterboxAdapter(ARTTSAdapter):
    """Build the same conditioned prompts used by offline inference."""

    name = "chatterbox"
    stage_keys = frozenset({"chatterbox_t3"})
    supported_output_sample_rates = frozenset({24000})

    def __init__(self, ctx: SpeechServingContext) -> None:
        super().__init__(ctx)
        self.config = ChatterboxConfig()
        self.tokenizer: PreTrainedTokenizerBase | None = None
        self.builtin_voice: VoiceConditioning | None = None
        self.conditioner: VoiceConditioner | None = None
        # Reference encoders and lazy initialization are shared by API requests.
        self.conditioning_lock = Lock()
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

    def prepare_prompt(self, text: str, reference: tuple[np.ndarray, int] | None) -> dict:
        """Load and condition on a worker thread, outside the API event loop."""
        model = resolve_stage_model_path(self.ctx.engine_client)
        if model is None:
            raise ValueError("Chatterbox requires a stage model path")
        with self.conditioning_lock:
            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(model)
            if reference is None:
                if self.builtin_voice is None:
                    self.builtin_voice = VoiceConditioning.from_builtin(model)
                voice = self.builtin_voice
            else:
                if self.conditioner is None:
                    self.conditioner = VoiceConditioner(model, self.config, torch.device("cpu"))
                wav, sample_rate = reference
                voice = self.conditioner.prepare(np.asarray(wav, dtype=np.float32), sample_rate)
            text_ids = self.tokenizer.encode(punc_norm(text), add_special_tokens=False)
            return build_prompt(text_ids, voice, self.config)

    async def build(
        self, request: OpenAICreateSpeechRequest, sampling_params_list: list, has_inline_ref_audio: bool
    ) -> PreparedRequest:
        """Resolve reference audio and build the engine's conditioned token prompt."""
        reference = None
        if request.ref_audio is not None:
            source = request.ref_audio[0] if isinstance(request.ref_audio, list) else request.ref_audio
            wav, sample_rate, _ = await self.ctx.server._resolve_ref_audio(source)
            reference = (wav, sample_rate)
        prompt = await self.build_async(request.input, reference)
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
