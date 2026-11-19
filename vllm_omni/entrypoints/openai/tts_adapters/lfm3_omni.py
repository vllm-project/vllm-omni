# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""LFM3-Omni speech prompts for the shared OpenAI speech endpoint."""

import math

from vllm.utils.async_utils import make_async

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import (
    ARTTSAdapter,
    PreparedRequest,
    apply_max_new_tokens,
    speech_inputs,
)
from vllm_omni.request_capabilities import ParameterConstraint


@register_tts_adapter
class Lfm3OmniTTSAdapter(ARTTSAdapter):
    request_constraints = speech_inputs("voice", "instructions", unsupported=("reference_audio",)).model_copy(
        update={"parameters": {"voice": ParameterConstraint(support="supported", type="string", enum=("default",))}}
    )
    name = "lfm3_omni"
    stage_keys = frozenset({"lfm3_temporal"})
    model_archs = frozenset({"Lfm3OmniForConditionalGeneration", "Lfm3AudioForConditionalGeneration"})

    def __init__(self, ctx):
        super().__init__(ctx)
        self._tokenizer = None
        self._prepare_async = make_async(self._prepare, executor=getattr(ctx.server, "_tts_executor", None))

    @classmethod
    def stage_serves_speech(cls, model_stage, all_stage_keys):
        return "lfm3_code2wav" in all_stage_keys

    def validate(self, request):
        if not request.input or not request.input.strip():
            return "Input text cannot be empty"
        if request.ref_audio is not None:
            return "LFM3-Omni speech serving does not support reference-audio voice cloning"
        if request.voice not in (None, "default"):
            return "LFM3-Omni has no named voice table; use instructions to describe the voice"
        extra = request.extra_params or {}
        for name, minimum in (("audio_temperature", 0), ("audio_repetition_penalty", 0)):
            value = extra.get(name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < minimum
                or (name == "audio_repetition_penalty" and value == 0)
            ):
                return f"Invalid {name}"
        for name, minimum in (("audio_top_k", 0), ("audio_repetition_window", 1)):
            value = extra.get(name)
            if value is not None and (type(value) is not int or value < minimum):
                return f"Invalid {name}"
        if "audio_repetition_penalty" in extra and extra["audio_repetition_penalty"] is None:
            return "audio_repetition_penalty must be positive"
        if "audio_repetition_window" in extra and extra["audio_repetition_window"] is None:
            return "audio_repetition_window must be positive"
        return self._validate_max_new_tokens(request)

    def _prepare(self, request):
        from vllm.tokenizers import cached_tokenizer_from_config

        if self._tokenizer is None:
            engine = self.ctx.engine_client
            if engine is None:
                raise RuntimeError("LFM3-Omni speech preparation requires an engine client")
            self._tokenizer = cached_tokenizer_from_config(engine.model_config)
        system = request.instructions or "Perform TTS in American English using a female voice."
        prompt = self._tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": request.input}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = self._tokenizer.encode(prompt + "<|audio_start|>", add_special_tokens=False)
        return {"prompt_token_ids": prompt_ids}

    async def build(self, request, sampling_params_list, has_inline_ref_audio):
        return PreparedRequest(prompt=await self._prepare_async(request), tts_params={}, model_type=self.name)

    def apply_sampling_overrides(self, sampling_params_list, request, prompt=None, request_id=None):
        return apply_max_new_tokens(sampling_params_list, request)
