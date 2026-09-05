# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from transformers import AutoTokenizer, PreTrainedTokenizerBase
from vllm.inputs import TokensPrompt
from vllm.sampling_params import SamplingParams
from vllm.utils.async_utils import make_async

from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import (
    ARTTSAdapter,
    PreparedRequest,
    SpeechServingContext,
    apply_max_new_tokens,
)
from vllm_omni.model_executor.models.breeze_tts.prompt import build_breeze_prompt


@register_tts_adapter
class BreezeTTSAdapter(ARTTSAdapter):
    name = "breeze_tts"
    stage_keys = frozenset({"breeze_tts", "breeze_code2wav"})
    supported_output_sample_rates = frozenset({24000})

    def __init__(self, ctx: SpeechServingContext) -> None:
        super().__init__(ctx)
        self.tokenizer: PreTrainedTokenizerBase | None = None
        self._build_async = make_async(self._build_prompt, executor=ctx.server._tts_executor)

    def validate(self, request: OpenAICreateSpeechRequest) -> str | None:
        if not request.input.strip():
            return "Input text cannot be empty"
        if request.voice not in (None, "default"):
            return "Breeze voice design uses 'instructions'; named voices are not supported"
        if (
            request.ref_audio is not None
            or request.ref_audio_2 is not None
            or request.ref_text is not None
            or request.speaker_embedding is not None
        ):
            return "This Breeze integration supports reference-free voice design; reference audio is not supported"
        if request.task_type not in (None, "VoiceDesign"):
            return "Breeze supports task_type='VoiceDesign'"
        if request.x_vector_only_mode is not None:
            return "Breeze does not support x_vector_only_mode"
        if request.extra_params:
            return "Breeze does not support extra_params; configure sampling in the deployment YAML (CFG is fixed to 1)"
        return None

    def _build_prompt(self, request: OpenAICreateSpeechRequest, sampling: SamplingParams) -> TokensPrompt:
        if self.tokenizer is None:
            engine_client = self.ctx.engine_client
            if engine_client is None:
                raise RuntimeError("Breeze speech serving requires an engine client")
            model = engine_client.model_config.model
            self.tokenizer = AutoTokenizer.from_pretrained(model, config=engine_client.model_config.hf_config)
        return build_breeze_prompt(
            self.tokenizer,
            request.input,
            request.instructions or "",
            temperature=sampling.temperature,
            top_k=max(sampling.top_k, 0),
            top_p=sampling.top_p,
            repetition_penalty=sampling.repetition_penalty,
        )

    async def build(
        self,
        request: OpenAICreateSpeechRequest,
        sampling_params_list: list[SamplingParams],
        has_inline_ref_audio: bool,
    ) -> PreparedRequest:
        prompt = await self._build_async(request, sampling_params_list[0])
        return PreparedRequest(prompt=prompt, model_type=self.name)

    def apply_sampling_overrides(
        self,
        sampling_params_list: list[SamplingParams],
        request: OpenAICreateSpeechRequest,
        prompt: dict | None = None,
        request_id: str | None = None,
    ) -> list[SamplingParams]:
        return apply_max_new_tokens(sampling_params_list, request)
