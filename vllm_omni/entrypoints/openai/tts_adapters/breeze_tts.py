# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
import math

import numpy as np
import pybase64 as base64
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from vllm.inputs import TokensPrompt
from vllm.logger import init_logger
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
from vllm_omni.model_executor.models.breeze_tts.prompt import DEFAULT_INSTRUCTION, build_breeze_prompt

logger = init_logger(__name__)


@register_tts_adapter
class BreezeTTSAdapter(ARTTSAdapter):
    name = "breeze_tts"
    stage_keys = frozenset({"breeze_tts", "breeze_code2wav"})
    supported_output_sample_rates = frozenset({24000})

    def __init__(self, ctx: SpeechServingContext) -> None:
        super().__init__(ctx)
        self.tokenizer: PreTrainedTokenizerBase | None = None
        self._build_async = make_async(self._build_prompt, executor=ctx.server._tts_executor)

    def _load_codec_frame_rate(self) -> float:
        # Breeze-TTS-2 uses 1,920 waveform samples per frame at 24 kHz.
        return 24000 / 1920

    async def warmup(self) -> None:
        """Prime text, depth and reference paths through the ordinary runner."""
        server = self.ctx.server
        logger.info("Warming Breeze text graphs, CFG branches, reference encoder and streaming codec")

        async def generate(text: str, request_id: str, **kwargs: object) -> bytes:
            request = OpenAICreateSpeechRequest(
                input=text, model=server.model_name, response_format="wav", seed=42, max_new_tokens=64, **kwargs
            )
            audio, _ = await server._generate_audio_bytes(request, request_id=request_id)
            return audio

        reference_text = "Welcome to this demonstration of clear and natural speech synthesis."
        reference = await generate(reference_text, "breeze-warmup-single")
        await generate(
            "Hello.",
            "breeze-warmup-cfg",
            extra_params={"guidance_scale": 4.0, "temperature": 0.7, "top_k": 100, "top_p": 0.8},
        )
        await generate(
            "Hello again.",
            "breeze-warmup-reference",
            ref_audio="data:audio/wav;base64," + base64.b64encode(reference).decode("ascii"),
            ref_text=reference_text,
            extra_params={"guidance_scale": 4.0},
        )
        await asyncio.gather(
            generate(reference_text, "breeze-warmup-batch-a"),
            generate("Thank you for listening to this clear and natural voice demonstration.", "breeze-warmup-batch-b"),
        )
        logger.info("Breeze speech warmup complete")

    def validate(self, request: OpenAICreateSpeechRequest) -> str | None:
        if not request.input.strip():
            return "Input text cannot be empty"
        if request.voice not in (None, "default"):
            return "Breeze uses instructions or ref_audio/ref_text; named voices are not supported"
        if request.ref_audio_2 is not None or request.speaker_embedding is not None:
            return "Breeze accepts one reference recording and its transcript"
        has_reference = request.ref_audio is not None
        if has_reference:
            if (
                not isinstance(request.ref_audio, str)
                or not isinstance(request.ref_text, str)
                or not request.ref_text.strip()
            ):
                return "Breeze voice cloning requires one ref_audio URL and a non-empty ref_text transcript"
            error = self.ctx.server._validate_ref_audio_format(request.ref_audio)
            if error:
                return error
        elif request.ref_text is not None:
            return "Breeze ref_text requires ref_audio"
        expected_task = "Base" if has_reference else "VoiceDesign"
        if request.task_type not in (None, expected_task):
            return f"Breeze requires task_type='{expected_task}' for this conditioning"
        if request.x_vector_only_mode is not None:
            return "Breeze does not support x_vector_only_mode"
        extra = request.extra_params or {}
        supported = {"guidance_scale", "temperature", "top_k", "top_p", "repetition_penalty"}
        if unknown := set(extra) - supported:
            return f"Unsupported Breeze parameters: {sorted(unknown)}"
        for key, value in extra.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                return f"Breeze {key} must be a finite number"
        if extra.get("guidance_scale", 1.0) <= 0 or extra.get("repetition_penalty", 1.1) <= 0:
            return "Breeze guidance_scale and repetition_penalty must be positive"
        if extra.get("temperature", 0.9) < 0 or not 0 < extra.get("top_p", 1.0) <= 1:
            return "Breeze requires temperature >= 0 and 0 < top_p <= 1"
        top_k = extra.get("top_k", 50)
        if not isinstance(top_k, int) or top_k < -1:
            return "Breeze top_k must be -1, 0, or a positive integer"
        return None

    def _build_prompt(
        self,
        request: OpenAICreateSpeechRequest,
        sampling: SamplingParams,
        reference: tuple[np.ndarray, int] | None,
    ) -> TokensPrompt:
        if self.tokenizer is None:
            engine_client = self.ctx.engine_client
            if engine_client is None:
                raise RuntimeError("Breeze speech serving requires an engine client")
            model = engine_client.model_config.model
            self.tokenizer = AutoTokenizer.from_pretrained(model, config=engine_client.model_config.hf_config)
        extra = request.extra_params or {}
        return build_breeze_prompt(
            self.tokenizer,
            request.input,
            DEFAULT_INSTRUCTION if request.instructions is None else request.instructions,
            ref_audio=reference,
            ref_text=request.ref_text,
            guidance_scale=float(extra.get("guidance_scale", 1.0)),
            temperature=float(extra.get("temperature", sampling.temperature)),
            top_k=max(int(extra.get("top_k", sampling.top_k)), 0),
            top_p=float(extra.get("top_p", sampling.top_p)),
            repetition_penalty=float(extra.get("repetition_penalty", sampling.repetition_penalty)),
        )

    async def build(
        self,
        request: OpenAICreateSpeechRequest,
        sampling_params_list: list[SamplingParams],
        has_inline_ref_audio: bool,
    ) -> PreparedRequest:
        reference = None
        if request.ref_audio is not None:
            waveform, sample_rate, _ = await self.ctx.server._resolve_ref_audio(request.ref_audio)
            reference = (np.asarray(waveform, dtype=np.float32), sample_rate)
        prompt = await self._build_async(request, sampling_params_list[0], reference)
        return PreparedRequest(prompt=prompt, model_type=self.name)

    def apply_sampling_overrides(
        self,
        sampling_params_list: list[SamplingParams],
        request: OpenAICreateSpeechRequest,
        prompt: dict | None = None,
        request_id: str | None = None,
    ) -> list[SamplingParams]:
        params = apply_max_new_tokens(sampling_params_list, request)
        params = list(params)
        params[0] = params[0].clone()
        if params[0].top_k == 0:
            params[0].top_k = -1
        extra = request.extra_params or {}
        if "repetition_penalty" in extra:
            params[0].repetition_penalty = extra["repetition_penalty"]
        if prompt is not None:
            engine_client = self.ctx.engine_client
            if engine_client is None:
                raise RuntimeError("Breeze speech serving requires an engine client")
            model_config = engine_client.model_config
            available = model_config.max_model_len - len(prompt["prompt_token_ids"])
            if available <= 0:
                raise ValueError("Breeze conditioning fills the model context; shorten the text or reference audio")
            # Both CFG branches must stop after the same number of frames,
            # including when the conditioned prompt reaches the context limit.
            params[0].max_tokens = min(params[0].max_tokens, available)
        return params
