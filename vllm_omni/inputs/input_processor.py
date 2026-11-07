# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Preserve model-declared request context through upstream input processing."""

import time
from collections.abc import Mapping
from typing import Any, cast

from vllm.exceptions import VLLMValidationError
from vllm.inputs import EngineInput, PromptType, split_enc_dec_input
from vllm.lora.request import LoRARequest
from vllm.pooling_params import PoolingParams
from vllm.renderers.inputs.preprocess import parse_model_prompt
from vllm.sampling_params import SamplingParams
from vllm.tasks import SupportedTask
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.kv_hints import KvHintsEnvelope

from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.inputs.mm_rng import MM_RNG_CONTEXT_KEY, MMRNGContext, contextualize_mm_features


class OmniInputProcessor(InputProcessor):
    def process_inputs(
        self,
        request_id: str,
        prompt: PromptType | EngineInput,
        params: SamplingParams | PoolingParams,
        supported_tasks: tuple[SupportedTask, ...],
        arrival_time: float | None = None,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
        resumable: bool = False,
        session_id: str | None = None,
        kv_hints: KvHintsEnvelope | None = None,
    ) -> EngineCoreRequest:
        if isinstance(prompt, dict) and "type" in prompt:
            engine_input = cast(EngineInput, prompt)
        else:
            # Rendering mutates the sender cache. Preserve upstream's early
            # validation order so rejected raw requests cannot leave unsent
            # entries behind. super() still owns the complete input validation.
            self._validate_params(params, supported_tasks)
            self._validate_lora(lora_request)
            parallel_config = self.vllm_config.parallel_config
            num_ranks = (
                parallel_config.data_parallel_size_local
                if parallel_config.local_engines_only
                else parallel_config.data_parallel_size
            )
            if data_parallel_rank is not None and not (0 <= data_parallel_rank < num_ranks):
                raise VLLMValidationError(f"data_parallel_rank {data_parallel_rank} is out of range [0, {num_ranks}).")

            # Render once, using the same parser and renderer as upstream. Keep
            # the result so request metadata survives process_inputs' projection.
            if arrival_time is None:
                arrival_time = time.time()
            parsed_prompt = parse_model_prompt(self.model_config, prompt)
            tok_params = self.renderer.default_cmpl_tok_params.with_kwargs(**(tokenization_kwargs or {}))
            (engine_input,) = self.renderer.render_cmpl([parsed_prompt], tok_params)

        request = super().process_inputs(
            request_id=request_id,
            prompt=engine_input,
            params=params,
            supported_tasks=supported_tasks,
            arrival_time=arrival_time,
            lora_request=lora_request,
            tokenization_kwargs=tokenization_kwargs,
            trace_headers=trace_headers,
            priority=priority,
            data_parallel_rank=data_parallel_rank,
            resumable=resumable,
            session_id=session_id,
            kv_hints=kv_hints,
        )
        _, decoder_input = split_enc_dec_input(engine_input)
        context = decoder_input.get(MM_RNG_CONTEXT_KEY)
        if context is None:
            if request.mm_features:
                seed_field = getattr(self.renderer.get_mm_processor(), "mm_rng_seed_field", None)
                if seed_field and any(f.data is not None and seed_field in f.data for f in request.mm_features):
                    raise ValueError("Seeded multimodal inputs require their processor-produced mm_rng_context")
            return request
        if not isinstance(context, MMRNGContext):
            raise TypeError("mm_rng_context must be a model-produced MMRNGContext")
        request = OmniEngineCoreRequest.from_request(request, mm_rng_context=context)
        request.mm_features = contextualize_mm_features(request.mm_features or [], context)
        return request
