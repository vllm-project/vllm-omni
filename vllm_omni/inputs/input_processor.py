# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Preserve processor-owned metadata across upstream request construction."""

import time
from collections.abc import Mapping
from typing import Any, cast

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
from vllm_omni.inputs.processed_media import ProcessedMediaProvenance


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
            # Keep invalid sampling/adapter requests from doing media work.
            self._validate_params(params, supported_tasks)
            self._validate_lora(lora_request)
            if arrival_time is None:
                arrival_time = time.time()
            parsed = parse_model_prompt(self.model_config, prompt)
            tok_params = self.renderer.default_cmpl_tok_params.with_kwargs(**(tokenization_kwargs or {}))
            (engine_input,) = self.renderer.render_cmpl([parsed], tok_params)

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
        provenance = decoder_input.get("_omni_processed_media")
        # JSON dictionaries and additional_information cannot mint this type.
        if isinstance(provenance, ProcessedMediaProvenance):
            provenance.validate(request.prompt_token_ids, request.mm_features)
            return OmniEngineCoreRequest.from_request(request, processed_media_provenance=provenance)
        return request
