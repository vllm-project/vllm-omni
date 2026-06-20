# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CSM-1B 2-stage pipeline topology (frozen).

Stage 0 (LLM_AR): the Llama-style backbone runs under vLLM PagedAttention; cb0 is
sampled per 80 ms frame; the 31-step depth decoder runs INLINE inside Stage 0's
``forward()`` to produce cb1..cb31; the finished 32-code frame is emitted forward
as a latent payload (``codes.audio``). The backbone<->depth feedback (the next
frame's Sigma-embedding) is resolved entirely inside Stage 0 via a per-request
cache + the ``preprocess`` re-inject hook -- it NEVER crosses the stage boundary.

Stage 1 (LLM_GENERATION): the Mimi vocoder decodes the 32-code frames into PCM
audio (code2wav), stateless per frame.

This replaces the single-stage scaffold (CsmForGeneration / ``inference_stream``)
whose private per-frame ``positions`` loop corrupted the paged KV heap and caused
the deferred illegal-memory-access (A3 design §1). The 2-stage shape mirrors the
two shipping precedents: qwen3_tts (talker + code2wav) and mimo_audio (the
multi-codebook RVQ feedback twin).
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.csm"

CSM_PIPELINE = PipelineConfig(
    model_type="csm",
    # Pipeline-level default arch = Stage-0 backbone; Stage 1 overrides per-stage.
    model_arch="CsmBackboneForConditionalGeneration",
    stages=(
        StagePipelineConfig(
            # STAGE 0 -- backbone AR + inline 31-step depth + Sigma feedback.
            stage_id=0,
            model_stage="csm",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=True,
            engine_output_type="latent",  # emits 32-code frames forward, not audio
            # Producing stage declares the async-chunk next-stage processor
            # (mirrors qwen3_tts pipeline.py:28).
            async_chunk_process_next_stage_input_func=f"{_PROC}.backbone2mimi_async_chunk",
            sampling_constraints={
                "detokenize": False,
                # Real frame-level EOS is "cb0..cb30 all-zero" forced in
                # compute_logits (token id 0); this scheduler stop matches it.
                "stop_token_ids": [0],
            },
        ),
        StagePipelineConfig(
            # STAGE 1 -- Mimi vocoder (code2wav).
            stage_id=1,
            model_stage="mimi",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            model_arch="CsmMimiVocoder",  # per-stage arch override
            # Consuming stage declares the sync (non-async) input processor
            # (mirrors qwen3_tts pipeline.py:43).
            sync_process_input_func=f"{_PROC}.backbone2mimi",
            sampling_constraints={"detokenize": True},
        ),
    ),
)
