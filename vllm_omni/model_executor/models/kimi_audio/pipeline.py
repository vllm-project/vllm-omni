# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi Audio pipeline topology.

Stage 0: Fused LLM with bifurcation — text + audio logits.
Stage 1: Flow-matching detokenizer + vocoder → waveform.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.kimi_audio"

KIMI_AUDIO_PIPELINE = PipelineConfig(
    model_type="kimi_audio",
    model_arch="KimiAudioForConditionalGeneration",
    hf_architectures=("MoonshotKimiaForCausalLM", "KimiAudioForConditionalGeneration"),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="fused_llm",
            model_arch="KimiAudioLLMForConditionalGeneration",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            engine_output_type="latent",
            async_chunk_process_next_stage_input_func=(f"{_PROC}.llm2detokenizer_async_chunk"),
            custom_process_next_stage_input_func=f"{_PROC}.llm2detokenizer_full_payload",
            sampling_constraints={
                "detokenize": True,
            },
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="audio_detokenizer",
            model_arch="KimiAudioDetokenizerForConditionalGeneration",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            input_modalities=("audio",),
            custom_process_input_func=f"{_PROC}.llm2detokenizer",
            sync_process_input_func=f"{_PROC}.llm2detokenizer_token_only",
            sampling_constraints={"detokenize": False},
        ),
    ),
)
