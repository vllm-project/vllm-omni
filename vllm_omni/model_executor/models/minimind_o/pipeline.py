# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.
"""MiniMind-O pipeline topology (frozen).

Stage 0: Thinker  — multimodal understanding + text generation
Stage 1: Talker   — text embeddings → 8-layer Mimi codec tokens
Stage 2: Code2Wav — Mimi codec tokens → 24kHz audio waveform
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.minimind_o"

MINIMIND_O_PIPELINE = PipelineConfig(
    model_type="minimind_o",
    model_arch="MiniMindOForConditionalGeneration",
    hf_architectures=(
        "MiniMindOmni",
        "MiniMindOForConditionalGeneration",
        "MiniMindOMoeForConditionalGeneration",
    ),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="thinker",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            requires_multimodal_data=True,
            engine_output_type="latent",
            sampling_constraints={"detokenize": True},
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="talker",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(0,),
            engine_output_type="latent",
            custom_process_input_func=f"{_PROC}.thinker2talker",
            sampling_constraints={
                "detokenize": True,
                "stop_token_ids": [2050],  # MIMI_CODEC_EOS_TOKEN_ID
            },
        ),
        StagePipelineConfig(
            stage_id=2,
            model_stage="code2wav",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(1,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            custom_process_input_func=f"{_PROC}.talker2code2wav",
            sampling_constraints={"detokenize": True},
        ),
    ),
)


# Single-stage thinker-only variant
MINIMIND_O_THINKER_ONLY_PIPELINE = PipelineConfig(
    model_type="minimind_o_thinker_only",
    model_arch="MiniMindOForConditionalGeneration",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="thinker",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            requires_multimodal_data=True,
            engine_output_type="latent",
            sampling_constraints={"detokenize": True},
        ),
    ),
)
