# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen2.5-Omni pipeline topology (frozen).

Stage 0: Thinker  — multimodal understanding + text generation
Stage 1: Talker   — text embeddings → speech tokens
Stage 2: Code2Wav — speech tokens → audio waveform
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.qwen2_5_omni"

QWEN2_5_OMNI_PIPELINE = PipelineConfig(
    model_type="qwen2_5_omni",
    model_arch="Qwen2_5OmniForConditionalGeneration",
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
                "stop_token_ids": [8294],
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
            sampling_constraints={"detokenize": True},
        ),
    ),
)


# Single-stage thinker-only variant for the abort test.
QWEN2_5_OMNI_THINKER_ONLY_PIPELINE = PipelineConfig(
    model_type="qwen2_5_omni_thinker_only",
    model_arch="Qwen2_5OmniForConditionalGeneration",
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
