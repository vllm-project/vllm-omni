# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Whisper ASR pipeline topology (frozen)."""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

WHISPER_PIPELINE = PipelineConfig(
    model_type="whisper",
    model_arch="WhisperForConditionalGeneration",
    hf_architectures=("WhisperForConditionalGeneration",),
    default_deploy_config_name="whisper.yaml",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="whisper",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            requires_multimodal_data=True,
            engine_output_type="text",
            model_arch="WhisperForConditionalGeneration",
            sampling_constraints={"detokenize": True},
        ),
    ),
)
