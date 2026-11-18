# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qwen3-ASR topology, with an optional injected forced-aligner stage."""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

QWEN3_ASR_PIPELINE = PipelineConfig(
    model_type="qwen3_asr",
    model_arch="Qwen3ASRForConditionalGeneration",
    default_deploy_config_name="qwen3_asr.yaml",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="asr",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            requires_multimodal_data=True,
            engine_output_type="text",
            model_arch="Qwen3ASRForConditionalGeneration",
            sampling_constraints={"detokenize": True},
        ),
    ),
)
