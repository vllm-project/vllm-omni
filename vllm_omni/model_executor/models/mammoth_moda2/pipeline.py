# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MammothModa2 pipeline topology (frozen).

Stage 0: AR  — multimodal understanding + latent generation
Stage 1: DiT — latent → image

For text/image understanding tasks (text output only), use MAMMOTH_MODA2_AR_PIPELINE.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.mammoth_moda2"

MAMMOTH_MODA2_PIPELINE = PipelineConfig(
    model_type="mammoth_moda2",
    default_deploy_config_name="mammoth_moda2.yaml",
    model_arch="MammothModa2ForConditionalGeneration",
    hf_architectures=("Mammothmoda2Model", "MammothModa2ForConditionalGeneration"),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="ar",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=False,
            owns_tokenizer=True,
            requires_multimodal_data=True,
            engine_output_type="latent",
            # Use the existing full-payload connector path:
            # per-step slices remain device-resident in the request-end
            # accumulator, while ar2dit_full_payload performs the request-end
            # CPU/float32 adaptation before the connector's normal transport.
            custom_process_next_stage_input_func=f"{_PROC}.ar2dit_full_payload",
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="dit",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="image",
            owns_tokenizer=False,
            requires_multimodal_data=False,
            engine_output_type="image",
            sync_process_input_func=f"{_PROC}.ar2dit_token_only",
            requires_full_payload_input=True,
        ),
    ),
)

# Single-stage AR variant for understanding / summarization tasks
MAMMOTH_MODA2_AR_PIPELINE = PipelineConfig(
    model_type="mammoth_moda2_ar",
    default_deploy_config_name="mammoth_moda2_ar.yaml",
    model_arch="MammothModa2ForConditionalGeneration",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="ar",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            requires_multimodal_data=True,
            engine_output_type="text",
        ),
    ),
)
