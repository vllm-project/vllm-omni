# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-Image pipeline topologies (frozen).
Two-stage (default):
  Stage 0: AR — multimodal understanding + token_ids generation
  Stage 1: DiT     — diffusion image generation
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

GLM_IMAGE_PIPELINE = PipelineConfig(
    model_type="glm_image",
    model_arch="GlmImageForConditionalGeneration",
    hf_architectures=("GlmImageForConditionalGeneration",),
    diffusers_class_name="GlmImagePipeline",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="ar",
            execution_type=StageExecutionType.LLM_AR,
            requires_multimodal_data=True,
            input_sources=(),
            final_output=False,
            owns_tokenizer=True,
            model_arch="GlmImageForConditionalGeneration",
            engine_output_type="token_ids",
            model_subdir="vision_language_encoder",
            tokenizer_subdir="processor",
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="dit",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(0,),
            requires_multimodal_data=True,
            final_output=True,
            final_output_type="image",
            model_arch="GlmImagePipeline",
            custom_process_input_func="vllm_omni.model_executor.stage_input_processors.glm_image.ar2diffusion",
            omni_kv_config={"need_recv_cache": False},
        ),
    ),
)
