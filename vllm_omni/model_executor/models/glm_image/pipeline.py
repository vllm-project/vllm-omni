# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
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

_PROCESSOR = "vllm_omni.model_executor.stage_input_processors.glm_image"

GLM_IMAGE_PIPELINE = PipelineConfig(
    model_type="glm_image",
    default_deploy_config_name="glm_image.yaml",
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
            # Supplies target_h/target_w for prompts that did not come from the
            # serving layer; without them the AR stage never sees the grid
            # scaffold and decodes past EOS to the max_tokens ceiling.
            prompt_transform_func=f"{_PROCESSOR}.prepare_ar_prompt",
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
            custom_process_input_func=f"{_PROCESSOR}.ar2diffusion",
            omni_kv_config={"need_recv_cache": False},
        ),
    ),
)
