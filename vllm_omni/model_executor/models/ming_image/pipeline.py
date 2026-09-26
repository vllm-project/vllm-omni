# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Two-stage Ming-Image topology."""

from vllm_omni.config.stage_config import PipelineConfig, StageExecutionType, StagePipelineConfig

_PROC = "vllm_omni.model_executor.stage_input_processors.ming_image"
_CHECKPOINT = "vllm_omni.model_executor.models.ming_image.checkpoint"

MING_IMAGE_PIPELINE = PipelineConfig(
    model_type="ming_image",
    default_deploy_config_name="ming_image.yaml",
    model_arch="MingImageForConditionalGeneration",
    hf_architectures=("MingImageForConditionalGeneration",),
    diffusers_class_name="MingImageDiffusionPipeline",
    diffusers_class_aliases=("MingImageLayeredDiffusionPipeline",),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="mllm",
            model_arch="MingImageForConditionalGeneration",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=False,
            owns_tokenizer=True,
            requires_multimodal_data=True,
            engine_output_type="latent",
            model_subdir="mllm",
            tokenizer_subdir="mllm",
            sampling_constraints={"detokenize": False},
            model_path_resolver=f"{_CHECKPOINT}.resolve_ming_image_model_root",
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="dit",
            model_arch="MingImageDiffusionPipeline",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(0,),
            final_output=True,
            final_output_type="image",
            custom_process_input_func=f"{_PROC}.thinker2image",
        ),
    ),
)

__all__ = ["MING_IMAGE_PIPELINE"]
