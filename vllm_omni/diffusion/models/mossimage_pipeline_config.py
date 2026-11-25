# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MOSS-Image pipeline topology and default deployment."""

from vllm_omni.config.stage_config import PipelineConfig, StageExecutionType, StagePipelineConfig
from vllm_omni.diffusion.model_metadata import get_diffusion_model_metadata

MOSS_IMAGE_PIPELINE = PipelineConfig(
    serving_capabilities=get_diffusion_model_metadata("MOSSImagePipeline").serving_capabilities,
    model_type="moss_image",
    model_arch="MOSSImagePipeline",
    diffusers_class_name="MOSSImagePipeline",
    default_deploy_config_name="moss_image.yaml",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="diffusion",
            execution_type=StageExecutionType.DIFFUSION,
            final_output=True,
            final_output_type="image",
            model_arch="MOSSImagePipeline",
        ),
    ),
)
