# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qwen-Image 2.1 pipeline topology and default deployment."""

from vllm_omni.config.stage_config import PipelineConfig, StageExecutionType, StagePipelineConfig

QWEN_IMAGE_21_PIPELINE = PipelineConfig(
    model_type="qwen_image_21",
    model_arch="QwenImage21Pipeline",
    diffusers_class_name="QwenImage21Pipeline",
    default_deploy_config_name="qwen_image_21.yaml",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="diffusion",
            execution_type=StageExecutionType.DIFFUSION,
            final_output=True,
            final_output_type="image",
            model_arch="QwenImage21Pipeline",
        ),
    ),
)
