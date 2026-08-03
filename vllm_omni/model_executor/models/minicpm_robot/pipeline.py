# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniCPM-RobotManip single-stage VLA policy topology."""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

MINICPM_ROBOT_MANIP_PIPELINE = PipelineConfig(
    model_type="MiniCPMRobotManip",
    default_deploy_config_name="MiniCPMRobotManip.yaml",
    model_arch="MiniCPMRobotManipPipeline",
    hf_architectures=("MiniCPMV_VLA",),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="diffusion",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=True,
            final_output_type="actions",
            model_arch="MiniCPMRobotManipPipeline",
        ),
    ),
)
