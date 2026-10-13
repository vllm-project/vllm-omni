# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""SolarWM-H3 native diffusion stage, with an optional AR deployment."""

from vllm_omni.config.stage_config import PipelineConfig, StageExecutionType, StagePipelineConfig

SOLARWM_H3_PIPELINE = PipelineConfig(
    model_type="solarwm_h3",
    model_arch="SolarWMH3Pipeline",
    diffusers_class_name="SolarWMH3Pipeline",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="diffusion",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=True,
            final_output_type="video",
            model_arch="SolarWMH3Pipeline",
        ),
    ),
)
