# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Lightweight topology; importing the registry does not construct the model."""

from vllm_omni.config.stage_config import PipelineConfig, StageExecutionType, StagePipelineConfig

LINGBOT_VLA_V2_PIPELINE = PipelineConfig(
    model_type="lingbot_vla_v2",
    model_arch="LingBotVLAV2Pipeline",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="diffusion",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=True,
            final_output_type="action",
            model_arch="LingBotVLAV2Pipeline",
        ),
    ),
)
