# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Lightweight pipeline topology for Motubrain 2.0."""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

MOTUBRAIN2_PIPELINE = PipelineConfig(
    model_type="motubrain2",
    model_arch="Motubrain2Pipeline",
    diffusers_class_name="Motubrain2Pipeline",
    default_deploy_config_name="motubrain2.yaml",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="diffusion",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=True,
            final_output_type="action",
            model_arch="Motubrain2Pipeline",
        ),
    ),
)
