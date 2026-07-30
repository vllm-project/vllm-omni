# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wan2.2 TI2V pipeline topology."""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

WAN2_2_TI2V_PIPELINE = PipelineConfig(
    model_type="wan2_2_ti2v",
    model_arch="WanPipeline",
    diffusers_class_name="WanPipeline",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="dit",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=True,
            final_output_type="video",
            model_arch="WanPipeline",
        ),
    ),
)
