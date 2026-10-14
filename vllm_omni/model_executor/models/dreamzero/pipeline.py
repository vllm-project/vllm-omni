# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DreamZero single-stage diffusion topology."""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

DREAMZERO_PIPELINE = PipelineConfig(
    model_type="dreamzero",
    default_deploy_config_name="dreamzero.yaml",
    model_arch="DreamZeroPipeline",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="diffusion",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=True,
            final_output_type="image",
            model_arch="DreamZeroPipeline",
        ),
    ),
)
