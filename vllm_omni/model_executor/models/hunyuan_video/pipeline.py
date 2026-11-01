# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HunyuanVideo-1.5 pipeline topology."""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

HUNYUAN_VIDEO_15_PIPELINE = PipelineConfig(
    model_type="hunyuan_video_15",
    model_arch="HunyuanVideo15Pipeline",
    diffusers_class_name="HunyuanVideo15Pipeline",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="dit",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=True,
            final_output_type="video",
            model_arch="HunyuanVideo15Pipeline",
        ),
    ),
)
