# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from vllm_omni.config.stage_config import PipelineConfig, StageExecutionType, StagePipelineConfig

from .assets import INVOKER_RENDER_PIPELINE_CLASS_NAME

INVOKER_RENDER_PIPELINE = PipelineConfig(
    model_type="invoker_render",
    model_arch=INVOKER_RENDER_PIPELINE_CLASS_NAME,
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="diffusion",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=True,
            final_output_type="video",
            model_arch=INVOKER_RENDER_PIPELINE_CLASS_NAME,
        ),
    ),
)
