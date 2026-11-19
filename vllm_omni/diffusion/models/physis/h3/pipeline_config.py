# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from vllm_omni.config.stage_config import PipelineConfig, StageExecutionType, StagePipelineConfig
from vllm_omni.serving_capabilities import (
    TEXT_TO_VIDEO,
    declare_capabilities,
)

PHYSIS_H3_PIPELINE = PipelineConfig(
    serving_capabilities=declare_capabilities(
        TEXT_TO_VIDEO,
        sources=("vllm_omni/diffusion/models/physis/h3/pipeline_config.py",),
    ),
    model_type="physis_h3",
    default_deploy_config_name="physis_h3.yaml",
    model_arch="PhysisH3Pipeline",
    diffusers_class_name="PhysisH3Pipeline",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="diffusion",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=True,
            final_output_type="video",
            model_arch="PhysisH3Pipeline",
        ),
    ),
)
