# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from vllm_omni.config.stage_config import PipelineConfig, StageExecutionType, StagePipelineConfig
from vllm_omni.serving_capabilities import (
    TEXT_TO_VIDEO,
    declare_capabilities,
)

PHYSIS_LTX_PIPELINE = PipelineConfig(
    serving_capabilities=declare_capabilities(
        TEXT_TO_VIDEO,
        sources=("vllm_omni/diffusion/models/physis/ltx/pipeline_config.py",),
    ),
    model_type="physis_ltx",
    default_deploy_config_name="physis_ltx.yaml",
    model_arch="PhysisLTXPipeline",
    diffusers_class_name="PhysisLTXPipeline",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="diffusion",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=True,
            final_output_type="video",
            model_arch="PhysisLTXPipeline",
        ),
    ),
)
