# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from vllm_omni.config.stage_config import PipelineConfig, StageExecutionType, StagePipelineConfig

PHYSIS_WAN_PIPELINE = PipelineConfig(
    model_type="physis_wan",
    default_deploy_config_name="physis_wan.yaml",
    model_arch="PhysisWanPipeline",
    diffusers_class_name="PhysisWanPipeline",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="diffusion",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=True,
            final_output_type="video",
            model_arch="PhysisWanPipeline",
        ),
    ),
)
