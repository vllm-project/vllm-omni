# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""WaveServe Wan single-stage diffusion topology.

The diffusion implementation lives under
``vllm_omni.diffusion.models.waveserve_wan``; this module only registers the
Omni pipeline name and default deploy recipe, matching DreamZero / LingBot.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

WAVESERVE_WAN_PIPELINE = PipelineConfig(
    model_type="waveserve_wan",
    default_deploy_config_name="waveserve_wan.yaml",
    model_arch="WaveServeWanPipeline",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="diffusion",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=True,
            final_output_type="video",
            model_arch="WaveServeWanPipeline",
        ),
    ),
)
