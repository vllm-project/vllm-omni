# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SoulX-Singer pipeline topologies."""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

SOULXSINGER_SVS_PIPELINE = PipelineConfig(
    model_type="soulxsinger_svs",
    default_deploy_config_name="soulxsinger_svs.yaml",
    model_arch="SoulXSingerPipeline",
    hf_architectures=("SoulXSingerPipeline",),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="dit",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=True,
            final_output_type="audio",
            model_arch="SoulXSingerPipeline",
        ),
    ),
)

SOULXSINGER_SVC_PIPELINE = PipelineConfig(
    model_type="soulxsinger_svc",
    default_deploy_config_name="soulxsinger_svc.yaml",
    model_arch="SoulXSingerSVCPipeline",
    hf_architectures=("SoulXSingerSVCPipeline",),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="dit",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=True,
            final_output_type="audio",
            model_arch="SoulXSingerSVCPipeline",
        ),
    ),
)
