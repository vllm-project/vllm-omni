# SPDX-License-Identifier: Apache-2.0
"""Cosmos-Dreams single-stage autoregressive diffusion topology."""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

COSMOS_DREAMS_PIPELINE = PipelineConfig(
    model_type="cosmos_dreams",
    default_deploy_config_name="cosmos_dreams.yaml",
    model_arch="CosmosDreamsPipeline",
    diffusers_class_name="CosmosDreamsPipeline",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="diffusion",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=True,
            final_output_type="video",
            model_arch="CosmosDreamsPipeline",
        ),
    ),
)

COSMOS_DREAMS_TRANSFER_PIPELINE = PipelineConfig(
    model_type="cosmos_dreams_transfer",
    default_deploy_config_name="cosmos_dreams_transfer.yaml",
    model_arch="CosmosDreamsTransferPipeline",
    diffusers_class_name="CosmosDreamsTransferPipeline",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="diffusion",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=True,
            final_output_type="video",
            model_arch="CosmosDreamsTransferPipeline",
        ),
    ),
)
