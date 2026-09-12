# SPDX-License-Identifier: Apache-2.0
"""Cosmos3-Nano-Sim-Bimanual single-stage autoregressive diffusion topology."""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

COSMOS3_NANO_SIM_BIMANUAL_PIPELINE = PipelineConfig(
    model_type="cosmos3_nano_sim_bimanual",
    default_deploy_config_name="cosmos3_nano_sim_bimanual.yaml",
    model_arch="Cosmos3NanoSimBimanualPipeline",
    diffusers_class_name="Cosmos3NanoSimBimanualPipeline",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="diffusion",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=True,
            final_output_type="video",
            model_arch="Cosmos3NanoSimBimanualPipeline",
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
