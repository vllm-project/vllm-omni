# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Cosmos3-Nano-Sim-Bimanual single-stage autoregressive diffusion topology."""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)
from vllm_omni.diffusion.model_metadata import get_diffusion_model_metadata

COSMOS3_NANO_SIM_BIMANUAL_PIPELINE = PipelineConfig(
    serving_capabilities=get_diffusion_model_metadata("Cosmos3NanoSimBimanualPipeline").serving_capabilities,
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
