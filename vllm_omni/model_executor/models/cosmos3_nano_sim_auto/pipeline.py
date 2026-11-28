# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Cosmos3-Nano-Sim-Auto single-stage autoregressive diffusion topology."""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

COSMOS3_NANO_SIM_AUTO_PIPELINE = PipelineConfig(
    model_type="cosmos3_nano_sim_auto",
    default_deploy_config_name="cosmos3_nano_sim_auto.yaml",
    model_arch="Cosmos3NanoSimAutoPipeline",
    diffusers_class_name="Cosmos3NanoSimAutoPipeline",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="diffusion",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=True,
            final_output_type="video",
            model_arch="Cosmos3NanoSimAutoPipeline",
        ),
    ),
)
