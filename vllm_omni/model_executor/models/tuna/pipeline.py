# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tuna/Tuna-2 pipeline topology."""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

TUNA_PIPELINE = PipelineConfig(
    model_type="tuna",
    hf_architectures=(
        "TunaPipeline",
        "Tuna2PixelPipeline",
        "Tuna2RPixelPipeline",
        "Tuna2PixelModel",
    ),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="diffusion",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=True,
            final_output_type="image",
            extras={
                "default_sampling_params": {
                    "seed": 42,
                    "num_inference_steps": 50,
                    "guidance_scale": 3.0,
                },
            },
        ),
    ),
)
