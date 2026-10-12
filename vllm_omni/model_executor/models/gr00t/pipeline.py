# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GR00T N1.7 single-stage policy topology."""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

GR00T_N1D7_PIPELINE = PipelineConfig(
    model_type="Gr00tN1d7",
    default_deploy_config_name="Gr00tN1d7.yaml",
    model_arch="Gr00tN1d7Pipeline",
    hf_architectures=("Gr00tN1d7",),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="diffusion",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=True,
            final_output_type="actions",
            model_arch="Gr00tN1d7Pipeline",
        ),
    ),
)
