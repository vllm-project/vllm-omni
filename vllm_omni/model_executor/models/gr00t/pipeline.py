# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GR00T-N1.7 single-stage diffusion topology."""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

GR00T_PIPELINE = PipelineConfig(
    model_type="gr00t",
    model_arch="Gr00tN1d7Pipeline",
    # GR00T-N1.7-3B/config.json declares model_type="Gr00tN1d7" and
    # architectures=["Gr00tN1d7"], neither of which matches our registry
    # key "gr00t".  Declare the upstream architecture so the StageConfig
    # factory can resolve this pipeline via the hf_architectures
    # fallback (see stage_config.create_from_model lines 1095-1101).
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
