# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""GR00T N1.7 single-stage policy topology."""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)
from vllm_omni.serving_capabilities import (
    declare_capabilities,
)

GR00T_N1D7_PIPELINE = PipelineConfig(
    serving_capabilities=declare_capabilities(
        unsupported_reason="outside_standard_operations",
        sources=("vllm_omni/model_executor/models/gr00t/pipeline.py",),
    ),
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
