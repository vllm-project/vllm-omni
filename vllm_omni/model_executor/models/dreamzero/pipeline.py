# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""DreamZero single-stage diffusion topology."""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)
from vllm_omni.serving_capabilities import (
    declare_capabilities,
)

DREAMZERO_PIPELINE = PipelineConfig(
    serving_capabilities=declare_capabilities(
        unsupported_reason="outside_standard_operations",
        sources=("vllm_omni/model_executor/models/dreamzero/pipeline.py",),
    ),
    model_type="dreamzero",
    default_deploy_config_name="dreamzero.yaml",
    model_arch="DreamZeroPipeline",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="diffusion",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=True,
            final_output_type="image",
            model_arch="DreamZeroPipeline",
        ),
    ),
)
