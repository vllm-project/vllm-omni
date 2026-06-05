# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""π0 (Pi-Zero) single-stage diffusion topology.

π0 is a flow-matching VLA: one diffusion stage takes a robot observation and
emits a continuous action chunk. The actions ride in ``multimodal_output`` (the
``final_output_type`` is nominal, matching DreamZero's single-stage shape).
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

PI0_PIPELINE = PipelineConfig(
    model_type="pi0",
    model_arch="Pi0Pipeline",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="diffusion",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=True,
            final_output_type="image",
            model_arch="Pi0Pipeline",
        ),
    ),
)
