# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""minWM Action2V single-stage diffusion topology.

Registered so ``--deploy-config vllm_omni/deploy/minwm_action2v_stepwise.yaml``
can launch the streaming stage; like Self-Forcing there is no default deploy
config, so plain ``Omni`` callers keep request mode.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

MINWM_PIPELINE = PipelineConfig(
    model_type="minwm",
    model_arch="MinWMAction2VPipeline",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="diffusion",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=True,
            final_output_type="video",
            model_arch="MinWMAction2VPipeline",
        ),
    ),
)
