# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Self-Forcing single-stage diffusion topology.

Registered so ``--deploy-config vllm_omni/deploy/self_forcing_stepwise.yaml``
can launch the streaming stage. Like LingBot-World, there is deliberately no
``default_deploy_config_name``: plain ``Omni`` callers keep request mode and
streaming stays opt-in.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

SELF_FORCING_PIPELINE = PipelineConfig(
    model_type="self_forcing",
    model_arch="SelfForcingPipeline",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="diffusion",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=True,
            final_output_type="video",
            model_arch="SelfForcingPipeline",
        ),
    ),
)
