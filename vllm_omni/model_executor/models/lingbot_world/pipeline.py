# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""LingBot-World 2.0 single-stage diffusion topology.

Serving resolves a deploy config through this registration, so the stepwise
stage cannot be launched with ``--deploy-config`` until the pipeline name is
registered here.

Deliberately no ``default_deploy_config_name``: model-type inference matches
this checkpoint by name, so a default would silently retarget every ``Omni``
and ``AsyncOmni`` caller that passes no deploy config (the offline replay
example, the tick example) onto the stepwise serving topology. Stepwise
serving stays opt-in through
``--deploy-config vllm_omni/deploy/lingbot_world_v2_stepwise.yaml``.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

LINGBOT_WORLD_PIPELINE = PipelineConfig(
    model_type="lingbot_world",
    model_arch="LingBotWorldCausalDMDPipeline",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="diffusion",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=True,
            final_output_type="video",
            model_arch="LingBotWorldCausalDMDPipeline",
        ),
    ),
)
