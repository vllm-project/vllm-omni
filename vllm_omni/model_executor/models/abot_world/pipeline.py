# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""ABot-World single-stage diffusion topology.

Serving resolves a deploy config through this registration, so the stepwise
stage cannot be launched with ``--deploy-config`` until the pipeline name is
registered here.

Deliberately no ``default_deploy_config_name``: model-type inference matches
this checkpoint by name, so a default would move every ``Omni`` and
``AsyncOmni`` caller that passes no deploy config onto the stepwise serving
topology. Stepwise serving stays opt-in through
``--deploy-config vllm_omni/deploy/abot_world_stepwise.yaml``.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

ABOT_WORLD_PIPELINE = PipelineConfig(
    model_type="abot_world",
    model_arch="ABotWorldCausalPipeline",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="diffusion",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=True,
            final_output_type="video",
            model_arch="ABotWorldCausalPipeline",
        ),
    ),
)
