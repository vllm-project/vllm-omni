# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""TaoMate-H3 single-stage streaming topology.

Streaming serving is opt-in through
``--deploy-config vllm_omni/deploy/taomate_h3_usp4_realtime.yaml``. There is
deliberately no ``default_deploy_config_name``: the checkpoint directory is the
MiniMax-H3 snapshot, whose model-type inference must keep resolving to the
regular MiniMax-H3 pipeline unless a deploy config names TaoMate-H3.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)
from vllm_omni.serving_capabilities import (
    TEXT_TO_VIDEO,
    declare_capabilities,
)

TAOMATE_H3_PIPELINE = PipelineConfig(
    serving_capabilities=declare_capabilities(
        TEXT_TO_VIDEO,
        sources=("vllm_omni/model_executor/models/taomate_h3/pipeline.py",),
    ),
    model_type="taomate_h3",
    model_arch="TaoMateH3Pipeline",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="diffusion",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=True,
            final_output_type="video",
            model_arch="TaoMateH3Pipeline",
        ),
    ),
)
