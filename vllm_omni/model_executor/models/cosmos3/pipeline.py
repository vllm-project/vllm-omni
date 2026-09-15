# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cosmos3 policy single-stage topology for online OpenPI serving.

Cosmos3 policy checkpoints (e.g. ``nvidia/Cosmos3-Nano-Policy-DROID``) are one
diffusion stage: robot observation -> action chunk. They are
registered in ``OMNI_PIPELINES`` because online OpenPI serving
(``/v1/realtime/robot/openpi``) requires a registered pipeline so the deploy
yaml's ``model_config.policy_server_config`` reaches the websocket handshake.

This pipeline declares neither ``hf_architectures`` nor
``diffusers_class_name``: policy checkpoints share their HF metadata
(``model_type=cosmos3_omni``, ``model_index.json`` ``_class_name=
Cosmos3OmniDiffusersPipeline``) with the T2I/video Cosmos3 checkpoints, which
must keep resolving through the default single-stage diffusion fallback.
Select this pipeline explicitly via a deploy yaml ``pipeline:`` key, e.g.
``vllm serve nvidia/Cosmos3-Nano-Policy-DROID --omni --deploy-config
/absolute/path/to/vllm-omni/vllm_omni/deploy/cosmos3_policy_droid.yaml``.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

COSMOS3_POLICY_PIPELINE = PipelineConfig(
    model_type="cosmos3_policy",
    model_arch="Cosmos3OmniDiffusersPipeline",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="diffusion",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=True,
            final_output_type="action",
            model_arch="Cosmos3OmniDiffusersPipeline",
        ),
    ),
)
