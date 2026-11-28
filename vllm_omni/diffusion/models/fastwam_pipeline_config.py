# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Lightweight pipeline topology for the FastWAM family (CAE-FastWAM by default)."""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

# FastWAM is one diffusion stage (robot observation -> action chunk). Kept outside
# the ``fastwam`` package so importing the topology does not import the runtime
# pipeline. ``vllm serve amd/CAE-FastWAM --omni`` picks the CAE deploy config;
# original FastWAM checkpoints use ``--deploy-config fastwam_libero.yaml``.
FASTWAM_PIPELINE = PipelineConfig(
    model_type="fastwam",
    model_arch="FastWAMPipeline",
    default_deploy_config_name="cae_fastwam_libero.yaml",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="diffusion",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=True,
            final_output_type="actions",
            model_arch="FastWAMPipeline",
        ),
    ),
)
