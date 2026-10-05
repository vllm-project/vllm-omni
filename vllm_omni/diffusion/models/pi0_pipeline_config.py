# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lightweight pipeline topology for π0 (Pi-Zero)."""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

# π0 is one diffusion stage (robot observation -> action chunk via flow
# matching). It is registered in ``OMNI_PIPELINES`` because π0 is online-served
# via ``vllm serve --deploy-config pi0.yaml``. Keep this topology outside the
# ``pi0`` package: importing a package submodule executes ``pi0.__init__``,
# which imports the runtime pipeline and ``diffusion.data``.
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
            final_output_type="action",
            model_arch="Pi0Pipeline",
        ),
    ),
)
