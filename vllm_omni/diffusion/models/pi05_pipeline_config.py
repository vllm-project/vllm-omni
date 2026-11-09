# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Lightweight pipeline topology for π0.5."""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)
from vllm_omni.serving_capabilities import (
    declare_capabilities,
)

# π0.5 is one diffusion stage (robot observation -> action chunk via flow
# matching). It is registered in ``OMNI_PIPELINES`` because π0.5 is online-served
# via ``vllm serve --deploy-config pi05.yaml``. Keep this topology outside the
# ``pi05`` package: importing a package submodule executes ``pi05.__init__``,
# which imports the runtime pipeline and ``diffusion.data``.
PI05_PIPELINE = PipelineConfig(
    serving_capabilities=declare_capabilities(
        unsupported_reason="outside_standard_operations",
        sources=("vllm_omni/diffusion/models/pi05_pipeline_config.py",),
    ),
    model_type="pi05",
    model_arch="Pi05Pipeline",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="diffusion",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=True,
            final_output_type="action",
            model_arch="Pi05Pipeline",
        ),
    ),
)
