# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CausVid Wan single-stage diffusion topology (Noisy KV).

Diffusion implementation: ``vllm_omni.diffusion.models.causvid_wan``.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

# Opt-in via deploy ``pipeline: causvid_wan`` / explicit model_arch. Do not alias
# ``WanPipeline`` — that Diffusers class is reserved for ``wan2_2_ti2v`` discovery
# when the checkpoint basename does not disambiguate.
CAUSVID_WAN_PIPELINE = PipelineConfig(
    model_type="causvid_wan",
    default_deploy_config_name="causvid_wan_self_forcing.yaml",
    model_arch="CausVidWanPipeline",
    diffusers_class_name="CausVidWanPipeline",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="diffusion",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=True,
            final_output_type="video",
            model_arch="CausVidWanPipeline",
        ),
    ),
)
