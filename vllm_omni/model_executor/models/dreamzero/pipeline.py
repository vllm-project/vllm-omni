# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DreamZero diffusion topologies.

Two registered topologies:

* ``DREAMZERO_PIPELINE`` (``model_type="dreamzero"``) — the original single
  monolithic diffusion stage. Unchanged; the default for compatibility.
* ``DREAMZERO_DISAGGREGATED_PIPELINE`` (``model_type="dreamzero_disaggregated"``)
  — the three-stage encode -> denoise -> decode topology. All three stages run
  as native ``StageExecutionType.DIFFUSION`` stages distinguished by
  ``model_stage``. The generic diffusion transition processor moves the typed
  ``StagePayload`` between them.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

#: Dotted path to the generic diffusion -> diffusion transition processor. One
#: model-agnostic adapter for every disaggregated diffusion model (RFC #4590 §6).
#: Re-exported from the processor module (its single source of truth) so this
#: topology and the function it names cannot drift.
from vllm_omni.model_executor.stage_input_processors.diffusion import (  # noqa: E402
    GENERIC_DIFFUSION_PROCESSOR,
)

DREAMZERO_PIPELINE = PipelineConfig(
    model_type="dreamzero",
    default_deploy_config_name="dreamzero.yaml",
    model_arch="DreamZeroPipeline",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="diffusion",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=True,
            final_output_type="image",
            model_arch="DreamZeroPipeline",
        ),
    ),
)

DREAMZERO_DISAGGREGATED_PIPELINE = PipelineConfig(
    model_type="dreamzero_disaggregated",
    model_arch="DreamZeroPipeline",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="encode",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=False,
            model_arch="DreamZeroPipeline",
            # Producer hook: pack the encode->denoise payload for stage 1.
            custom_process_next_stage_input_func=GENERIC_DIFFUSION_PROCESSOR,
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="denoise",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(0,),
            final_output=False,
            model_arch="DreamZeroPipeline",
            # Consumer hook: unpack the encode->denoise payload into the request.
            custom_process_input_func=GENERIC_DIFFUSION_PROCESSOR,
            # Producer hook: pack the denoise->decode payload for stage 2.
            custom_process_next_stage_input_func=GENERIC_DIFFUSION_PROCESSOR,
        ),
        StagePipelineConfig(
            stage_id=2,
            model_stage="decode",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(1,),
            final_output=True,
            final_output_type="image",
            model_arch="DreamZeroPipeline",
            # Consumer hook: unpack the denoise->decode payload into the request.
            custom_process_input_func=GENERIC_DIFFUSION_PROCESSOR,
        ),
    ),
)
