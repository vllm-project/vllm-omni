# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DreamZero single-stage and opt-in disaggregated topologies.

Stage 0 owns tokenizer, UMT5, CLIP and observation VAE encoding. Stage 1
owns CausalWan DiT and paged KV. Stage 2 is weightless action postprocess.

Stage roles and payload edges are defined here. Device placement, TP size
and connector selection live in deploy/dreamzero_disaggregated.yaml.
"""

from vllm_omni.config.stage_config import (
    DiffusionStageRole,
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)
from vllm_omni.diffusion.models.dreamzero.utils import DREAMZERO_STAGE_PAYLOAD_KEY

_DREAMZERO_MODEL_ARCH = "DreamZeroPipeline"
_DIFFUSION_HANDOFF = "vllm_omni.model_executor.stage_input_processors.diffusion_disagg.diffusion_stage_handoff"

_DREAMZERO_PAYLOAD_KEYS = (DREAMZERO_STAGE_PAYLOAD_KEY,)


DREAMZERO_PIPELINE = PipelineConfig(
    model_type="dreamzero",
    default_deploy_config_name="dreamzero.yaml",
    model_arch=_DREAMZERO_MODEL_ARCH,
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="diffusion",
            execution_type=StageExecutionType.DIFFUSION,
            stage_role=DiffusionStageRole.FULL,
            input_sources=(),
            final_output=True,
            final_output_type="image",
            model_arch=_DREAMZERO_MODEL_ARCH,
        ),
    ),
)


DREAMZERO_DISAGGREGATED_PIPELINE = PipelineConfig(
    model_type="dreamzero_disaggregated",
    default_deploy_config_name="dreamzero_disaggregated.yaml",
    model_arch=_DREAMZERO_MODEL_ARCH,
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="encode",
            execution_type=StageExecutionType.DIFFUSION,
            stage_role=DiffusionStageRole.ENCODE,
            stage_output_payload_keys=_DREAMZERO_PAYLOAD_KEYS,
            input_sources=(),
            final_output=False,
            model_arch=_DREAMZERO_MODEL_ARCH,
            coordinated_session_lifecycle=True,
            # Forward the non-final stage payload through custom_output.
            engine_output_type="custom",
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="denoise",
            execution_type=StageExecutionType.DIFFUSION,
            stage_role=DiffusionStageRole.DENOISE,
            stage_input_payload_keys=_DREAMZERO_PAYLOAD_KEYS,
            stage_output_payload_keys=_DREAMZERO_PAYLOAD_KEYS,
            input_sources=(0,),
            final_output=False,
            model_arch=_DREAMZERO_MODEL_ARCH,
            coordinated_session_lifecycle=True,
            engine_output_type="custom",
            custom_process_input_func=_DIFFUSION_HANDOFF,
        ),
        StagePipelineConfig(
            stage_id=2,
            model_stage="decode",
            execution_type=StageExecutionType.DIFFUSION,
            stage_role=DiffusionStageRole.DECODE,
            stage_input_payload_keys=_DREAMZERO_PAYLOAD_KEYS,
            input_sources=(1,),
            final_output=True,
            final_output_type="image",
            model_arch=_DREAMZERO_MODEL_ARCH,
            coordinated_session_lifecycle=True,
            custom_process_input_func=_DIFFUSION_HANDOFF,
        ),
    ),
)
