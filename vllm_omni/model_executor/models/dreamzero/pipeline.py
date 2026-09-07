# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DreamZero diffusion pipeline topologies (frozen).

DreamZero runs as a single-stage diffusion model by default: the encoders, the
CausalWan DiT denoise loop and the action postprocess all live on one worker
(``DREAMZERO_PIPELINE``, selected by ``pipeline: dreamzero``). Nothing about
that default changes here.

``DREAMZERO_DISAGGREGATED_PIPELINE`` is the explicit opt-in three-stage variant:

    Request
      -> Stage 0: Encode      (tokenizer + UMT5 + CLIP + VAE encode)
      -> Stage 1: Denoise      (CausalWan DiT + AR-Diffusion paged KV, TP=4)
      -> Stage 2: Decode        (action postprocess -> response)
      -> Response

The stages are wired model-agnostically through ``DiffusionStageRole`` and the
generic cross-stage handoff
``vllm_omni.model_executor.stage_input_processors.diffusion_disagg.diffusion_stage_handoff``,
so DreamZero contributes only its role split and its payload schema -- never a
transport choice. Both edges declare the same single payload key; the payload's
``boundary`` field carries the semantic difference.

Runtime knobs (device placement, Stage-1 tensor parallelism, AR-Diffusion
backend, connector wiring) live in ``vllm_omni/deploy/dreamzero_disaggregated.yaml``;
select it with ``--deploy-config``. Switching to a different transport is a
deploy-config edit and touches no DreamZero source.
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

# One transport key for both stage edges; ``boundary`` inside the payload tells
# encode->denoise apart from denoise->decode.
_DREAMZERO_PAYLOAD_KEYS = (DREAMZERO_STAGE_PAYLOAD_KEY,)


# --- Single-stage (default) --------------------------------------------------
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


# --- Encode / Denoise / Decode disaggregation (opt-in) ----------------------
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
            # Surface the encode payload on custom_output so the connector (or,
            # on fallback, the orchestrator) forwards it downstream.
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
            custom_process_input_func=_DIFFUSION_HANDOFF,
        ),
    ),
)
