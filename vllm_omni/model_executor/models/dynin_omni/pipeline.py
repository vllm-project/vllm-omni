# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dynin-Omni pipeline topology (frozen).

Stage 0: token2text  — multimodal understanding / text generation (comprehension)
Stage 1: token2image — stage-0 tokens → image latents
Stage 2: token2audio — stage-1 tokens → audio latents

All three stages run on the generation worker (``LLM_GENERATION``); the
inter-stage hand-off uses the worker-connector full-payload data plane
(``*_full_payload`` producers + ``token2text_to_token2image`` /
``token2image_to_token2audio`` consumers). Deploy knobs (devices, GPU memory,
batched tokens, connectors) live in ``vllm_omni/deploy/dynin_omni*.yaml``.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.dynin_omni"

DYNIN_OMNI_PIPELINE = PipelineConfig(
    model_type="dynin_omni",
    default_deploy_config_name="dynin_omni.yaml",
    model_arch="DyninOmniForConditionalGeneration",
    # Arch-fallback safety net: route by hf_config.architectures when the
    # auto-detected model_type does not match the registry key exactly.
    hf_architectures=("DyninOmniForConditionalGeneration",),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="token2text",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            engine_output_type="latent",
            custom_process_next_stage_input_func=f"{_PROC}.token2text_to_token2image_full_payload",
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="token2image",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="image",
            engine_output_type="latent",
            custom_process_input_func=f"{_PROC}.token2text_to_token2image",
            custom_process_next_stage_input_func=f"{_PROC}.token2image_to_token2audio_full_payload",
        ),
        StagePipelineConfig(
            stage_id=2,
            model_stage="token2audio",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(1,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="latent",
            custom_process_input_func=f"{_PROC}.token2image_to_token2audio",
        ),
    ),
)
