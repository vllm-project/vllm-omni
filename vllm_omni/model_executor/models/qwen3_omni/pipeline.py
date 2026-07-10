# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3-Omni-MoE pipeline topology (frozen).

Stage 0: Thinker — multimodal understanding + text generation
Stage 1: Talker  — text embeddings → RVQ codec codes
Stage 2: Code2Wav — RVQ codes → audio waveform
"""

from transformers import Qwen3OmniMoeConfig

from vllm_omni.config.endpoint_policy import EndpointRestriction, OmniServingCapability
from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
    pipeline_cfg_resolver,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.qwen3_omni"

QWEN3_OMNI_PIPELINE = PipelineConfig(
    model_type="qwen3_omni_moe",
    model_arch="Qwen3OmniMoeForConditionalGeneration",
    endpoint_restrictions=(
        EndpointRestriction(
            OmniServingCapability.COMPLETIONS,
            "Qwen3-Omni requires chat template structure for thinker-talker handoff. Use /v1/chat/completions instead.",
        ),
    ),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="thinker",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            requires_multimodal_data=True,
            hf_config_name="thinker_config",
            engine_output_type="latent",
            custom_process_next_stage_input_func=(f"{_PROC}.thinker2talker_full_payload"),
            async_chunk_process_next_stage_input_func=(f"{_PROC}.thinker2talker_async_chunk"),
            sampling_constraints={"detokenize": True},
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="talker",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(0,),
            hf_config_name="talker_config",
            engine_output_type="latent",
            custom_process_input_func=f"{_PROC}.thinker2talker",
            sync_process_input_func=f"{_PROC}.thinker2talker_token_only",
            custom_process_next_stage_input_func=(f"{_PROC}.talker2code2wav_full_payload"),
            async_chunk_process_next_stage_input_func=(f"{_PROC}.talker2code2wav_async_chunk"),
            sampling_constraints={
                "detokenize": False,
                "stop_token_ids": [2150],
            },
        ),
        StagePipelineConfig(
            stage_id=2,
            model_stage="code2wav",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(1,),
            final_output=True,
            final_output_type="audio",
            hf_config_name="thinker_config",
            engine_output_type="audio",
            custom_process_input_func=f"{_PROC}.talker2code2wav",
            sampling_constraints={"detokenize": True},
        ),
    ),
)

QWEN3_OMNI_THINKER_ONLY_PIPELINE = PipelineConfig(
    model_type="qwen3_omni_moe_thinker_only",
    model_arch="Qwen3OmniMoeForConditionalGeneration",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="thinker",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            requires_multimodal_data=True,
            hf_config_name="thinker_config",
            engine_output_type="latent",
            sampling_constraints={"detokenize": True},
        ),
    ),
)


@pipeline_cfg_resolver(config_type=Qwen3OmniMoeConfig)
def resolve_qwen3_omni_pipeline(
    hf_config: Qwen3OmniMoeConfig,
) -> PipelineConfig:
    """Select the right pipeline variant based on the HF config, since some variants,
    e.g., Qwen3-Omni-30B-A3B-Captioner, are thinker only.

    By default, we load the full pipeline, as this is the common case.
    """
    # If we have a config and it explicitly disabled audio input, load thinker only
    if not hf_config.enable_audio_output:
        return QWEN3_OMNI_THINKER_ONLY_PIPELINE
    return QWEN3_OMNI_PIPELINE
