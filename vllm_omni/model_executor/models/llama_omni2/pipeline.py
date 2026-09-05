# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""LLaMA-Omni 2 three-stage speech-to-speech pipeline."""

from vllm_omni.config.endpoint_policy import EndpointRestriction, OmniServingCapability
from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.llama_omni2"

LLAMA_OMNI2_PIPELINE = PipelineConfig(
    model_type="omni2_speech2s_qwen2",
    model_arch="Omni2Speech2SQwen2ForCausalLM",
    hf_architectures=("Omni2Speech2SQwen2ForCausalLM",),
    default_deploy_config_name="llama_omni2.yaml",
    endpoint_restrictions=(
        EndpointRestriction(
            OmniServingCapability.COMPLETIONS,
            "LLaMA-Omni 2 speech input requires chat message structure. Use /v1/chat/completions.",
        ),
    ),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="thinker",
            execution_type=StageExecutionType.LLM_AR,
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            requires_multimodal_data=True,
            hf_config_name="thinker_config",
            engine_output_type="latent",
            custom_process_next_stage_input_func=f"{_PROC}.thinker2talker_full_payload",
            async_chunk_process_next_stage_input_func=f"{_PROC}.thinker2talker_async_chunk",
            sampling_constraints={"detokenize": True},
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="talker",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(0,),
            hf_config_name="talker_config",
            engine_output_type="latent",
            sync_process_input_func=f"{_PROC}.thinker2talker_token_only",
            custom_process_next_stage_input_func=f"{_PROC}.talker2code2wav_full_payload",
            async_chunk_process_next_stage_input_func=f"{_PROC}.talker2code2wav_async_chunk",
            sampling_constraints={
                "detokenize": False,
                "stop_token_ids": [151643],
            },
        ),
        StagePipelineConfig(
            stage_id=2,
            model_stage="code2wav",
            model_arch="LlamaOmni2Code2Wav",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(1,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            sampling_constraints={"detokenize": False},
        ),
    ),
)
