# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Explicit-only Soprano pipeline: native AR -> continuous H -> FP32 audio."""

from vllm_omni.config.stage_config import PipelineConfig, StageExecutionType, StagePipelineConfig

_PROC = "vllm_omni.model_executor.stage_input_processors.soprano"
SOPRANO_PIPELINE = PipelineConfig(
    model_type="soprano",
    default_deploy_config_name="soprano.yaml",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="soprano_ar",
            model_arch="SopranoForCausalLM",
            owns_tokenizer=True,
            engine_output_type="latent",
            recompute_preemption="fail",
            prompt_transform_func="vllm_omni.model_executor.models.soprano.contracts.transform_prompt",
            custom_process_next_stage_input_func=f"{_PROC}.ar_to_decoder_full_payload",
            async_chunk_process_next_stage_input_func=f"{_PROC}.ar_to_decoder_async_chunk",
            sampling_constraints={"detokenize": False},
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="soprano_decoder",
            model_arch="SopranoDecoderForGeneration",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            sync_process_input_func=f"{_PROC}.ar_to_decoder_token_only",
            requires_full_payload_input=True,
            sampling_constraints={"detokenize": False},
        ),
    ),
)
