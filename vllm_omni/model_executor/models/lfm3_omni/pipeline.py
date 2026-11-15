# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The temporal backbone and feedback depth head share a single AR stage."""

from dataclasses import replace

from vllm_omni.config.stage_config import PipelineConfig, StageExecutionType, StagePipelineConfig

_PROC = "vllm_omni.model_executor.stage_input_processors.lfm3_omni"

LFM3_OMNI_PIPELINE = PipelineConfig(
    model_type="lfm3_omni",
    model_arch="Lfm3OmniForConditionalGeneration",
    default_deploy_config_name="lfm3_omni.yaml",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="lfm3_temporal",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            requires_multimodal_data=True,
            engine_output_type="latent",
            custom_process_next_stage_input_func=f"{_PROC}.temporal_to_codec_full_payload",
            async_chunk_process_next_stage_input_func=f"{_PROC}.temporal_to_codec_async_chunk",
            recompute_preemption="fail",
            sampling_constraints={"detokenize": True, "ignore_eos": True, "stop_token_ids": [124900]},
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="lfm3_code2wav",
            model_arch="Lfm3OmniCode2Wav",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            sync_process_input_func=f"{_PROC}.temporal_to_codec_token_only",
            requires_full_payload_input=True,
            sampling_constraints={"detokenize": False},
        ),
    ),
)

LFM3_OMNI_TEXT_PIPELINE = replace(
    LFM3_OMNI_PIPELINE,
    model_type="lfm3_omni_text",
    default_deploy_config_name="lfm3_omni_text.yaml",
    stages=(
        replace(
            LFM3_OMNI_PIPELINE.stages[0],
            sampling_constraints={"detokenize": True},
            custom_process_next_stage_input_func=None,
            async_chunk_process_next_stage_input_func=None,
        ),
    ),
)
